#!/usr/bin/env python3
# Run backtests from predicted OHLC and produce reports (Milestone 11).
#
# Behavior:
#  - Loads predicted OHLC from a provided CSV or from the configured forecasts directory.
#  - If no predictions are available, falls back to the persistence forecaster using a
#    synthetic dataset (for CLI smoke tests / reproducibility).
#  - Generates signals for configured strategies (threshold, Triple-TEMA), runs the
#    BacktestEngine, computes metrics, and assembles a report using src.reporting.report.build_report.
#  - Writes outputs (predicted_ohlc.csv, forecast_metrics.json, equity_*.csv, metrics_*.json, trades_*.json, plots)
#    to the configured reports directory or an --out override.
#
# Usage:
#     python scripts/backtest.py --config configs/default.yaml --out outputs/milestone11

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Tuple

# Ensure repo root is on sys.path so `src` imports work when running the script directly.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.data.split import split_by_cutoff
from src.forecasting.persistence import forecast_from_split
from src.backtest.engine import BacktestEngine
from src.metrics.forecast import compute_forecast_metrics
from src.metrics.risk import compute_backtest_metrics
from src.strategies.threshold import generate_signals as generate_threshold_signals
from src.strategies.triple_ema import generate_signals as generate_tema_signals
from src.reporting.report import build_report

OHLC = ["Open", "High", "Low", "Close"]


def generate_synthetic_ohlc(n_days: int = 120, start: str = "2020-01-01", seed: int = 1234) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_days)
    rets = rng.normal(loc=0.0002, scale=0.01, size=n_days)
    close = 100.0 * np.exp(np.cumsum(rets))
    open_ = np.empty_like(close)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    highs = np.maximum(open_, close) + np.abs(rng.normal(0, 0.2, size=n_days))
    lows = np.minimum(open_, close) - np.abs(rng.normal(0, 0.2, size=n_days))
    adj_close = close.copy()
    volume = rng.integers(100_000, 1_000_000, size=n_days)

    df = pd.DataFrame(
        {
            "Open": open_,
            "High": highs,
            "Low": lows,
            "Close": close,
            "Adj Close": adj_close,
            "Volume": volume,
        },
        index=idx,
    )
    return df.astype(float)


def _try_load_predictions(pred_path: str) -> pd.DataFrame:
    if not os.path.exists(pred_path):
        raise FileNotFoundError(pred_path)
    df = pd.read_csv(pred_path, index_col=0, parse_dates=True)
    # Ensure required columns present
    missing = [c for c in OHLC if c not in df.columns]
    if missing:
        raise ValueError(f"Predicted OHLC missing columns: {missing}")
    return df[OHLC].astype(float)


def _find_forecast_in_cfg(cfg: Dict) -> Tuple[str, bool]:
    # Prefer explicit forecasts_dir in config; try common outputs location as fallback
    fc_dir = cfg["io"].get("forecasts_dir", "forecasts")
    p = os.path.join(fc_dir, "predicted_ohlc.csv")
    if os.path.exists(p):
        return p, True
    # Fallback to outputs/milestone8 used elsewhere in repo
    alt = os.path.join("outputs", "milestone8", "predicted_ohlc.csv")
    if os.path.exists(alt):
        return alt, True
    return p, False


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    p.add_argument("--pred", default=None, help="Optional predicted_ohlc.csv path override")
    p.add_argument("--out", default=None, help="Output reports directory (overrides config.io.reports_dir)")
    p.add_argument("--seed", default=1234, type=int, help="Seed for synthetic data (used if no predictions found)")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        cfg = load_config(args.config)
        logging.info("Loaded config %s", args.config)
    except Exception as exc:
        logging.warning("Could not load config %s: %s. Falling back to defaults.", args.config, exc)
        cfg = None

    out_dir = args.out or (cfg["io"]["reports_dir"] if cfg is not None else "reports")
    os.makedirs(out_dir, exist_ok=True)

    # Prepare data (synthetic path for smoke tests)
    df = generate_synthetic_ohlc(n_days=120, start=(cfg["data"].get("start_date") if cfg else "2020-01-01"), seed=int(args.seed))
    logging.info("Generated synthetic OHLC with %d rows", len(df))

    # Split into train/test using cutoff index consistent with other scripts (cutoff_idx=89 -> 30-day test)
    cutoff_idx = 89
    try:
        cutoff_ts = df.index[cutoff_idx]
        train_df, test_df = split_by_cutoff(df, cutoff_ts.isoformat(), horizon=(int(cfg["data"]["horizon_days"]) if cfg else 30))
    except Exception as exc:
        logging.warning("split_by_cutoff failed (%s). Falling back to positional split.", exc)
        train_df = df.iloc[: cutoff_idx + 1].copy()
        test_df = df.iloc[cutoff_idx + 1 : cutoff_idx + 1 + (int(cfg["data"]["horizon_days"]) if cfg else 30)].copy()
        if len(test_df) != (int(cfg["data"]["horizon_days"]) if cfg else 30):
            raise RuntimeError("Not enough rows for test window in fallback split.")

    logging.info("Train rows: %d, Test rows: %d", len(train_df), len(test_df))

    # Locate predictions
    pred_df = None
    method = "persistence"
    if args.pred:
        try:
            pred_df = _try_load_predictions(args.pred)
            method = f"loaded:{args.pred}"
            logging.info("Loaded predictions from %s", args.pred)
        except Exception as exc:
            logging.warning("Failed to load predictions from %s: %s", args.pred, exc)

    if pred_df is None and cfg is not None:
        candidate, ok = _find_forecast_in_cfg(cfg)
        if ok:
            try:
                pred_df = _try_load_predictions(candidate)
                method = f"loaded:{candidate}"
                logging.info("Loaded predictions from %s", candidate)
            except Exception as exc:
                logging.warning("Failed to load predictions from %s: %s", candidate, exc)

    # If still no predictions available, fall back to persistence forecaster for smoke tests
    if pred_df is None:
        logging.info("No predicted OHLC found; using persistence forecaster for smoke run")
        pred_df = forecast_from_split(train_df, test_df, method="last")
        method = "persistence"

    # Sanity ensure columns and index are correct
    pred_df = pred_df[OHLC].astype(float)
    # Compute forecast metrics
    try:
        fmetrics = compute_forecast_metrics(test_df, pred_df)
    except Exception as exc:
        logging.warning("Failed to compute forecast metrics: %s", exc)
        fmetrics = None

    # Prepare backtest engine from config defaults
    if cfg is not None:
        bt_cfg = cfg["backtest"]
        initial_cash = float(bt_cfg.get("initial_cash", 100.0))
        cost_bps = float(bt_cfg.get("cost_bps", 10.0))
        slippage_bps = float(bt_cfg.get("slippage_bps", 5.0))
        allow_fractional = bool(bt_cfg.get("allow_fractional", True))
        execution_price = str(bt_cfg.get("execution_price", "open"))
        mark_to_market = str(bt_cfg.get("mark_to_market", "close"))
    else:
        initial_cash = 100.0
        cost_bps = 10.0
        slippage_bps = 5.0
        allow_fractional = True
        execution_price = "open"
        mark_to_market = "close"

    engine = BacktestEngine(
        initial_cash=initial_cash,
        cost_bps=cost_bps,
        slippage_bps=slippage_bps,
        allow_fractional=allow_fractional,
        execution_price=execution_price,
        mark_to_market=mark_to_market,
    )

    # Strategy list: threshold and triple-TEMA
    strategy_results: Dict[str, Dict] = {}

    # Threshold strategy
    try:
        sig_thresh = generate_threshold_signals(pred_df)
        trades_th, equity_th = engine.run(test_df, sig_thresh)
        metrics_th = compute_backtest_metrics(equity_th, trades_th, rf=0.0)
        strategy_results["Threshold"] = {"trades": trades_th, "equity": equity_th, "metrics": metrics_th}
        logging.info("Threshold strategy: trades=%d final_equity=%.4f", len(trades_th), float(equity_th.iloc[-1]))
    except Exception as exc:
        logging.exception("Threshold strategy/backtest failed: %s", exc)

    # Triple-TEMA strategy (use config-specified period / operator_precedence if available)
    try:
        if cfg is not None:
            tema_period = int(cfg["model"].get("tema_period", 3))
            operator_precedence = str(cfg["model"].get("operator_precedence"))
        else:
            tema_period = 3
            operator_precedence = None
        # triple_ema.generate_signals accepts period and operator_precedence arguments
        sig_tema = generate_tema_signals(
            pred_df,
            period=tema_period,
            operator_precedence=operator_precedence,
            cfg_path=(args.config if cfg is not None else "configs/default.yaml"),
        )
        trades_te, equity_te = engine.run(test_df, sig_tema)
        metrics_te = compute_backtest_metrics(equity_te, trades_te, rf=0.0)
        strategy_results["Triple-TEMA"] = {"trades": trades_te, "equity": equity_te, "metrics": metrics_te}
        logging.info("Triple-TEMA strategy: trades=%d final_equity=%.4f", len(trades_te), float(equity_te.iloc[-1]))
    except Exception as exc:
        logging.exception("Triple-TEMA strategy/backtest failed: %s", exc)

    # Build and persist report
    try:
        created = build_report(out_dir, pred_df=pred_df, actual_df=test_df, forecast_metrics=fmetrics, strategy_results=strategy_results)
        logging.info("Report created: %s", json.dumps(created, indent=2))
        print("Report files created:")
        for k, v in created.items():
            print(f"  {k}: {v}")
    except Exception as exc:
        logging.exception("Failed to build report in %s: %s", out_dir, exc)
        return 2

    # Print concise summary
    print("\nBacktest summary:")
    print(f"  Prediction source/method: {method}")
    if fmetrics is not None:
        print("  Forecast summary:", json.dumps(fmetrics.get("summary", {}), indent=2))
    for name, res in strategy_results.items():
        metrics = res.get("metrics", {})
        print(f"\n  Strategy: {name}")
        print(json.dumps(metrics, indent=2, default=str))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
