#!/usr/bin/env python3
"""
Milestone 5 runner: walking skeleton end-to-end using a persistence forecaster.

Flow:
- generate synthetic OHLC data
- (optional) adjust prices using src.data.adjust.adjust_ohlc if available
- split into train/test (30 trading days)
- forecast 30-day OHLC using persistence forecaster
- generate threshold and triple-TEMA signals from predictions
- backtest each signal stream on the actual test window
- compute forecast metrics and backtest/risk metrics
- save results to outputs/milestone5/
"""
from __future__ import annotations

import sys
import os

# Ensure the project root is on sys.path so "src" imports work when running the script directly.
# This mirrors common practice for repository-local scripts.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import json
import logging
from typing import Dict, List

import numpy as np
import pandas as pd

# Import pipeline pieces from src/
from src.data.split import split_by_cutoff
from src.forecasting.persistence import forecast_from_split
from src.backtest.engine import BacktestEngine
from src.metrics.forecast import compute_forecast_metrics
from src.metrics.risk import compute_backtest_metrics

# optional adjust function - guard import
try:
    from src.data.adjust import adjust_ohlc  # type: ignore
except Exception:
    adjust_ohlc = None  # type: ignore


OHLC = ["Open", "High", "Low", "Close"]


def generate_synthetic_ohlc(
    n_days: int = 120, start: str = "2020-01-01", seed: int = 1234
) -> pd.DataFrame:
    """
    Generate a deterministic synthetic OHLC DataFrame with business-day index.

    Returns columns: Open, High, Low, Close, Adj Close, Volume
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_days)
    # generate log returns with small drift
    rets = rng.normal(loc=0.0002, scale=0.01, size=n_days)
    close = 100.0 * np.exp(np.cumsum(rets))
    # open is previous close (for day 0, open == close[0])
    open_ = np.empty_like(close)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    # highs / lows around open/close with some noise
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
    # ensure float dtype and a clean index
    return df.astype(float)


def generate_threshold_signals(pred_df: pd.DataFrame) -> pd.Series:
    """
    Stateful threshold strategy on predicted OHLC:
    - Entry: if predicted Close > predicted Open -> enter long
    - Exit: if predicted Close <= predicted Open -> exit
    - Signals are pre-open decisions for each forecast date.
    Returns pd.Series of int {0,1} indexed by pred_df.index with name 'signal'
    """
    sig = pd.Series(index=pred_df.index, dtype=int, name="signal")
    holding = False
    for dt in pred_df.index:
        c = float(pred_df.loc[dt, "Close"])
        o = float(pred_df.loc[dt, "Open"])
        if not holding and c > o:
            holding = True
            sig.loc[dt] = 1
        elif holding and c <= o:
            holding = False
            sig.loc[dt] = 0
        else:
            sig.loc[dt] = 1 if holding else 0
    return sig


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def tema(series: pd.Series, span: int) -> pd.Series:
    e1 = ema(series, span)
    e2 = ema(e1, span)
    e3 = ema(e2, span)
    return 3.0 * e1 - 3.0 * e2 + e3


def generate_triple_ema_signals(pred_df: pd.DataFrame, period: int = 3) -> pd.Series:
    """
    Triple EMA rule from TDD with explicit operator precedence:
    entry = ((Low < TEMA(Low)) or (High < TEMA(High))) and ((Close < TEMA(Close)) or (Open < TEMA(Open)))
    exit  = ((Low > TEMA(Low)) or (High > TEMA(High))) and ((Close > TEMA(Close)) or (Open > TEMA(Open)))
    """
    tema_df = pd.DataFrame(index=pred_df.index, columns=OHLC, dtype=float)
    for col in OHLC:
        tema_df[col] = tema(pred_df[col], period)

    sig = pd.Series(index=pred_df.index, dtype=int, name="signal")
    holding = False
    for dt in pred_df.index:
        low = float(pred_df.loc[dt, "Low"])
        high = float(pred_df.loc[dt, "High"])
        close = float(pred_df.loc[dt, "Close"])
        open_ = float(pred_df.loc[dt, "Open"])

        t_low = float(tema_df.loc[dt, "Low"])
        t_high = float(tema_df.loc[dt, "High"])
        t_close = float(tema_df.loc[dt, "Close"])
        t_open = float(tema_df.loc[dt, "Open"])

        entry_cond = ((low < t_low) or (high < t_high)) and ((close < t_close) or (open_ < t_open))
        exit_cond = ((low > t_low) or (high > t_high)) and ((close > t_close) or (open_ > t_open))

        if not holding and entry_cond:
            holding = True
            sig.loc[dt] = 1
        elif holding and exit_cond:
            holding = False
            sig.loc[dt] = 0
        else:
            sig.loc[dt] = 1 if holding else 0
    return sig


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def trades_to_serializable(trades: List[Dict]) -> List[Dict]:
    out = []
    for tr in trades:
        tr2 = dict(tr)
        d = tr2.get("date")
        # convert Timestamp to ISO string if present
        try:
            tr2["date"] = pd.Timestamp(d).isoformat()
        except Exception:
            tr2["date"] = str(d)
        out.append(tr2)
    return out


def main(output_dir: str = "outputs/milestone5"):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.makedirs(output_dir, exist_ok=True)

    # 1) Generate data
    df = generate_synthetic_ohlc(n_days=120, start="2020-01-01", seed=1234)
    logging.info("Generated synthetic OHLC with %d rows", len(df))

    # 2) Optional adjust (no-op if adjust_ohlc unavailable)
    if adjust_ohlc is not None:
        try:
            df = adjust_ohlc(df)
            logging.info("Applied price adjustments using adjust_ohlc")
        except Exception as exc:
            logging.warning("adjust_ohlc failed, proceeding with raw data: %s", exc)

    # 3) Split into train/test with a 30-day horizon.
    # Choose cutoff so there are at least 30 trading days after it.
    # We'll pick the 90th row as cutoff (0-based index 89)
    cutoff_idx = 89
    cutoff_ts = df.index[cutoff_idx]
    try:
        train_df, test_df = split_by_cutoff(df, cutoff_ts.isoformat(), horizon=30)
    except Exception as exc:
        logging.warning("split_by_cutoff failed (%s). Falling back to positional split.", exc)
        train_df = df.iloc[: cutoff_idx + 1].copy()
        test_df = df.iloc[cutoff_idx + 1 : cutoff_idx + 1 + 30].copy()
        if len(test_df) != 30:
            raise RuntimeError("Not enough rows for test window in fallback split.")

    logging.info("Train rows: %d, Test rows: %d", len(train_df), len(test_df))

    # 4) Forecast using persistence forecaster
    pred_df = forecast_from_split(train_df, test_df, method="last")
    pred_csv = os.path.join(output_dir, "predicted_ohlc.csv")
    pred_df.to_csv(pred_csv)
    logging.info("Wrote persistence forecast to %s", pred_csv)

    # 5) Forecast metrics
    fmetrics = compute_forecast_metrics(test_df, pred_df)
    save_json(fmetrics, os.path.join(output_dir, "forecast_metrics.json"))
    logging.info("Saved forecast metrics")

    # 6) Generate signals
    sig_threshold = generate_threshold_signals(pred_df)
    sig_tema = generate_triple_ema_signals(pred_df, period=3)

    # 7) Backtest both strategies
    engine = BacktestEngine(
        initial_cash=100.0,
        cost_bps=10.0,
        slippage_bps=5.0,
        allow_fractional=True,
        execution_price="open",
        mark_to_market="close",
    )

    trades_th, equity_th = engine.run(test_df, sig_threshold)
    trades_te, equity_te = engine.run(test_df, sig_tema)

    # 8) Compute backtest/risk metrics
    metrics_th = compute_backtest_metrics(equity_th, trades_th, rf=0.0)
    metrics_te = compute_backtest_metrics(equity_te, trades_te, rf=0.0)

    # 9) Save outputs
    equity_th.to_csv(os.path.join(output_dir, "equity_threshold.csv"))
    equity_te.to_csv(os.path.join(output_dir, "equity_tema.csv"))

    save_json(metrics_th, os.path.join(output_dir, "threshold_backtest_metrics.json"))
    save_json(metrics_te, os.path.join(output_dir, "tema_backtest_metrics.json"))

    save_json(trades_to_serializable(trades_th), os.path.join(output_dir, "threshold_trades.json"))
    save_json(trades_to_serializable(trades_te), os.path.join(output_dir, "tema_trades.json"))

    logging.info("Saved backtest outputs to %s", output_dir)

    # Print concise summary
    print("Forecast summary (summary across series):")
    print(json.dumps(fmetrics.get("summary", {}), indent=2))
    print("\nThreshold strategy metrics summary:")
    print(json.dumps(metrics_th, indent=2))
    print("\nTriple-TEMA strategy metrics summary:")
    print(json.dumps(metrics_te, indent=2))


if __name__ == "__main__":
    main()
