#!/usr/bin/env python3
"""
Generate 30-day forecasts (Milestone 8).

Behavior:
 - Attempts to load trained per-series DNN models and scalers from models_dir and run
   the recursive forecaster to produce a 30-day OHLC path.
 - If models or scalers are unavailable (or loading fails), falls back to the
   simple persistence forecaster (Milestone 5 behavior).
 - Writes `predicted_ohlc.csv` and `forecast_metrics.json` to the chosen output directory.

Usage:
    python scripts/forecast.py --config configs/default.yaml --out outputs/milestone8
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Dict

# Ensure repo root is on sys.path so `src` imports work when running the script directly.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import numpy as np
import pandas as pd

from src.data.split import split_by_cutoff
from src.data.scale import ScalerManager

# Recursive forecaster (Milestone 8)
from src.forecasting.recursive import forecast_from_split as recursive_forecast_from_split

# Persistence fallback (Milestone 5)
from src.forecasting.persistence import forecast_from_split as persistence_forecast_from_split

from src.metrics.forecast import compute_forecast_metrics
from src.utils.config import load_config

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


def _try_load_models(models_dir: str) -> Dict[str, object]:
    """
    Attempt to load DNN models for each OHLC series from models_dir.
    Returns a dict mapping series -> model if all models were successfully loaded.
    Raises exceptions on unrecoverable failures.
    """
    models = {}
    try:
        # Import the wrapper lazily: if TF is unavailable, DNNRegressor.load will raise and we'll fallback
        from src.models.dnn import DNNRegressor  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"DNN model support unavailable: {exc}")

    for col in OHLC:
        candidate = os.path.join(models_dir, f"dnn_{col}.keras")
        # DNNRegressor.load is robust to .keras/.h5 variants and directories
        if not os.path.exists(candidate) and not os.path.exists(candidate + ".keras") and not os.path.exists(
            os.path.join(models_dir, "dnn_" + col)
        ):
            raise FileNotFoundError(f"Model file for {col} not found at expected location: {candidate}")
        try:
            mdl = DNNRegressor.load(candidate)
            models[col] = mdl
        except Exception as exc:
            raise RuntimeError(f"Failed to load model for {col} from {candidate}: {exc}")
    return models


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    p.add_argument("--models", default=None, help="Optional models directory override")
    p.add_argument("--out", default="outputs/milestone8", help="Output directory to save forecasts")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    # Load config (best-effort; if it fails we fall back to reasonable defaults)
    try:
        cfg = load_config(args.config)
        logging.info("Loaded config %s", args.config)
    except Exception as exc:
        logging.warning("Could not load config %s: %s. Falling back to defaults.", args.config, exc)
        cfg = None

    lag_t = int(cfg["data"]["lag_t"]) if cfg is not None else 5
    horizon = int(cfg["data"]["horizon_days"]) if cfg is not None else 30
    models_dir = args.models or (cfg["io"]["models_dir"] if cfg is not None else "models")

    # Prepare data (synthetic path for Milestone 8)
    df = generate_synthetic_ohlc(n_days=120, start=(cfg["data"].get("start_date") if cfg else "2020-01-01"), seed=1234)
    logging.info("Generated synthetic OHLC with %d rows", len(df))

    # Split train/test (reuse the same fallback cutoff logic as other scripts)
    cutoff_idx = 89
    try:
        cutoff_ts = df.index[cutoff_idx]
        train_df, test_df = split_by_cutoff(df, cutoff_ts.isoformat(), horizon=horizon)
    except Exception as exc:
        logging.warning("split_by_cutoff failed (%s). Falling back to positional split.", exc)
        train_df = df.iloc[: cutoff_idx + 1].copy()
        test_df = df.iloc[cutoff_idx + 1 : cutoff_idx + 1 + horizon].copy()
        if len(test_df) != horizon:
            raise RuntimeError("Not enough rows for test window in fallback split.")
    logging.info("Train rows: %d, Test rows: %d", len(train_df), len(test_df))

    # Attempt to load scalers and models and run recursive forecaster
    use_recursive = False
    pred_df = None

    scaler_path = os.path.join(models_dir, "scalers.joblib")
    if os.path.exists(scaler_path):
        try:
            scaler_mgr = ScalerManager()
            scaler_mgr.load(scaler_path)
            logging.info("Loaded scalers from %s", scaler_path)
            # Try loading all models
            try:
                models = _try_load_models(models_dir)
                logging.info("Loaded models for series: %s", ", ".join(sorted(models.keys())))
                # Run recursive forecaster
                pred_df = recursive_forecast_from_split(models, scaler_mgr, train_df, test_df, lag_t)
                use_recursive = True
            except Exception as exc:
                logging.warning("Could not load models/scalers for recursive forecasting: %s", exc)
                use_recursive = False
        except Exception as exc:
            logging.warning("Failed to load scaler file %s: %s", scaler_path, exc)
            use_recursive = False
    else:
        logging.info("No scalers found at %s; falling back to persistence forecast", scaler_path)

    if not use_recursive:
        logging.info("Using persistence forecaster (no trained models available)")
        pred_df = persistence_forecast_from_split(train_df, test_df, method="last")

    # Sanity: ensure columns and index match contract
    pred_df = pred_df[OHLC].astype(float)
    pred_csv = os.path.join(out_dir, "predicted_ohlc.csv")
    pred_df.to_csv(pred_csv)
    logging.info("Wrote forecast to %s (method=%s)", pred_csv, "recursive" if use_recursive else "persistence")

    # Compute and save forecast metrics
    try:
        fmetrics = compute_forecast_metrics(test_df, pred_df)
        metrics_path = os.path.join(out_dir, "forecast_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(fmetrics, f, indent=2, default=str)
        logging.info("Saved forecast metrics to %s", metrics_path)
    except Exception as exc:
        logging.warning("Failed to compute/save forecast metrics: %s", exc)

    # Print concise summary
    print("Forecast saved to:", pred_csv)
    if cfg is not None:
        print("Config used:", args.config)
    print("Method:", "recursive" if use_recursive else "persistence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
