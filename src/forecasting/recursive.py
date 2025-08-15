"""
Recursive forecaster for multi-step OHLC prediction (Milestone 8).

This module provides:
- recursive_forecast(models, scaler_mgr, train_df, test_index, lag_t)
  which uses one-step models (per-series regressors) in a recursive loop to
  produce a multi-day OHLC path without leaking future data.

Contracts:
- `models` is a dict-like mapping series name -> model with a .predict(X) method.
  The model.predict(X) must accept a 2D numpy array shaped (n_samples, lag_t)
  and return a 1D array-like of length n_samples (or shape (n_samples,1)).
- `scaler_mgr` is a ScalerManager (src.data.scale.ScalerManager) with fitted scalers
  for each OHLC series (keys: "Open","High","Low","Close").
- `train_df` is a pandas DataFrame with OHLC columns and a DatetimeIndex.
- `test_index` is an iterable of pd.Timestamp for the forecast horizon.
- `lag_t` is the integer input window size used by the models (t).

The returned DataFrame is indexed by `test_index` and has columns:
["Open","High","Low","Close"] (dtype float).
"""
from __future__ import annotations

from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd

from src.data.scale import ScalerManager

OHLC = ["Open", "High", "Low", "Close"]


def _validate_inputs(train_df: pd.DataFrame, test_index: Iterable[pd.Timestamp], lag_t: int) -> pd.DatetimeIndex:
    if train_df is None or len(train_df) == 0:
        raise ValueError("train_df must be a non-empty DataFrame")
    for c in OHLC:
        if c not in train_df.columns:
            raise ValueError(f"train_df must contain '{c}' column")
    if lag_t <= 0:
        raise ValueError("lag_t must be a positive integer")
    test_idx = pd.DatetimeIndex(test_index)
    return test_idx


def recursive_forecast(
    models: Dict[str, Any],
    scaler_mgr: ScalerManager,
    train_df: pd.DataFrame,
    test_index: Iterable[pd.Timestamp],
    lag_t: int,
) -> pd.DataFrame:
    """
    Produce a recursive multi-step OHLC forecast.

    Parameters
    ----------
    models:
        Mapping from series name ("Open","High","Low","Close") to a fitted model
        object exposing `predict(X)` where X has shape (n_samples, lag_t).
    scaler_mgr:
        ScalerManager with fitted scalers for each series (same names as above).
    train_df:
        Historical DataFrame with OHLC columns (index=trading dates).
    test_index:
        Iterable of timestamps for forecast dates (length = horizon).
    lag_t:
        Window size used by models.

    Returns
    -------
    pred_df: pd.DataFrame
        DataFrame indexed by test_index with columns ["Open","High","Low","Close"].
    """
    test_idx = _validate_inputs(train_df, test_index, lag_t)

    # Basic sanity checks for models and scalers
    for c in OHLC:
        if c not in models:
            raise KeyError(f"Model for '{c}' not provided in `models`")
        if not scaler_mgr.has(c):
            raise KeyError(f"Scaler for '{c}' not found in ScalerManager")

    # Initialize scaled rolling windows from the last `lag_t` training observations
    last_windows: Dict[str, np.ndarray] = {}
    for c in OHLC:
        series = train_df[c].astype(float)
        scaled_series = scaler_mgr.transform(series, name=c)  # returns pandas Series
        if len(scaled_series) < lag_t:
            raise RuntimeError(f"Not enough history for series '{c}' to build lag window (have {len(scaled_series)}, need {lag_t})")
        last_windows[c] = scaled_series.values[-lag_t:].astype(float).copy()

    rows = []
    # Iterate horizons in chronological order; no access to future actuals
    for _d in test_idx:
        # Predict in scaled space for each series
        preds_scaled: Dict[str, float] = {}
        for c in OHLC:
            x = last_windows[c].reshape(1, -1)  # shape (1, lag_t)
            yhat = models[c].predict(x)
            yhat_arr = np.asarray(yhat).reshape(-1)
            if yhat_arr.size == 0:
                raise RuntimeError(f"Model for '{c}' returned empty prediction")
            preds_scaled[c] = float(yhat_arr[0])

        # Inverse-transform predicted scaled values to original price scale
        row = {}
        for c in OHLC:
            inv = scaler_mgr.inverse_transform(np.array([preds_scaled[c]]), name=c)
            # inverse_transform returns a pandas Series
            row[c] = float(inv.iloc[0])

        rows.append(row)

        # Roll the scaled windows forward by appending the scaled prediction
        for c in OHLC:
            w = last_windows[c]
            w = np.concatenate([w[1:], np.array([preds_scaled[c]], dtype=float)])
            last_windows[c] = w

    pred_df = pd.DataFrame(rows, index=test_idx)[OHLC]
    return pred_df


def forecast_from_split(
    models: Dict[str, Any],
    scaler_mgr: ScalerManager,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    lag_t: int,
) -> pd.DataFrame:
    """
    Convenience wrapper: forecast using train/test DataFrames and return DataFrame
    aligned to test_df.index.
    """
    return recursive_forecast(models, scaler_mgr, train_df, test_df.index, lag_t)
