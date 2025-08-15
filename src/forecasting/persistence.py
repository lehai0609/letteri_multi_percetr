"""
Minimal persistence forecaster for Milestone 5.

Provides:
- persistence_forecast(train_df, test_index, method='last', drift_window=5)

Behavior / Contracts:
- Expects train_df to be a DataFrame with columns ['Open','High','Low','Close'] and a DatetimeIndex.
- Returns a DataFrame indexed by test_index with columns ['Open','High','Low','Close'].
- Does not access any future data beyond train_df (no leakage).
"""
from __future__ import annotations

from typing import Iterable, List, Literal, Optional

import numpy as np
import pandas as pd


OHLC = ["Open", "High", "Low", "Close"]


def _validate_inputs(train_df: pd.DataFrame, test_index: Iterable[pd.Timestamp]) -> pd.DatetimeIndex:
    if train_df is None or len(train_df) == 0:
        raise ValueError("train_df must be a non-empty DataFrame")
    for c in OHLC:
        if c not in train_df.columns:
            raise ValueError(f"train_df must contain '{c}' column")
    test_idx = pd.DatetimeIndex(test_index)
    return test_idx


def _last_values_forecast(train_df: pd.DataFrame, test_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Repeat the last observed OHLC bar for every forecast date.
    """
    last = train_df[OHLC].iloc[-1].astype(float)
    data = {c: np.repeat(float(last[c]), len(test_index)) for c in OHLC}
    return pd.DataFrame(data, index=test_index)[OHLC]


def _drift_forecast(train_df: pd.DataFrame, test_index: pd.DatetimeIndex, window: int = 5) -> pd.DataFrame:
    """
    Simple additive drift forecast: compute mean delta (difference) over the last `window`
    observations for each series and extrapolate linearly for each horizon step.

    Predicted value at step i: last + mean_delta * i
    """
    last = train_df[OHLC].iloc[-1].astype(float)
    mean_deltas = {}
    for c in OHLC:
        s = train_df[c].dropna().astype(float)
        if len(s) < 2:
            mean_deltas[c] = 0.0
        else:
            diffs = s.diff().dropna()
            # if diffs shorter than window, take what is available
            mean_deltas[c] = float(diffs.tail(window).mean()) if len(diffs) > 0 else 0.0

    rows = []
    for step in range(1, len(test_index) + 1):
        row = {c: float(last[c] + mean_deltas[c] * step) for c in OHLC}
        rows.append(row)
    return pd.DataFrame(rows, index=test_index)[OHLC]


def persistence_forecast(
    train_df: pd.DataFrame,
    test_index: Iterable[pd.Timestamp],
    *,
    method: Literal["last", "drift"] = "last",
    drift_window: int = 5,
) -> pd.DataFrame:
    """
    Produce a simple persistence forecast for OHLC series.

    Parameters
    ----------
    train_df:
        Historical DataFrame (index=trading dates) with OHLC columns.
    test_index:
        Iterable of timestamps for forecast dates (length = horizon).
    method:
        'last' - repeat the last observed bar for every forecast date.
        'drift' - simple linear drift using mean delta over last `drift_window` observations.
    drift_window:
        Number of last diffs to use when method == 'drift'.

    Returns
    -------
    pred_df: pd.DataFrame
        DataFrame indexed by test_index with columns ['Open','High','Low','Close'].
    """
    test_idx = _validate_inputs(train_df, test_index)

    if method == "last":
        return _last_values_forecast(train_df, test_idx)
    elif method == "drift":
        return _drift_forecast(train_df, test_idx, window=drift_window)
    else:
        raise ValueError(f"Unknown method '{method}'. Supported: 'last', 'drift'.")


def forecast_from_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    method: Literal["last", "drift"] = "last",
    drift_window: int = 5,
) -> pd.DataFrame:
    """
    Convenience wrapper: given train and test DataFrames (as produced by split_by_cutoff),
    produce a predictions DataFrame aligned to test_df.index.
    """
    return persistence_forecast(train_df, test_df.index, method=method, drift_window=drift_window)
