"""
Feature windowing utilities for Milestone 6.

Provides:
- build_xy(series, t) -> (X, y)
  Builds a supervised dataset where each row of X is `t` consecutive values
  from the input series and y is the immediately following value.

  For a series of length N and window t, the number of samples is max(0, N - t).
  X shape = (N - t, t), y shape = (N - t,)

- last_window(series, t) -> np.ndarray
  Return the last t values from the series as a 1-D numpy array (used for
  recursive forecasting initialization).
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import pandas as pd


def _to_float_array(series: Sequence) -> np.ndarray:
    """
    Convert input to a 1-D float numpy array.

    - If a pandas Series is provided, drop NA values and convert to float.
    - If an array-like is provided, coerce to numpy float array.
    """
    if isinstance(series, pd.Series):
        arr = series.dropna().astype(float).to_numpy()
    else:
        arr = np.asarray(series, dtype=float)
    if arr.ndim != 1:
        arr = arr.flatten()
    return arr


def build_xy(series: Sequence, t: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build supervised (X, y) dataset from a 1-D series.

    Parameters
    ----------
    series : Sequence
        1-D array-like of numeric values (pandas Series or numpy array).
    t : int
        Window length (number of lagged inputs).

    Returns
    -------
    X : np.ndarray, shape (n_samples, t)
    y : np.ndarray, shape (n_samples,)

    Notes
    -----
    If the series length N <= t, returns empty arrays with shapes (0, t) and (0,).
    """
    if t <= 0:
        raise ValueError("t must be a positive integer")

    arr = _to_float_array(series)
    n = arr.shape[0]
    if n <= t:
        return np.empty((0, t), dtype=float), np.empty((0,), dtype=float)

    n_samples = n - t
    X = np.empty((n_samples, t), dtype=float)
    for i in range(n_samples):
        X[i, :] = arr[i : i + t]
    y = arr[t : t + n_samples].astype(float)
    return X, y


def last_window(series: Sequence, t: int) -> np.ndarray:
    """
    Return the last `t` values from the series as a 1-D numpy array.

    Raises ValueError if the series has fewer than t non-NA values.
    """
    if t <= 0:
        raise ValueError("t must be a positive integer")

    arr = _to_float_array(series)
    if arr.shape[0] < t:
        raise ValueError(f"series must contain at least {t} values to build last_window")
    return arr[-t:].astype(float).copy()
