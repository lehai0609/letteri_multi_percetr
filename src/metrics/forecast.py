"""
Forecast metrics for the Letteri replication.

Provides:
- mae, rmse, mape, explained_variance_score
- compute_forecast_metrics(actual_df, pred_df) -> per-series metrics + aggregated summary

All functions are defensive about empty inputs and alignment by index.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


def _align_arrays(a: pd.Series, b: pd.Series):
    """
    Align two series by intersection of their indices, drop pairs with NaNs,
    and return as numpy float arrays.
    """
    if a is None or b is None:
        return np.array([]), np.array([])
    idx = a.index.intersection(b.index)
    if len(idx) == 0:
        return np.array([]), np.array([])
    a2 = a.loc[idx].astype(float)
    b2 = b.loc[idx].astype(float)
    mask = (~a2.isna()) & (~b2.isna())
    if mask.sum() == 0:
        return np.array([]), np.array([])
    return a2.loc[mask].to_numpy(dtype=float), b2.loc[mask].to_numpy(dtype=float)


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Mean Absolute Error"""
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Root Mean Squared Error"""
    if len(y_true) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(np.asarray(y_true) - np.asarray(y_pred)))))


def mape(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """
    Mean Absolute Percentage Error.

    This implementation ignores samples where the true value is (near) zero.
    If all true values are effectively zero, returns NaN.
    Returns fractional MAPE (e.g., 0.05 == 5%).
    """
    if len(y_true) == 0:
        return float("nan")
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom_mask = np.abs(y_true) > 1e-8
    if not np.any(denom_mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[denom_mask] - y_pred[denom_mask]) / y_true[denom_mask])))


def explained_variance_score(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """
    Explained variance score: 1 - Var(y - y_pred) / Var(y)

    Returns 1.0 for perfect predictions. If Var(y) == 0:
      - returns 1.0 if predictions are identical to y
      - returns 0.0 otherwise
    """
    if len(y_true) == 0:
        return float("nan")
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    var_true = float(np.var(y_true, ddof=0))
    var_diff = float(np.var(y_true - y_pred, ddof=0))
    if var_true == 0.0:
        return 1.0 if var_diff == 0.0 else 0.0
    return float(1.0 - (var_diff / var_true))


def compute_forecast_metrics(
    actual_df: pd.DataFrame, pred_df: pd.DataFrame, series: List[str] | None = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute forecast metrics (MAE, RMSE, MAPE, EVS) for each series (Open, High, Low, Close).

    Returns a dictionary:
      {
        "Open": {"mae": ..., "rmse": ..., "mape": ..., "evs": ...},
        "High": {...},
        "Low": {...},
        "Close": {...},
        "summary": {"mae": ..., "rmse": ..., "mape": ..., "evs": ...}  # mean across series (ignoring NaNs)
      }

    Raises ValueError if required columns are missing.
    """
    if series is None:
        series = ["Open", "High", "Low", "Close"]

    missing_actual = [s for s in series if s not in actual_df.columns]
    missing_pred = [s for s in series if s not in pred_df.columns]
    if missing_actual:
        raise ValueError(f"actual_df is missing required columns: {missing_actual}")
    if missing_pred:
        raise ValueError(f"pred_df is missing required columns: {missing_pred}")

    results: Dict[str, Dict[str, float]] = {}
    maes: List[float] = []
    rmses: List[float] = []
    mapes: List[float] = []
    evss: List[float] = []

    for s in series:
        y_true, y_pred = _align_arrays(actual_df[s], pred_df[s])
        if len(y_true) == 0:
            # no data to score
            res = {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan"), "evs": float("nan")}
        else:
            res = {
                "mae": mae(y_true, y_pred),
                "rmse": rmse(y_true, y_pred),
                "mape": mape(y_true, y_pred),
                "evs": explained_variance_score(y_true, y_pred),
            }
        results[s] = res
        maes.append(res["mae"])
        rmses.append(res["rmse"])
        mapes.append(res["mape"])
        evss.append(res["evs"])

    # compute mean of metrics across series, ignoring NaNs
    def _nanmean(lst: Sequence[float]) -> float:
        arr = np.asarray(lst, dtype=float)
        if np.all(np.isnan(arr)):
            return float("nan")
        return float(np.nanmean(arr))

    results["summary"] = {
        "mae": _nanmean(maes),
        "rmse": _nanmean(rmses),
        "mape": _nanmean(mapes),
        "evs": _nanmean(evss),
    }
    return results
