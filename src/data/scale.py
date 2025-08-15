"""
Scaler manager for per-series scaling.

Provides:
- ScalerManager: fit/transform/inverse per-series, save/load all scalers.

Supports sklearn StandardScaler and MinMaxScaler (configured by scaler_type).
Designed to avoid leakage: caller must fit only on training data.

API:
    s = ScalerManager(scaler_type="standard")
    s.fit(series, name="Close")
    xs = s.transform(series, name="Close")
    orig = s.inverse_transform(xs, name="Close")
    s.save("models/scalers.joblib")
    s.load("models/scalers.joblib")
"""
from __future__ import annotations

import os
import logging
from typing import Dict, Union, Optional

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler  # type: ignore

Logger = logging.getLogger(__name__)


class ScalerManager:
    def __init__(self, *, scaler_type: str = "standard") -> None:
        """
        Args:
            scaler_type: "standard" or "minmax"
        """
        scaler_type = scaler_type.lower()
        if scaler_type not in {"standard", "minmax"}:
            raise ValueError("scaler_type must be 'standard' or 'minmax'")
        self.scaler_type = scaler_type
        self._scalers: Dict[str, object] = {}

    def _make_scaler(self):
        if self.scaler_type == "standard":
            return StandardScaler()
        return MinMaxScaler()

    def fit(self, series: pd.Series, name: str) -> None:
        """
        Fit a scaler for the provided series and store it under `name`.

        The scaler expects 2D arrays (n_samples, 1) so series values are reshaped.
        """
        if not isinstance(series, pd.Series):
            raise TypeError("series must be a pandas Series")
        values = series.dropna().values.reshape(-1, 1)
        if values.size == 0:
            raise ValueError("Cannot fit scaler on empty series")
        scaler = self._make_scaler()
        scaler.fit(values)
        self._scalers[name] = scaler
        Logger.debug("Fitted scaler '%s' of type %s", name, type(scaler).__name__)

    def transform(self, series: pd.Series, name: str) -> pd.Series:
        """
        Transform a series using the named scaler and return a Series aligned to the input index.

        Raises KeyError if scaler not found.
        """
        if name not in self._scalers:
            raise KeyError(f"Scaler for '{name}' not found. Call fit(...) first.")
        scaler = self._scalers[name]
        vals = series.values.reshape(-1, 1)
        transformed = scaler.transform(vals).reshape(-1)
        return pd.Series(transformed, index=series.index, name=series.name)

    def inverse_transform(self, values: Union[pd.Series, np.ndarray], name: str) -> pd.Series:
        """
        Inverse-transform values (1D array or Series) using the named scaler.

        Returns a pandas Series (if input had index it's preserved, otherwise index None).
        """
        if name not in self._scalers:
            raise KeyError(f"Scaler for '{name}' not found. Call fit(...) first.")
        scaler = self._scalers[name]
        if isinstance(values, pd.Series):
            idx = values.index
            arr = values.values.reshape(-1, 1)
            inv = scaler.inverse_transform(arr).reshape(-1)
            return pd.Series(inv, index=idx, name=values.name)
        else:
            arr = np.asarray(values).reshape(-1, 1)
            inv = scaler.inverse_transform(arr).reshape(-1)
            return pd.Series(inv)

    def fit_transform(self, series: pd.Series, name: str) -> pd.Series:
        """
        Convenience: fit then transform the same series.
        """
        self.fit(series, name)
        return self.transform(series, name)

    def save(self, path: str) -> None:
        """
        Persist all scalers to the given path using joblib.

        If the directory does not exist, it is created.
        """
        dirname = os.path.dirname(path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
        joblib.dump({"scaler_type": self.scaler_type, "scalers": self._scalers}, path)
        Logger.debug("Saved %d scalers to %s", len(self._scalers), path)

    def load(self, path: str) -> None:
        """
        Load scalers from a joblib file saved by save(...).
        """
        data = joblib.load(path)
        self.scaler_type = data.get("scaler_type", self.scaler_type)
        self._scalers = data.get("scalers", {})
        Logger.debug("Loaded %d scalers from %s", len(self._scalers), path)

    def has(self, name: str) -> bool:
        return name in self._scalers

    def get_scaler(self, name: str):
        return self._scalers.get(name)
