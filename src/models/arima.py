"""
Auto-ARIMA baseline forecaster (Milestone 10).

Provides:
- AutoARIMABaseline: wrapper around statsmodels.tsa.arima.model.ARIMA with automatic
  order selection for per-series fitting and multi-step forecasting.
- forecast_from_split: convenience wrapper to produce an OHLC DataFrame of forecasts
  aligned to a provided test_df.index (same shape contract as other forecasters).

Notes:
- statsmodels is an optional dependency; importing this module will not raise immediately,
  but attempting to fit will raise ImportError if statsmodels is not installed.
- The implementation is defensive: if statsmodels raises during fit (often due to too-
  short series), we re-raise a RuntimeError with a clear message.
- Uses simple grid search for automatic order selection instead of pmdarima's auto_arima.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd

OHLC = ["Open", "High", "Low", "Close"]


try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.stats.diagnostic import acorr_ljungbox
    statsmodels_available = True
except Exception:  # pragma: no cover - exercised only in environments lacking statsmodels
    ARIMA = None  # type: ignore
    adfuller = None  # type: ignore
    acorr_ljungbox = None  # type: ignore
    statsmodels_available = False


def _require_statsmodels() -> None:
    if not statsmodels_available:
        raise ImportError(
            "statsmodels is required for the ARIMA baseline but is not installed. "
            "Install it with `pip install statsmodels` to use Auto-ARIMA functionality."
        )


def _determine_differencing(series: pd.Series, alpha: float = 0.05) -> int:
    """
    Determine the degree of differencing needed using ADF test.
    
    Returns the minimum differencing order (d) to make series stationary.
    """
    _require_statsmodels()
    
    def is_stationary(data: np.ndarray) -> bool:
        if len(data) < 10:  # Need minimum observations for ADF test
            return True
        try:
            result = adfuller(data, autolag='AIC')
            return result[1] <= alpha  # p-value <= alpha means stationary
        except Exception:
            return True  # Default to stationary if test fails
    
    data = series.dropna().values
    if len(data) < 10:
        return 0
    
    # Test up to 2 differences (d=0,1,2)
    for d in range(3):
        if d == 0:
            test_data = data
        else:
            test_data = np.diff(data, n=d)
        
        if is_stationary(test_data):
            return d
    
    return 2  # Default to d=2 if still not stationary


def _auto_select_order(
    series: pd.Series,
    max_p: int = 5,
    max_q: int = 5,
    max_d: int = 2,
    seasonal: bool = False
) -> Tuple[int, int, int]:
    """
    Simple grid search to find best ARIMA order (p,d,q) based on AIC.
    
    Returns (p, d, q) tuple with lowest AIC.
    """
    _require_statsmodels()
    
    data = series.dropna().values
    if len(data) < 10:
        return (1, 0, 1)  # Default for very short series
    
    # Determine differencing order
    d = min(_determine_differencing(series), max_d)
    
    best_aic = float('inf')
    best_order = (1, d, 1)
    
    # Grid search over p and q
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            if p == 0 and q == 0:
                continue  # Skip (0,d,0) model
            
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = ARIMA(data, order=(p, d, q))
                    fitted = model.fit(method_kwargs={'warn_convergence': False})
                    
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)
            except Exception:
                continue  # Skip problematic orders
    
    return best_order


class AutoARIMABaseline:
    """
    Per-series Auto-ARIMA baseline.

    Wrapper around statsmodels ARIMA with automatic order selection providing fit/forecast/diagnostics.

    Parameters
    ----------
    seasonal : bool
        Whether to consider seasonal models (default False).
    max_p : int
        Maximum autoregressive order (default 5).
    max_q : int
        Maximum moving average order (default 5).
    max_d : int
        Maximum differencing order (default 2).
    auto_arima_kwargs : dict
        Extra keyword arguments (for compatibility, mostly ignored).
    """

    def __init__(
        self, 
        *, 
        seasonal: bool = False, 
        max_p: int = 5,
        max_q: int = 5,
        max_d: int = 2,
        **auto_arima_kwargs: Any
    ) -> None:
        self.seasonal = bool(seasonal)
        self.max_p = max_p
        self.max_q = max_q
        self.max_d = max_d
        self.auto_arima_kwargs = dict(auto_arima_kwargs)
        self.model = None
        self.fitted_model = None
        self.order = None
        self.fitted = False
        self._train_len = 0

    def fit(self, series: pd.Series) -> "AutoARIMABaseline":
        """
        Fit an Auto-ARIMA model to a univariate pandas Series.

        Raises
        ------
        ValueError : if the input is invalid (None / empty / non-numeric)
        RuntimeError : if the underlying statsmodels ARIMA fails (e.g., too-short series)
        ImportError : if statsmodels is not available
        """
        _require_statsmodels()
        if series is None:
            raise ValueError("series must be a pandas Series (got None)")
        s = pd.Series(series).dropna().astype(float)
        if s.shape[0] == 0:
            raise ValueError("series must contain at least one non-NaN observation")
        # record training length for diagnostics / sanity checks
        self._train_len = int(len(s))
        try:
            # Automatically select best ARIMA order
            self.order = _auto_select_order(
                s, 
                max_p=self.max_p, 
                max_q=self.max_q, 
                max_d=self.max_d, 
                seasonal=self.seasonal
            )
            
            # Fit the selected model
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model = ARIMA(s.values, order=self.order)
                self.fitted_model = self.model.fit(method_kwargs={'warn_convergence': False})
            
            self.fitted = True
            return self
        except Exception as exc:
            # Surface a clear runtime error for tests and callers
            raise RuntimeError(f"Auto-ARIMA fit failed (series length={len(s)}): {exc}") from exc

    def forecast(self, h: int) -> np.ndarray:
        """
        Forecast `h` steps ahead using the fitted ARIMA model.

        Returns a 1-D numpy array of length `h`.

        Raises
        ------
        RuntimeError : if called before fit()
        ValueError : if h <= 0
        """
        if not self.fitted or self.fitted_model is None:
            raise RuntimeError("Model not fitted. Call fit(series) before forecast().")
        if h <= 0:
            raise ValueError("h (horizon) must be a positive integer")
        
        try:
            # statsmodels uses forecast(steps=...)
            preds = self.fitted_model.forecast(steps=int(h))
            arr = np.asarray(preds, dtype=float).reshape(-1)
            if arr.shape[0] != int(h):
                # Defensive: ensure caller receives expected-length array
                raise RuntimeError(f"ARIMA forecast returned unexpected length {arr.shape[0]} (expected {h})")
            return arr
        except Exception as exc:
            raise RuntimeError(f"ARIMA forecast failed: {exc}") from exc

    def diagnostics(self) -> Dict[str, Any]:
        """
        Return basic diagnostics for the fitted model: aic, bic, order (if available).
        """
        if not self.fitted or self.fitted_model is None:
            return {}
        out: Dict[str, Any] = {}
        # statsmodels fitted model exposes aic/bic as properties
        try:
            out["aic"] = float(self.fitted_model.aic)
        except Exception:
            out["aic"] = None
        try:
            out["bic"] = float(self.fitted_model.bic)
        except Exception:
            out["bic"] = None
        # Include the selected order
        try:
            out["order"] = self.order
        except Exception:
            out["order"] = None
        try:
            out["seasonal_order"] = getattr(self.fitted_model, "seasonal_order", None)
        except Exception:
            out["seasonal_order"] = None
        return out


def forecast_from_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    seasonal: bool = False,
    arima_kwargs: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Convenience wrapper: fit Auto-ARIMA per OHLC series on train_df and forecast
    a horizon equal to len(test_df). Returns a DataFrame indexed like test_df with
    columns ['Open','High','Low','Close'].

    Parameters
    ----------
    train_df : pd.DataFrame
        Historical train frame containing OHLC columns.
    test_df : pd.DataFrame
        DataFrame whose index defines the forecast dates (horizon=len(test_df)).
    seasonal : bool
        Whether to allow seasonal ARIMA terms (m>1). Forwarded to AutoARIMABaseline.
    arima_kwargs : dict
        Extra kwargs forwarded to AutoARIMABaseline (max_p, max_q, max_d supported).

    Raises
    ------
    ValueError : if required columns are missing or inputs invalid
    RuntimeError : if ARIMA fitting/forecasting fails for any series (e.g., too-short series)
    ImportError : if statsmodels is not installed
    """
    if train_df is None or len(train_df) == 0:
        raise ValueError("train_df must be a non-empty DataFrame")
    if test_df is None or len(test_df) == 0:
        raise ValueError("test_df must be a non-empty DataFrame (index defines horizon)")
    missing = [c for c in OHLC if c not in train_df.columns]
    if missing:
        raise ValueError(f"train_df is missing required columns: {missing}")
    missing_pred = [c for c in OHLC if c not in test_df.columns]
    # We only require test_df.index, not necessarily OHLC columns. If OHLC absent, we will still use its index.
    idx = test_df.index
    h = len(idx)

    arima_kwargs = dict(arima_kwargs or {})

    preds: List[Dict[str, float]] = []

    # Fit and forecast per-series
    for c in OHLC:
        try:
            baseline = AutoARIMABaseline(seasonal=seasonal, **arima_kwargs)
            series = train_df[c].dropna().astype(float)
            if len(series) == 0:
                raise ValueError(f"train_df column '{c}' contains no valid observations")
            baseline.fit(series)
            arr = baseline.forecast(h)
            # Convert predicted array into floats for the DataFrame assembly
            vals = [float(x) for x in arr]
        except Exception as exc:
            # Re-wrap errors to provide context per-series
            raise RuntimeError(f"ARIMA baseline failed for series '{c}': {exc}") from exc

        # collect column-wise predictions; we'll transpose later
        if not preds:
            # initialize rows
            preds = [{c: vals[i]} for i in range(h)]
        else:
            for i in range(h):
                preds[i][c] = vals[i]

    pred_df = pd.DataFrame(preds, index=idx)[OHLC]
    return pred_df


def buy_and_hold_signals(index: Iterable[pd.Timestamp]) -> pd.DataFrame:
    """
    Produce a buy-and-hold signal DataFrame aligned to `index`.

    Returns a DataFrame with a single column 'signal' where all entries are 1 (long).
    """
    idx = pd.DatetimeIndex(index)
    sig = pd.Series(1, index=idx, name="signal")
    return pd.DataFrame({"signal": sig})


__all__ = ["AutoARIMABaseline", "forecast_from_split", "buy_and_hold_signals"]
