"""
Unit test for Milestone 8: recursive forecaster.

Verifies that a stub model which predicts the last scaled input value
produces the same multi-step OHLC path as the persistence forecaster
when using the recursive_forecast implementation and fitted scalers.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import pandas.testing as pdt

from src.forecasting.recursive import recursive_forecast
from src.forecasting.persistence import persistence_forecast
from src.data.scale import ScalerManager


OHLC = ["Open", "High", "Low", "Close"]


def test_recursive_forecast_matches_persistence(ohlc_random_walk: pd.DataFrame) -> None:
    # Arrange: use the synthetic fixture and the same split logic as train.py fallback
    df = ohlc_random_walk
    cutoff_idx = 89
    horizon = 30
    lag_t = 5

    train_df = df.iloc[: cutoff_idx + 1].copy()
    test_df = df.iloc[cutoff_idx + 1 : cutoff_idx + 1 + horizon].copy()
    assert len(test_df) == horizon

    # Fit scalers on training data (matching expected pipeline)
    scaler_mgr = ScalerManager(scaler_type="standard")
    for col in OHLC:
        scaler_mgr.fit(train_df[col], name=col)

    # Stub model: returns the last value of the input window (persistence in scaled space)
    class LastScaledModel:
        def predict(self, X: np.ndarray, verbose: int = 0) -> np.ndarray:
            # Return last column (shape (n_samples,))
            arr = np.asarray(X)
            return arr[:, -1]

    models = {col: LastScaledModel() for col in OHLC}

    # Act
    rec_pred = recursive_forecast(models, scaler_mgr, train_df, test_df.index, lag_t)
    pers_pred = persistence_forecast(train_df, test_df.index, method="last")

    # Assert: the recursive forecast using the last-scaled-value models should equal
    # the persistence forecast (values and index). Allow a tiny tolerance for numeric ops.
    pdt.assert_frame_equal(rec_pred, pers_pred, check_dtype=False, atol=1e-8, rtol=1e-6)
