"""
Unit tests for Milestone 2 - data layer core:
- Weekend/gap detection (validate_calendar)
- split_by_cutoff returns exact horizon length
- ScalerManager roundtrip (fit/transform/inverse)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from src.data.ingest import DataIngestor
from src.data.split import split_by_cutoff
from src.data.scale import ScalerManager
from src.utils import dates as du


def test_validate_calendar_detects_missing(ohlc_random_walk: pd.DataFrame, caplog):
    # Drop a trading day to simulate a missing day / weekend gap detection
    df = ohlc_random_walk.copy()
    missing_idx = df.index[10]
    df = df.drop(missing_idx)

    caplog.set_level(logging.WARNING)
    ing = DataIngestor(source="csv", path_or_ticker="unused")
    # Should log a warning about missing trading days
    ing.validate_calendar(df)

    messages = "\n".join(r.message for r in caplog.records)
    assert "missing trading days" in messages.lower() or "missing trading" in messages.lower()


def test_split_by_cutoff_returns_exact_horizon(ohlc_random_walk: pd.DataFrame):
    df = ohlc_random_walk.copy()
    # Choose a cutoff such that there are at least 30 trading days after it.
    # Pick cutoff at index position 60 (0-based) which leaves 59 days after (120 total)
    cutoff = df.index[60]
    # Use horizon 30
    train_df, test_df = split_by_cutoff(df, cutoff.strftime("%Y-%m-%d"), horizon=30, calendar="NYSE")
    assert len(test_df) == 30
    # Train should end at or before the cutoff (aligned to previous trading day)
    cutoff_aligned = du.align_to_trading_day(cutoff, direction="previous", calendar="NYSE")
    assert train_df.index.max() <= cutoff_aligned


def test_scaler_roundtrip_train_transform_inverse(ohlc_random_walk: pd.DataFrame, tmp_path):
    df = ohlc_random_walk.copy()
    # Use first 80 rows as "train"
    train = df.iloc[:80]
    series = train["Close"]

    sm = ScalerManager(scaler_type="standard")
    sm.fit(series, name="Close")
    transformed = sm.transform(series, name="Close")
    inverted = sm.inverse_transform(transformed, name="Close")

    # Values must match original within a small tolerance
    assert np.allclose(inverted.values, series.values, rtol=1e-6, atol=1e-8)

    # Test save/load behavior
    p = tmp_path / "scalers.joblib"
    sm.save(str(p))
    sm2 = ScalerManager(scaler_type="standard")
    sm2.load(str(p))
    assert sm2.has("Close")
    restored = sm2.inverse_transform(transformed, name="Close")
    assert np.allclose(restored.values, series.values, rtol=1e-6, atol=1e-8)
