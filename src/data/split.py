"""
Splitter utilities for time-series train/test splits.

Provides:
- split_by_cutoff(df, cutoff, horizon=30, calendar="NYSE"):
    Returns (train_df, test_df) where test_df contains exactly `horizon`
    trading days following `cutoff` (exclusive by default). Train_df ends
    at the cutoff (inclusive) and contains all earlier rows.

Behavior / Contracts:
- Expects df index to be a timezone-naive, normalized DatetimeIndex and to
  contain trading-day rows (delegates validation to src.utils.dates.ensure_datetime_index).
- If there are not enough trading days after cutoff to satisfy `horizon`,
  raises ValueError.
"""
from __future__ import annotations

from typing import Tuple, Optional

import pandas as pd

from src.utils import dates as du


def split_by_cutoff(
    df: pd.DataFrame,
    cutoff: str,
    *,
    horizon: int = 30,
    include_cutoff_in_test: bool = False,
    calendar: Optional[str] = "NYSE",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a time-series DataFrame into training and test windows based on a cutoff date.

    Args:
        df: input DataFrame indexed by DatetimeIndex with OHLCV columns
        cutoff: cutoff date (train ends here). Can be a date-like string.
        horizon: number of trading days in the test window (default 30)
        include_cutoff_in_test: if True and cutoff is a trading day, include it as first day of test
        calendar: calendar name passed to date utilities

    Returns:
        (train_df, test_df) where:
          - test_df.index length == horizon
          - train_df.index max == cutoff (if cutoff is in df.index) or train_df contains all rows <= cutoff-aligned

    Raises:
        ValueError if cutoff is not present/alignable or if not enough trading days exist after cutoff.
    """
    # Ensure df index is proper DatetimeIndex
    du.ensure_datetime_index(df)

    cutoff_ts = du.parse_date(cutoff)

    # Align cutoff to nearest trading day 'previous' so that train ends at last trading day <= cutoff.
    # This mirrors typical practice where cutoff is inclusive for training.
    cutoff_aligned = du.align_to_trading_day(cutoff_ts, direction="previous", calendar=calendar)

    # Build test dates: next `horizon` trading days after the cutoff_aligned (exclusive by default)
    test_start_inclusive = cutoff_aligned if include_cutoff_in_test else du.add_trading_days(cutoff_aligned, 1, calendar=calendar)
    try:
        test_index = du.next_trading_days(test_start_inclusive - pd.Timedelta(days=1), horizon, include_start=False, calendar=calendar)
    except ValueError as exc:
        # Re-raise with context
        raise ValueError(f"Not enough trading days after cutoff {cutoff} to build a {horizon}-day horizon") from exc

    # Ensure test_index length matches horizon
    if len(test_index) != horizon:
        raise ValueError(f"Expected {horizon} trading days for test window, got {len(test_index)}")

    # Extract rows from df corresponding to train and test windows.
    # Train: all rows with index <= cutoff_aligned
    train_df = df.loc[df.index <= cutoff_aligned].copy()
    # Test: rows in df that intersect with the test_index. It's OK if some forecast dates are missing
    # from df (e.g., when using cached partial data); we will require exact match to avoid leakage.
    test_df = df.loc[df.index.isin(test_index)].copy()

    if len(test_df) != horizon:
        # Identify missing dates
        missing = list(test_index.difference(test_df.index))
        raise ValueError(f"Test window missing {len(missing)} trading days: {missing[:5]}")

    return train_df, test_df
