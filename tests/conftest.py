"""
Pytest fixtures for synthetic OHLC data used by unit and integration tests.

Milestone 1: provide deterministic, offline-friendly fixtures:
 - ohlc_random_walk: DataFrame (DatetimeIndex business days) with columns
   ['Open','High','Low','Close','Adj Close','Volume'] (~120 rows).
 - mini_ohlc_csv: Path to a CSV file containing the first 30 rows of the
   synthetic OHLC (useful for tests that avoid network access).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def rng_seed() -> int:
    """Deterministic RNG seed for all synthetic fixtures in the test session."""
    return 42


@pytest.fixture
def ohlc_random_walk(rng_seed: int) -> pd.DataFrame:
    """
    Generate a simple, reproducible OHLC random-walk DataFrame.

    Schema:
      - index: pd.DatetimeIndex (business days), timezone-naive, normalized
      - columns: ['Open','High','Low','Close','Adj Close','Volume']

    Length: 120 business days by default (enough for training/testing windows).
    """
    rs = np.random.RandomState(rng_seed)
    periods = 120
    start = pd.Timestamp("2020-01-01")
    idx = pd.bdate_range(start=start, periods=periods).normalize()

    # Simulate log-returns and build a price path (geometric-like)
    returns = rs.normal(loc=0.0005, scale=0.01, size=periods)
    price = 100.0 * np.exp(np.cumsum(returns))

    close = price
    # Open is previous close with tiny random noise (first open = first close)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    open_ = open_ * (1.0 + rs.normal(loc=0.0, scale=0.001, size=periods))

    # High is the max of open/close inflated by a small positive factor
    high = np.maximum(open_, close) * (1.0 + np.abs(rs.normal(loc=0.0, scale=0.005, size=periods)))
    # Low is the min of open/close deflated by a small positive factor
    low = np.minimum(open_, close) * (1.0 - np.abs(rs.normal(loc=0.0, scale=0.005, size=periods)))

    adj_close = close.copy()
    volume = rs.randint(100_000, 1_000_000, size=periods).astype(int)

    df = pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Adj Close": adj_close,
            "Volume": volume,
        },
        index=idx,
    )

    # Enforce column order and types consistent with data contract
    df = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    df["Volume"] = df["Volume"].astype(int)
    df.index.name = None

    return df


@pytest.fixture
def mini_ohlc_csv(tmp_path, ohlc_random_walk: pd.DataFrame) -> str:
    """
    Write the first 30 rows of the synthetic OHLC to a CSV and return the path.
    Useful for tests that prefer reading from a file instead of relying on live data.
    """
    p = tmp_path / "mini_ohlc.csv"
    ohlc_random_walk.head(30).to_csv(p, index=True)
    return str(p)
