import pandas as pd

from src.utils import dates as du


def test_get_horizon_dates_length_and_type():
    idx = du.get_horizon_dates("2021-10-16", horizon=30)
    assert isinstance(idx, pd.DatetimeIndex)
    assert len(idx) == 30
    # timezone-naive and normalized
    assert idx.tz is None
    assert all(ts == ts.normalize() for ts in idx)
    assert idx.is_monotonic_increasing
