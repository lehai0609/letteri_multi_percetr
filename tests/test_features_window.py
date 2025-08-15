import numpy as np
import pandas as pd
import pytest

from src.features.window import build_xy, last_window


def test_build_xy_sequence():
    s = pd.Series(range(1, 11))  # 1..10
    X, y = build_xy(s, t=5)
    assert X.shape == (5, 5)
    assert y.shape == (5,)
    # first sample
    assert np.allclose(X[0], np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert pytest.approx(y[0]) == 6.0
    # last sample (window starting at 5 -> values 5..9)
    assert np.allclose(X[-1], np.array([5.0, 6.0, 7.0, 8.0, 9.0]))
    assert pytest.approx(y[-1]) == 10.0


def test_last_window_sequence():
    s = pd.Series(range(1, 11))
    last = last_window(s, t=5)
    assert isinstance(last, np.ndarray)
    assert last.shape == (5,)
    assert np.allclose(last, np.array([6.0, 7.0, 8.0, 9.0, 10.0]))


def test_build_xy_with_list_input():
    lst = list(range(1, 11))
    X, y = build_xy(lst, t=5)
    assert X.shape == (5, 5)
    assert y.shape == (5,)
    assert pytest.approx(y[0]) == 6.0


def test_build_xy_small_series_returns_empty():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    X, y = build_xy(s, t=5)
    assert X.shape == (0, 5)
    assert y.shape == (0,)


def test_last_window_too_short_raises():
    s = pd.Series([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError):
        last_window(s, t=5)


def test_invalid_t_raises():
    s = pd.Series(range(1, 11))
    with pytest.raises(ValueError):
        build_xy(s, t=0)
    with pytest.raises(ValueError):
        last_window(s, t=0)
