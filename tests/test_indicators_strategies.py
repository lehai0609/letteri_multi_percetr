import pandas as pd
import pytest

from src.indicators.tema import ema, tema, tema_ohlc
from src.strategies.threshold import generate_signals as threshold_signals
from src.strategies.triple_ema import (
    generate_signals as triple_signals,
    evaluate_operator_expression,
)


def test_tema_matches_manual_ema_chain():
    idx = pd.bdate_range("2020-01-01", periods=5)
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)
    period = 3

    ema1 = s.ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, adjust=False).mean()
    tema_ref = 3.0 * ema1 - 3.0 * ema2 + ema3

    pd.testing.assert_series_equal(tema(s, period), tema_ref)


def test_tema_ohlc_on_fixture(ohlc_random_walk):
    df = ohlc_random_walk.head(10).copy()
    tema_df = tema_ohlc(df, period=3)
    # returns the same index and OHLC columns
    assert list(tema_df.columns) == ["Open", "High", "Low", "Close"]
    assert tema_df.shape[0] == df.shape[0]


def test_threshold_strategy_simple():
    idx = pd.bdate_range("2020-01-01", periods=4)
    df = pd.DataFrame(
        {
            "Open": [100.0, 100.0, 100.0, 100.0],
            "High": [101.0, 101.0, 101.0, 101.0],
            "Low": [99.0, 99.0, 99.0, 99.0],
            "Close": [110.0, 105.0, 95.0, 95.0],
        },
        index=idx,
    )

    sig = threshold_signals(df)
    assert list(sig["signal"]) == [1, 1, 0, 0]


def test_evaluate_operator_expression_basic_true():
    expr = "((Low < TEMA_Low) and (Close < TEMA_Close))"
    vars_map = {"Low": 10.0, "TEMA_Low": 11.0, "Close": 9.0, "TEMA_Close": 10.0}
    assert evaluate_operator_expression(expr, vars_map) is True


def test_evaluate_operator_expression_error_returns_false():
    expr = "Low << TEMA_Low"  # invalid syntax
    vars_map = {"Low": 10.0, "TEMA_Low": 11.0}
    assert evaluate_operator_expression(expr, vars_map) is False


def test_triple_ema_signals_entry_on_sharp_drop():
    idx = pd.bdate_range("2020-01-01", periods=5)
    # constant price then a sharp drop then recovery
    prices = [100.0, 100.0, 100.0, 80.0, 100.0]
    df = pd.DataFrame(
        {
            "Open": prices,
            "High": [p * 1.01 for p in prices],
            "Low": [p * 0.99 for p in prices],
            "Close": prices,
        },
        index=idx,
    )

    sig = triple_signals(df, period=3)
    # entry should trigger on the sharp drop day (index position 3)
    drop_idx = idx[3]
    assert sig.loc[drop_idx, "signal"] == 1
    # prior day should be flat
    assert sig.loc[idx[2], "signal"] == 0
    # ensure at least one 0 -> 1 transition occurred
    diffs = sig["signal"].diff().fillna(0)
    assert (diffs == 1).any()
