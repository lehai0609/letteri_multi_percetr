import math

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import BacktestEngine


def test_backtest_fractional_shares():
    """
    Deterministic enter on day1 (BUY), hold day2, exit on day3 (SELL).
    Verify quantities, trade sides, and final cash using fractional shares.
    """
    dates = pd.date_range("2020-01-06", periods=3, freq="B")
    actual = pd.DataFrame(index=dates)
    actual["Open"] = [10.0, 11.0, 12.0]
    actual["Close"] = [10.5, 11.5, 12.5]

    signals = pd.Series([1, 1, 0], index=dates, name="signal")

    engine = BacktestEngine(
        initial_cash=100.0,
        cost_bps=10.0,
        slippage_bps=5.0,
        allow_fractional=True,
        execution_price="open",
        mark_to_market="close",
    )
    trades, equity = engine.run(actual, signals)

    # Two trades: BUY then SELL
    assert len(trades) == 2
    buy = trades[0]
    sell = trades[1]
    assert buy["side"] == "BUY"
    assert sell["side"] == "SELL"

    # Recompute expected values using the same formulas as the engine
    cost_rate = 10.0 / 10000.0
    slip_rate = 5.0 / 10000.0

    px_buy = float(actual.loc[dates[0], "Open"])
    fill_buy = px_buy * (1.0 + slip_rate)
    denom = fill_buy * (1.0 + cost_rate)
    expected_qty = 100.0 / denom

    assert pytest.approx(buy["qty"], rel=1e-12) == expected_qty

    px_sell = float(actual.loc[dates[2], "Open"])
    fill_sell = px_sell * (1.0 - slip_rate)
    expected_final_cash = expected_qty * fill_sell * (1.0 - cost_rate)

    assert pytest.approx(equity.iloc[-1], rel=1e-12) == expected_final_cash


def test_backtest_integer_shares():
    """
    Same scenario but integer share quantities (no fractional shares).
    Verify floor behaviour and resulting cash.
    """
    dates = pd.date_range("2020-01-06", periods=3, freq="B")
    actual = pd.DataFrame(index=dates)
    actual["Open"] = [10.0, 11.0, 12.0]
    actual["Close"] = [10.5, 11.5, 12.5]

    signals = pd.Series([1, 1, 0], index=dates, name="signal")

    engine = BacktestEngine(
        initial_cash=100.0,
        cost_bps=10.0,
        slippage_bps=5.0,
        allow_fractional=False,
        execution_price="open",
        mark_to_market="close",
    )
    trades, equity = engine.run(actual, signals)

    assert len(trades) == 2
    buy = trades[0]
    sell = trades[1]

    cost_rate = 10.0 / 10000.0
    slip_rate = 5.0 / 10000.0

    px_buy = float(actual.loc[dates[0], "Open"])
    fill_buy = px_buy * (1.0 + slip_rate)
    denom = fill_buy * (1.0 + cost_rate)
    expected_qty = float(np.floor(100.0 / denom))

    assert buy["qty"] == expected_qty

    # Manually compute expected final cash after buy (leaves some cash) and sell
    trade_value_buy = expected_qty * fill_buy
    commission_buy = trade_value_buy * cost_rate
    cash_after_buy = 100.0 - trade_value_buy - commission_buy

    px_sell = float(actual.loc[dates[2], "Open"])
    fill_sell = px_sell * (1.0 - slip_rate)
    trade_value_sell = expected_qty * fill_sell
    commission_sell = trade_value_sell * cost_rate
    expected_final_cash = cash_after_buy + (trade_value_sell - commission_sell)

    assert pytest.approx(equity.iloc[-1], rel=1e-12) == expected_final_cash
