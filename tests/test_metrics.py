import math

import numpy as np
import pandas as pd
import pytest

from src.metrics.risk import (
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    compute_trade_returns,
    trade_stats_from_returns,
    compute_backtest_metrics,
)


def test_max_drawdown_simple():
    """
    Known equity path: peak at 120 then trough at 90 -> drawdown = (120-90)/120 = 0.25
    """
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    eq = pd.Series([100.0, 120.0, 90.0, 95.0, 130.0], index=idx)
    mdd = max_drawdown(eq)
    assert pytest.approx(mdd, rel=1e-12) == 0.25


def test_sharpe_constant_returns_nan():
    """
    Constant periodic returns => zero volatility => Sharpe should be NaN.
    """
    returns = pd.Series([0.01, 0.01, 0.01])
    s = sharpe_ratio(returns, rf=0.0, periods_per_year=252)
    assert math.isnan(s)


def test_sortino_all_positive_returns_nan():
    """
    If there are no negative returns, downside deviation is undefined and Sortino should be NaN.
    """
    returns = pd.Series([0.02, 0.01, 0.005])
    s = sortino_ratio(returns, rf=0.0, periods_per_year=252)
    assert math.isnan(s)


def test_trade_returns_and_stats():
    """
    Two round-trip trades: +20% and -10% -> verify compute_trade_returns and trade_stats.
    """
    trades = [
        {"date": pd.Timestamp("2020-01-01"), "side": "BUY", "qty": 10, "price": 10.0, "commission": 0.0},
        {"date": pd.Timestamp("2020-01-03"), "side": "SELL", "qty": 10, "price": 12.0, "commission": 0.0},
        {"date": pd.Timestamp("2020-01-04"), "side": "BUY", "qty": 10, "price": 10.0, "commission": 0.0},
        {"date": pd.Timestamp("2020-01-05"), "side": "SELL", "qty": 10, "price": 9.0, "commission": 0.0},
    ]
    tr = compute_trade_returns(trades)
    assert len(tr) == 2
    assert pytest.approx(tr[0], rel=1e-12) == 0.2
    assert pytest.approx(tr[1], rel=1e-12) == -0.1

    stats = trade_stats_from_returns(tr)
    assert stats["trade_count"] == 2
    assert pytest.approx(stats["win_rate"], rel=1e-12) == 0.5
    assert pytest.approx(stats["avg_win"], rel=1e-12) == 0.2
    assert pytest.approx(stats["avg_loss"], rel=1e-12) == -0.1
    assert pytest.approx(stats["expectancy"], rel=1e-12) == 0.05


def test_compute_backtest_metrics_keys_present():
    """
    Basic end-to-end sanity: metrics dict contains expected keys and computes total_return and max_drawdown correctly.
    """
    idx = pd.date_range("2020-01-01", periods=3, freq="B")
    eq = pd.Series([100.0, 110.0, 105.0], index=idx)
    trades = [
        {"date": idx[0], "side": "BUY", "qty": 1, "price": 100.0, "commission": 0.0},
        {"date": idx[2], "side": "SELL", "qty": 1, "price": 105.0, "commission": 0.0},
    ]
    metrics = compute_backtest_metrics(eq, trades, rf=0.0, periods_per_year=252)
    expected_keys = {
        "total_return",
        "annualized_return",
        "max_drawdown",
        "sharpe",
        "sortino",
        "calmar",
        "trade_count",
        "win_rate",
        "avg_win",
        "avg_loss",
        "expectancy",
    }
    assert expected_keys.issubset(set(metrics.keys()))
    assert pytest.approx(metrics["total_return"], rel=1e-12) == 0.05
    # peak 110 -> trough 105 => drawdown = 5/110
    assert pytest.approx(metrics["max_drawdown"], rel=1e-12) == (5.0 / 110.0)
