"""
Risk metrics for the Letteri replication.

Functions:
- max_drawdown(equity)
- sharpe_ratio(returns, rf=0.0, periods_per_year=252)
- sortino_ratio(returns, rf=0.0, periods_per_year=252)
- calmar_ratio(equity, periods_per_year=252)
- compute_trade_returns(trades)
- trade_stats_from_returns(trade_returns)
- compute_backtest_metrics(equity, trades, rf=0.0, periods_per_year=252)

The implementations favor clarity and defensiveness for small sample sizes.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import math

import numpy as np
import pandas as pd


def max_drawdown(equity: pd.Series) -> float:
    """
    Compute maximum drawdown as the maximum peak-to-trough decline (fraction).

    Parameters
    ----------
    equity
        Series of portfolio equity indexed by date.

    Returns
    -------
    float
        Maximum drawdown (0..1). Returns 0.0 for non-decreasing series or empty input.
    """
    if equity is None or len(equity.dropna()) == 0:
        return 0.0
    eq = equity.dropna().astype(float)
    running_max = eq.cummax()
    # avoid division by zero
    denom = running_max.replace(0, np.nan)
    drawdowns = (running_max - eq) / denom
    mdd = float(drawdowns.max(skipna=True)) if not drawdowns.empty else 0.0
    if math.isnan(mdd):
        return 0.0
    return mdd


def _returns_from_equity(equity: pd.Series) -> pd.Series:
    """
    Helper to compute simple returns from an equity series.
    """
    return equity.astype(float).pct_change().dropna()


def sharpe_ratio(
    returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252
) -> float:
    """
    Annualized Sharpe ratio, using population std (ddof=0).

    Parameters
    ----------
    returns
        Series of simple periodic returns (already pct-change).
    rf
        Annual risk-free rate (e.g., 0.0). Converted to periodic by dividing by periods_per_year.
    periods_per_year
        Number of periods per year used for annualization.

    Returns
    -------
    float
        Annualized Sharpe ratio, or np.nan if undefined (e.g., zero volatility or empty).
    """
    if returns is None or len(returns.dropna()) == 0:
        return float("nan")
    r = returns.dropna().astype(float)
    rf_period = float(rf) / float(periods_per_year)
    excess = r - rf_period
    mean = float(excess.mean())
    std = float(excess.std(ddof=0))
    if std == 0.0 or math.isnan(std):
        return float("nan")
    return mean / std * math.sqrt(periods_per_year)


def sortino_ratio(
    returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252
) -> float:
    """
    Annualized Sortino ratio: mean excess return divided by downside deviation.

    Returns np.nan when downside deviation is zero or when inputs are empty.
    """
    if returns is None or len(returns.dropna()) == 0:
        return float("nan")
    r = returns.dropna().astype(float)
    rf_period = float(rf) / float(periods_per_year)
    excess = r - rf_period
    downside = excess[excess < 0.0]
    if len(downside) == 0:
        return float("nan")
    # downside deviation: sqrt(mean(square(downside)))
    downside_dev = float(np.sqrt(np.mean(np.square(downside.values))))
    if downside_dev == 0.0 or math.isnan(downside_dev):
        return float("nan")
    mean_excess = float(excess.mean())
    return mean_excess / downside_dev * math.sqrt(periods_per_year)


def annualized_return(equity: pd.Series, periods_per_year: int = 252) -> float:
    """
    Annualize total return computed from equity series.

    If equity has length <= 1, returns 0.0.
    """
    eq = equity.dropna().astype(float)
    if len(eq) <= 1:
        return 0.0
    total_ret = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    n_periods = float(len(eq))
    # guard against negative base raising to fractional power
    ann = (1.0 + total_ret) ** (periods_per_year / n_periods) - 1.0
    return float(ann)


def calmar_ratio(equity: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calmar ratio = annualized return / max_drawdown.

    Returns np.nan when max_drawdown is zero/undefined.
    """
    mdd = max_drawdown(equity)
    if mdd == 0.0:
        return float("nan")
    ann = annualized_return(equity, periods_per_year=periods_per_year)
    return float(ann / mdd)


def compute_trade_returns(trades: Sequence[Dict]) -> List[float]:
    """
    From a list of trades (dicts with keys 'side','qty','price','commission' optionally),
    compute percentage returns per completed round-trip trade (BUY -> SELL).

    Assumptions:
    - Trades are ordered chronologically.
    - A BUY is closed by the next SELL that follows it (full pairing).
    - Partial fills are handled by pairing quantities as they appear (simple approach).

    Returns
    -------
    List[float]
        List of trade returns as fractional returns (e.g., 0.2 == +20%).
    """
    returns: List[float] = []
    open_buy: Optional[Dict] = None

    for tr in trades:
        side = tr.get("side", "").upper()
        if side == "BUY":
            # start or overwrite open buy (we assume full round-trip)
            open_buy = dict(tr)
        elif side == "SELL":
            if open_buy is None:
                # sell without buy; skip
                continue
            # Determine qty used for pairing. Prefer the qty in the buy record.
            qty = float(open_buy.get("qty", tr.get("qty", 0.0)))
            buy_price = float(open_buy.get("price", 0.0))
            buy_comm = float(open_buy.get("commission", 0.0))
            sell_price = float(tr.get("price", 0.0))
            sell_comm = float(tr.get("commission", 0.0))
            buy_cost = buy_price * qty + buy_comm
            sell_proceeds = sell_price * qty - sell_comm
            if buy_cost == 0:
                ret = 0.0
            else:
                ret = sell_proceeds / buy_cost - 1.0
            returns.append(float(ret))
            open_buy = None
        else:
            # unknown side; ignore
            continue
    return returns


def trade_stats_from_returns(trade_returns: Sequence[float]) -> Dict[str, float]:
    """
    Compute win rate, avg_win, avg_loss, expectancy and trade_count from a list of trade returns.
    """
    arr = np.array(list(trade_returns), dtype=float)
    n = len(arr)
    if n == 0:
        return {
            "trade_count": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "expectancy": 0.0,
        }
    wins = arr[arr > 0.0]
    losses = arr[arr <= 0.0]
    win_rate = float(len(wins) / n)
    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
    expectancy = win_rate * avg_win - (1.0 - win_rate) * abs(avg_loss)
    return {
        "trade_count": int(n),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": expectancy,
    }


def compute_backtest_metrics(
    equity: pd.Series,
    trades: Sequence[Dict],
    rf: float = 0.0,
    periods_per_year: int = 252,
) -> Dict[str, float]:
    """
    Compute a dictionary of backtest risk/performance metrics.

    Returns keys:
      - total_return
      - annualized_return
      - max_drawdown
      - sharpe
      - sortino
      - calmar
      - trade_count
      - win_rate
      - avg_win
      - avg_loss
      - expectancy
    """
    eq = equity.dropna().astype(float)
    total_return = float(eq.iloc[-1] / eq.iloc[0] - 1.0) if len(eq) >= 2 else 0.0
    ann_ret = annualized_return(eq, periods_per_year=periods_per_year)
    mdd = max_drawdown(eq)
    rets = _returns_from_equity(eq)
    sharpe = sharpe_ratio(rets, rf=rf, periods_per_year=periods_per_year)
    sortino = sortino_ratio(rets, rf=rf, periods_per_year=periods_per_year)
    calmar = calmar_ratio(eq, periods_per_year=periods_per_year)
    trade_returns = compute_trade_returns(trades)
    stats = trade_stats_from_returns(trade_returns)
    result: Dict[str, float] = {
        "total_return": total_return,
        "annualized_return": ann_ret,
        "max_drawdown": mdd,
        "sharpe": float(sharpe) if not (sharpe is None) else float("nan"),
        "sortino": float(sortino) if not (sortino is None) else float("nan"),
        "calmar": float(calmar) if not (calmar is None) else float("nan"),
        "trade_count": float(stats["trade_count"]),
        "win_rate": float(stats["win_rate"]),
        "avg_win": float(stats["avg_win"]),
        "avg_loss": float(stats["avg_loss"]),
        "expectancy": float(stats["expectancy"]),
    }
    return result
