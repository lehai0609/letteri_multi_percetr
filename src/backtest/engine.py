"""
Backtest engine for the Letteri replication.

Provides:
- BacktestEngine: simple all-in/all-out portfolio that executes signals at the
  configured execution price (Open by default), supports fractional shares,
  commissions (bps) and slippage (bps), and returns a trades list + equity curve.
- run_backtest: convenience wrapper.

This module is intentionally small and well-tested by unit tests in tests/.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union

import math

import numpy as np
import pandas as pd


class BacktestEngine:
    """
    BacktestEngine executes a binary signal stream (0 = flat, 1 = long)
    against actual OHLC bars.

    Parameters
    ----------
    initial_cash:
        Starting cash (default 100.0)
    cost_bps:
        Commission in basis points (e.g. 10.0 -> 0.001)
    slippage_bps:
        Slippage per fill in basis points (e.g. 5.0 -> 0.0005)
    allow_fractional:
        If True, fractional shares allowed. Otherwise floor quantities to integers.
    execution_price:
        'open' or 'close' (price used to execute trades)
    mark_to_market:
        'close' or 'open' (price used to mark portfolio for equity curve)
    """

    def __init__(
        self,
        initial_cash: float = 100.0,
        cost_bps: float = 10.0,
        slippage_bps: float = 5.0,
        allow_fractional: bool = True,
        execution_price: str = "open",
        mark_to_market: str = "close",
    ) -> None:
        self.initial_cash = float(initial_cash)
        self.cost_rate = float(cost_bps) / 10000.0
        self.slip_rate = float(slippage_bps) / 10000.0
        self.allow_fractional = bool(allow_fractional)
        self.execution_price = execution_price.lower()
        self.mark_to_market = mark_to_market.lower()

        if self.execution_price not in ("open", "close"):
            raise ValueError("execution_price must be 'open' or 'close'")
        if self.mark_to_market not in ("open", "close"):
            raise ValueError("mark_to_market must be 'open' or 'close'")

    def _signal_series_from_df(
        self, signals: Union[pd.Series, pd.DataFrame]
    ) -> pd.Series:
        if isinstance(signals, pd.DataFrame):
            if "signal" in signals.columns:
                sig = signals["signal"].astype(int)
            else:
                # assume first column contains the signal
                sig = signals.iloc[:, 0].astype(int)
        else:
            sig = signals.astype(int)
        return sig

    def run(
        self, actual_df: pd.DataFrame, signals: Union[pd.Series, pd.DataFrame]
    ) -> Tuple[List[Dict[str, Any]], pd.Series]:
        """
        Run backtest.

        Parameters
        ----------
        actual_df:
            DataFrame with index = trading dates and at minimum columns ['Open','Close'].
        signals:
            Series or DataFrame aligned to actual_df.index containing binary signals (0/1)

        Returns
        -------
        trades, equity_curve
            trades: list of trade dicts with keys:
                'date','side'('BUY'/'SELL'),'qty','price','commission','cash','position'
            equity_curve: pd.Series indexed by dates with portfolio equity marked
                          at `mark_to_market` price for that day.
        """
        sig = self._signal_series_from_df(signals)
        # Align signals with actual data index (signals must be subset or equal)
        if not sig.index.equals(actual_df.index):
            # allow index intersection but require signals index to be subset of actual_df
            try:
                sig = sig.reindex(actual_df.index).fillna(method="ffill").fillna(0).astype(int)
            except Exception:
                raise ValueError("signals index must align with actual_df index or be reindexable")

        # Validate columns exist
        for col in ("Open", "Close"):
            if col not in actual_df.columns:
                raise ValueError(f"actual_df must contain '{col}' column")

        cash = float(self.initial_cash)
        position = 0.0
        trades: List[Dict[str, Any]] = []
        equity = pd.Series(index=sig.index, dtype=float)

        last_state = 0
        for date in sig.index:
            desired = int(sig.loc[date])

            # Determine execution price for a trade (based on execution_price config)
            if self.execution_price == "open":
                px_exec = float(actual_df.loc[date, "Open"])
            else:
                px_exec = float(actual_df.loc[date, "Close"])

            # Compute fill price considering slippage direction:
            # - Buys get a worse (higher) price: * (1 + slip)
            # - Sells get a worse (lower) price: * (1 - slip)
            fill_px = px_exec
            if desired > last_state:
                # entering long
                fill_px = px_exec * (1.0 + self.slip_rate)
            elif desired < last_state:
                # exiting long
                fill_px = px_exec * (1.0 - self.slip_rate)

            # Transition logic (all-in / all-out)
            if last_state == 0 and desired == 1:
                # Buy with all cash, accounting for commission so cash can't go negative.
                denom = fill_px * (1.0 + self.cost_rate)
                if denom <= 0:
                    qty = 0.0
                else:
                    qty = cash / denom
                if not self.allow_fractional:
                    qty = float(np.floor(qty))
                trade_value = qty * fill_px
                commission = trade_value * self.cost_rate
                cash -= (trade_value + commission)
                position = qty
                trades.append(
                    {
                        "date": date,
                        "side": "BUY",
                        "qty": qty,
                        "price": fill_px,
                        "commission": commission,
                        "cash": cash,
                        "position": position,
                    }
                )
            elif last_state == 1 and desired == 0:
                # Sell all
                trade_value = position * fill_px
                commission = trade_value * self.cost_rate
                proceeds = trade_value - commission
                cash += proceeds
                trades.append(
                    {
                        "date": date,
                        "side": "SELL",
                        "qty": position,
                        "price": fill_px,
                        "commission": commission,
                        "cash": cash,
                        "position": 0.0,
                    }
                )
                position = 0.0
            # Mark-to-market equity at configured price (close or open)
            if self.mark_to_market == "close":
                mtm_px = float(actual_df.loc[date, "Close"])
            else:
                mtm_px = float(actual_df.loc[date, "Open"])
            equity.loc[date] = cash + position * mtm_px
            last_state = desired

        return trades, equity


def run_backtest(
    actual_df: pd.DataFrame,
    signals: Union[pd.Series, pd.DataFrame],
    initial_cash: float = 100.0,
    cost_bps: float = 10.0,
    slippage_bps: float = 5.0,
    allow_fractional: bool = True,
    execution_price: str = "open",
    mark_to_market: str = "close",
) -> Tuple[List[Dict[str, Any]], pd.Series]:
    """
    Convenience wrapper that instantiates BacktestEngine and runs it.
    """
    engine = BacktestEngine(
        initial_cash=initial_cash,
        cost_bps=cost_bps,
        slippage_bps=slippage_bps,
        allow_fractional=allow_fractional,
        execution_price=execution_price,
        mark_to_market=mark_to_market,
    )
    return engine.run(actual_df, signals)
