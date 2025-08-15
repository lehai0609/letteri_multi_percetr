"""
TEMA indicator utilities.

Provides:
- ema(series, period): exponential moving average (pandas Series -> pandas Series)
- tema(series, period): triple exponential moving average
- tema_ohlc(df, period): convenience to compute TEMA for OHLC columns
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd


__all__ = ["ema", "tema", "tema_ohlc"]


def ema(series: pd.Series, period: int) -> pd.Series:
    """
    Compute the Exponential Moving Average (EMA) for a pandas Series.

    Parameters
    ----------
    series : pd.Series
        Input time series (numeric). Index is preserved.
    period : int
        EMA period (span). Must be a positive integer.

    Returns
    -------
    pd.Series
        EMA series aligned to the input index.
    """
    if period is None or period <= 0:
        raise ValueError("period must be a positive integer")
    # Ensure float dtype for numeric stability
    s = series.astype(float)
    # use span=period (alpha = 2/(period+1)), recursive calculation (adjust=False)
    return s.ewm(span=period, adjust=False).mean()


def tema(series: pd.Series, period: int) -> pd.Series:
    """
    Compute the Triple Exponential Moving Average (TEMA) for a pandas Series.

    TEMA = 3*EMA1 - 3*EMA2 + EMA3
    where EMA1 = EMA(series), EMA2 = EMA(EMA1), EMA3 = EMA(EMA2)

    Parameters
    ----------
    series : pd.Series
        Input numeric series.
    period : int
        Period for each EMA stage.

    Returns
    -------
    pd.Series
        TEMA series aligned to input index.
    """
    if period is None or period <= 0:
        raise ValueError("period must be a positive integer")
    ema1 = ema(series, period)
    ema2 = ema(ema1, period)
    ema3 = ema(ema2, period)
    return 3.0 * ema1 - 3.0 * ema2 + ema3


def tema_ohlc(df: pd.DataFrame, period: int = 3) -> pd.DataFrame:
    """
    Compute TEMA for OHLC columns in a DataFrame.

    Only the columns ['Open','High','Low','Close'] are processed (if present).
    Returns a DataFrame with the same index and those columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing OHLC columns.
    period : int
        TEMA period.

    Returns
    -------
    pd.DataFrame
    """
    out = pd.DataFrame(index=df.index)
    for col in ("Open", "High", "Low", "Close"):
        if col in df.columns:
            out[col] = tema(df[col].astype(float), period)
    return out
