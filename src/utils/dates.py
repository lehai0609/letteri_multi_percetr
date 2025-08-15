"""
Trading calendar utilities.

This module centralizes trading day calculations so the rest of the system
can assume consistent, gap-free trading calendars.

Features:
- Optional use of pandas-market-calendars ("NYSE") for accurate exchange days.
- Fallback to pandas CustomBusinessDay with USFederalHolidayCalendar.
- Last-resort fallback to plain weekdays (Mon-Fri) if holidays are unavailable.
- Helpers for parsing dates, getting trading ranges, horizon windows,
  checking/aligning a date to the nearest trading day, and simple index validation.

Typical usage:
    from src.utils import dates as du

    # Next 30 trading days after a cutoff (exclusive)
    horizon = du.get_horizon_dates("2021-10-16", horizon=30)

    # Validate a DataFrame index is a monotonic DatetimeIndex
    du.ensure_datetime_index(df)

    # Check if a specific date is a trading day
    if du.is_trading_day("2021-10-18"):
        ...

Notes:
- All returned timestamps are timezone-naive and normalized to midnight.
- Calendar parameter defaults to "NYSE"; pass holidays=... to override or extend.
"""

from __future__ import annotations

import logging as py_logging
from typing import Iterable, Optional, Sequence, Union

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import BDay, CustomBusinessDay

try:
    import pandas_market_calendars as mcal  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    mcal = None  # type: ignore

Logger = py_logging.getLogger(__name__)


DateLike = Union[str, pd.Timestamp, pd.Timestamp, "datetime.date", "datetime.datetime"]


def parse_date(value: DateLike) -> pd.Timestamp:
    """
    Parse a date-like into a tz-naive, normalized pandas Timestamp (00:00).
    """
    ts = pd.Timestamp(value)
    # Normalize to date (00:00 time); strip tz if present
    if ts.tz is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def ensure_datetime_index(obj: Union[pd.DataFrame, pd.Series], *, name: Optional[str] = None) -> None:
    """
    Validate that obj.index is a DatetimeIndex, monotonic increasing, no duplicates.

    Raises:
        TypeError, ValueError
    """
    idx = obj.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError(f"Expected a DatetimeIndex, got {type(idx)}")
    if not idx.is_monotonic_increasing:
        raise ValueError("DatetimeIndex must be monotonic increasing")
    if idx.has_duplicates:
        raise ValueError("DatetimeIndex contains duplicate entries")
    if name and obj.index.name and obj.index.name != name:
        Logger.debug("Index name '%s' != expected '%s' (not fatal)", obj.index.name, name)


def _business_day_offset(
    holidays: Optional[Iterable[DateLike]] = None,
) -> CustomBusinessDay:
    """
    Build a CustomBusinessDay offset with best available holiday info.
    """
    if holidays is not None:
        hol = pd.to_datetime(list(holidays))
        hol = pd.DatetimeIndex(hol).tz_localize(None).normalize()
        return CustomBusinessDay(holidays=hol)
    # Try US Federal holiday calendar as an approximation of US markets
    try:
        return CustomBusinessDay(calendar=USFederalHolidayCalendar())
    except Exception:  # pragma: no cover - extremely rare
        Logger.warning("Falling back to plain weekdays (Mon-Fri) without holiday exclusions.")
        return BDay()


def _valid_days_mcal(
    start: pd.Timestamp, end: pd.Timestamp, calendar: str = "NYSE"
) -> Optional[pd.DatetimeIndex]:
    """
    Use pandas-market-calendars if available to get valid trading days.
    Returns None if mcal not installed or calendar not found.
    """
    if mcal is None:  # pragma: no cover - optional dependency
        return None
    try:
        cal = mcal.get_calendar(calendar)
    except Exception:  # pragma: no cover
        Logger.warning("pandas-market-calendars calendar '%s' not found; falling back.", calendar)
        return None
    # valid_days returns a tz-aware DatetimeIndex (UTC) or tz-naive depending on version
    days = cal.valid_days(start_date=start, end_date=end)
    try:
        # Convert to tz-naive if tz-aware
        if getattr(days, "tz", None) is not None:
            days = days.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    return pd.DatetimeIndex(days).normalize()


def trading_days(
    start: DateLike,
    end: DateLike,
    *,
    calendar: Optional[str] = "NYSE",
    holidays: Optional[Iterable[DateLike]] = None,
) -> pd.DatetimeIndex:
    """
    Return all trading days from start to end inclusive.

    Prefers pandas-market-calendars (if available) for the given `calendar`.
    Otherwise uses a CustomBusinessDay with USFederalHolidayCalendar (or provided holidays),
    and finally plain weekdays if nothing else is available.

    Args:
        start, end: date-like (inclusive bounds)
        calendar: exchange calendar name for pandas-market-calendars (e.g., "NYSE")
        holidays: optional iterable of holiday dates to exclude (overrides default)

    Returns:
        DatetimeIndex (tz-naive, normalized)
    """
    s = parse_date(start)
    e = parse_date(end)
    if s > e:
        s, e = e, s

    # Try precise exchange calendar
    if calendar is not None:
        idx = _valid_days_mcal(s, e, calendar=calendar)
        if idx is not None:
            return idx

    # Fallback to business days with holiday exclusions
    offset = _business_day_offset(holidays=holidays)
    try:
        rng = pd.date_range(start=s, end=e, freq=offset, inclusive="both")
    except TypeError:
        # pandas < 1.4 compatibility: no 'inclusive' parameter
        rng = pd.date_range(start=s, end=e, freq=offset)
        # Ensure within bounds
        rng = rng[(rng >= s) & (rng <= e)]
    return pd.DatetimeIndex(rng).normalize()


def is_trading_day(
    date: DateLike,
    *,
    calendar: Optional[str] = "NYSE",
    holidays: Optional[Iterable[DateLike]] = None,
) -> bool:
    """
    True if the given date is a trading day (per configured calendar).
    """
    d = parse_date(date)
    idx = trading_days(d, d, calendar=calendar, holidays=holidays)
    return len(idx) == 1 and idx[0] == d


def align_to_trading_day(
    date: DateLike,
    *,
    direction: str = "next",  # "next" | "previous" | "nearest"
    calendar: Optional[str] = "NYSE",
    holidays: Optional[Iterable[DateLike]] = None,
    search_days: int = 365,
) -> pd.Timestamp:
    """
    Align a date to a trading day.

    Args:
        date: input date-like
        direction: "next", "previous", or "nearest"
        search_days: bounds the search window to avoid unbounded loops

    Returns:
        tz-naive, normalized Timestamp
    """
    d = parse_date(date)
    if is_trading_day(d, calendar=calendar, holidays=holidays):
        return d

    if direction not in {"next", "previous", "nearest"}:
        raise ValueError("direction must be one of {'next','previous','nearest'}")

    if direction == "next":
        future = trading_days(d + pd.Timedelta(days=1), d + pd.Timedelta(days=search_days),
                              calendar=calendar, holidays=holidays)
        if len(future) == 0:
            raise ValueError("No future trading day found within search_days")
        return future[0]
    elif direction == "previous":
        past = trading_days(d - pd.Timedelta(days=search_days), d - pd.Timedelta(days=1),
                            calendar=calendar, holidays=holidays)
        if len(past) == 0:
            raise ValueError("No previous trading day found within search_days")
        return past[-1]
    else:  # nearest
        prev = align_to_trading_day(d, direction="previous", calendar=calendar, holidays=holidays, search_days=search_days)
        nxt = align_to_trading_day(d, direction="next", calendar=calendar, holidays=holidays, search_days=search_days)
        # Choose the closer one; prefer previous on tie
        if (d - prev) <= (nxt - d):
            return prev
        return nxt


def next_trading_days(
    start: DateLike,
    n: int,
    *,
    include_start: bool = False,
    calendar: Optional[str] = "NYSE",
    holidays: Optional[Iterable[DateLike]] = None,
    lookahead_days: int = 370,  # ensures enough coverage for typical n<=252
) -> pd.DatetimeIndex:
    """
    Get the next n trading days from start.

    Args:
        include_start: if True and start is a trading day, include it as day 1.

    Returns:
        DatetimeIndex of length n.
    """
    s = parse_date(start)
    if include_start:
        first = align_to_trading_day(s, direction="next" if not is_trading_day(s, calendar=calendar, holidays=holidays) else "nearest",
                                     calendar=calendar, holidays=holidays)
        # Build range large enough, then slice
        rng = trading_days(first, first + pd.Timedelta(days=lookahead_days),
                           calendar=calendar, holidays=holidays)
        if len(rng) < n:
            raise ValueError(f"Not enough trading days after {s} to get {n}")
        return rng[:n]
    else:
        # strictly after the start date
        rng = trading_days(s + pd.Timedelta(days=1), s + pd.Timedelta(days=lookahead_days),
                           calendar=calendar, holidays=holidays)
        if len(rng) < n:
            raise ValueError(f"Not enough trading days after {s} to get {n}")
        return rng[:n]


def previous_trading_days(
    end: DateLike,
    n: int,
    *,
    include_end: bool = False,
    calendar: Optional[str] = "NYSE",
    holidays: Optional[Iterable[DateLike]] = None,
    lookback_days: int = 370,
) -> pd.DatetimeIndex:
    """
    Get the previous n trading days ending at or before 'end'.

    Args:
        include_end: if True and end is a trading day, include it as the last element.

    Returns:
        DatetimeIndex of length n in ascending order.
    """
    e = parse_date(end)
    if include_end:
        last = align_to_trading_day(e, direction="previous" if not is_trading_day(e, calendar=calendar, holidays=holidays) else "nearest",
                                    calendar=calendar, holidays=holidays)
        rng = trading_days(last - pd.Timedelta(days=lookback_days), last,
                           calendar=calendar, holidays=holidays)
        if len(rng) < n:
            raise ValueError(f"Not enough trading days before {e} to get {n}")
        return rng[-n:]
    else:
        rng = trading_days(e - pd.Timedelta(days=lookback_days), e - pd.Timedelta(days=1),
                           calendar=calendar, holidays=holidays)
        if len(rng) < n:
            raise ValueError(f"Not enough trading days before {e} to get {n}")
        return rng[-n:]


def add_trading_days(
    date: DateLike,
    n: int,
    *,
    calendar: Optional[str] = "NYSE",
    holidays: Optional[Iterable[DateLike]] = None,
) -> pd.Timestamp:
    """
    Add n trading days to a date (n can be negative).

    Returns:
        tz-naive, normalized Timestamp
    """
    d = parse_date(date)
    if n == 0:
        return align_to_trading_day(d, direction="nearest", calendar=calendar, holidays=holidays)
    if n > 0:
        idx = next_trading_days(d, n, include_start=False, calendar=calendar, holidays=holidays)
        return idx[-1]
    else:
        idx = previous_trading_days(d, -n, include_end=False, calendar=calendar, holidays=holidays)
        return idx[0]


def get_horizon_dates(
    cutoff: DateLike,
    *,
    horizon: int = 30,
    include_cutoff: bool = False,
    calendar: Optional[str] = "NYSE",
    holidays: Optional[Iterable[DateLike]] = None,
) -> pd.DatetimeIndex:
    """
    Convenience: next N trading days around a cutoff.

    Typically used to create the 30-day validation/test window after a training cutoff.

    Args:
        cutoff: cutoff date (train ends here)
        horizon: number of trading days
        include_cutoff: if True and cutoff is a trading day, include it as day 1; defaults to False

    Returns:
        DatetimeIndex of length = horizon
    """
    return next_trading_days(
        cutoff,
        horizon,
        include_start=include_cutoff,
        calendar=calendar,
        holidays=holidays,
    )


# Demonstration when run directly
if __name__ == "__main__":  # pragma: no cover - manual demo
    Logger.setLevel(py_logging.DEBUG)
    c = "NYSE"
    cutoff = "2021-10-16"
    print("Cutoff:", cutoff)
    print("Is trading day?", is_trading_day(cutoff, calendar=c))
    horizon = get_horizon_dates(cutoff, horizon=10, calendar=c)
    print("Next 10 trading days after cutoff:\n", horizon)
    prev5 = previous_trading_days("2021-10-20", 5, include_end=True, calendar=c)
    print("Previous 5 trading days up to 2021-10-20:\n", prev5)
    aligned = align_to_trading_day("2021-10-16", direction="next", calendar=c)
    print("Aligned next trading day:", aligned)