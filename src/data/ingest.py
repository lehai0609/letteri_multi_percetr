"""
DataIngestor - simple data ingestion utilities.

Provides:
- DataIngestor: load OHLCV data from a CSV file or yfinance.
- validate_calendar: ensure the DataFrame index is a trading calendar (uses src.utils.dates).
- Minimal defensive behavior to keep Milestone 2 moving: caching and optional local csv source.

Contracts (follow configs/default.yaml):
- Returned DataFrame index: DatetimeIndex (trading days), columns = ["Open","High","Low","Close","Adj Close","Volume"]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import logging

import pandas as pd

from src.utils import dates as du

Logger = logging.getLogger(__name__)

try:
    import yfinance as yf  # type: ignore
except Exception:  # pragma: no cover - optional
    yf = None  # type: ignore


REQUIRED_COLS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


@dataclass
class DataIngestor:
    """
    DataIngestor loads OHLCV time series from either a local CSV or yfinance.

    Args:
        source: "csv" or "yfinance"
        path_or_ticker: filesystem path (for csv) or ticker symbol (for yfinance)
        start_date/end_date: optional date-like strings (inclusive)
        cache_path: optional path to write/read a cached CSV
    """

    source: str
    path_or_ticker: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    cache_path: Optional[str] = None
    calendar: Optional[str] = "NYSE"

    def fetch(self) -> pd.DataFrame:
        """
        Fetch the data and return a DataFrame with the required schema.

        Raises:
            RuntimeError on unsupported source or missing provider (yfinance not installed).
            ValueError if required columns missing.
        """
        if self.source == "csv":
            df = self._from_csv(self.path_or_ticker)
        elif self.source == "yfinance":
            if yf is None:
                raise RuntimeError("yfinance is not installed; cannot fetch remote data")
            df = self._from_yfinance(self.path_or_ticker, self.start_date, self.end_date)
        else:
            raise RuntimeError(f"Unsupported source: {self.source}")

        # Ensure index is tz-naive, normalized DatetimeIndex and columns exist
        df = self._normalize_df(df)

        # Validate calendar monotonicity/duplicates
        self.validate_calendar(df)

        # Optionally cache to CSV if requested
        if self.cache_path:
            try:
                df.to_csv(self.cache_path, index=True)
            except Exception:
                Logger.exception("Failed to write cache to %s", self.cache_path)

        return df

    def _from_csv(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path, parse_dates=True, index_col=0)
        return df

    def _from_yfinance(self, ticker: str, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
        # yfinance returns a DataFrame indexed by Date with columns Open/High/Low/Close/Adj Close/Volume
        # Use period if start/end are None
        kwargs = {}
        if start:
            kwargs["start"] = start
        if end:
            kwargs["end"] = end
        df = yf.download(ticker, progress=False, **kwargs)
        # yfinance may name the adjusted close column "Adj Close" already
        return df

    def _normalize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        # Ensure index is DatetimeIndex, tz-naive and normalized
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                raise ValueError("Could not parse index to DatetimeIndex") from e

        # Strip timezone and normalize to midnight
        if getattr(df.index, "tz", None) is not None:
            try:
                df.index = df.index.tz_localize(None)
            except Exception:
                df.index = df.index.tz_convert(None)

        df.index = df.index.normalize()
        # Ensure required columns exist (case-sensitive)
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required OHLCV columns: {missing}")
        # Keep only required columns (preserve order)
        df = df.loc[:, REQUIRED_COLS].copy()
        # Sort index to be monotonic increasing
        df = df.sort_index()
        return df

    def validate_calendar(self, df: pd.DataFrame) -> None:
        """
        Validate the DataFrame index as a trading calendar.

        Checks:
        - DatetimeIndex, monotonic increasing, no duplicates (delegates to du.ensure_datetime_index)
        - Detect weekend gaps (non-business-day gaps) and log a warning if larger-than-expected gaps are found.

        This function does NOT automatically fill or backfill data; that is handled elsewhere.
        """
        # Basic structural checks
        du.ensure_datetime_index(df)

        # Identify gaps larger than one business day (using du.trading_days)
        # For each adjacent pair, ensure the next trading day equals expected next trading day
        idx = df.index
        if len(idx) < 2:
            return

        # Build expected trading days between first and last using the calendar utilities
        expected = du.trading_days(idx[0], idx[-1], calendar=self.calendar)
        # If expected length differs from observed, log details
        if len(expected) != len(idx):
            # Find missing days
            missing = expected.difference(idx)
            if len(missing) > 0:
                Logger.warning(
                    "Detected %d missing trading days between %s and %s. Example missing: %s",
                    len(missing),
                    idx[0].date(),
                    idx[-1].date(),
                    list(missing[:5]),
                )
            # Also see if there are unexpected extra days (rare)
            extra = idx.difference(expected)
            if len(extra) > 0:
                Logger.warning(
                    "Detected %d extra days (non-trading) in index between %s and %s. Example extra: %s",
                    len(extra),
                    idx[0].date(),
                    idx[-1].date(),
                    list(extra[:5]),
                )
