"""
PriceAdjuster - adjust OHLC using the Adj Close factor.

Provides:
- PriceAdjuster.adjust_ohlc(df): returns a DataFrame where Open/High/Low/Close
  have been adjusted by the factor (Adj Close / Close) on each row.

Behavior:
- Keeps columns in order: ["Open","High","Low","Close","Adj Close","Volume"]
- Does not forward/backfill; rows with NaN in Close or Adj Close will raise.
- Preserves index and dtype where reasonable.
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

Logger = logging.getLogger(__name__)

REQUIRED_COLS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


class PriceAdjuster:
    @staticmethod
    def adjust_ohlc(df: pd.DataFrame) -> pd.DataFrame:
        """
        Adjust Open/High/Low/Close by the per-row adjustment factor derived from Adj Close.

        factor = Adj Close / Close

        Args:
            df: DataFrame with REQUIRED_COLS present.

        Returns:
            DataFrame with adjusted Open/High/Low/Close (floats), keeping Adj Close and Volume.

        Raises:
            ValueError if required columns are missing or if Close contains zeros/NaNs that prevent factor computation.
        """
        # Basic validation
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns for adjustment: {missing}")

        out = df.copy()

        # Ensure numeric types for computation
        for c in ["Open", "High", "Low", "Close", "Adj Close"]:
            out[c] = pd.to_numeric(out[c], errors="coerce")

        # Identify problematic rows
        if out["Close"].isna().any() or out["Adj Close"].isna().any():
            raise ValueError("Close and Adj Close must not contain NaN values for adjustment.")

        if (out["Close"] == 0).any():
            raise ValueError("Close contains zero values; cannot compute adjustment factor.")

        factor = out["Adj Close"] / out["Close"]

        # Broadcast multiply OHLC by factor (elementwise)
        out["Open"] = out["Open"] * factor
        out["High"] = out["High"] * factor
        out["Low"] = out["Low"] * factor
        out["Close"] = out["Close"] * factor

        # Ensure no Inf/Nan introduced
        if np.isinf(out[["Open", "High", "Low", "Close"]].to_numpy()).any():
            raise ValueError("Infinite values found after adjustment.")

        # Reorder columns to canonical schema
        cols: Sequence[str] = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        out = out.loc[:, cols]

        return out
