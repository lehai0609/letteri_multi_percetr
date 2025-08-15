"""
Simple threshold strategy.

Entry rule:
  - Enter long when predicted Close > predicted Open (and we are flat).
Exit rule:
  - Exit (go flat) when predicted Close <= predicted Open (and we are long).

Output:
  - DataFrame indexed like predicted_df with a single column "signal" containing 1 for long and 0 for flat.
"""
from __future__ import annotations

from typing import Sequence

import pandas as pd


__all__ = ["generate_signals"]


def generate_signals(predicted_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate threshold-based signals from a predicted OHLC DataFrame.

    Parameters
    ----------
    predicted_df : pd.DataFrame
        DataFrame containing at least 'Open' and 'Close' columns. Index represents forecast dates.

    Returns
    -------
    pd.DataFrame
        DataFrame with single column 'signal' (0 or 1) and the same index as predicted_df.
    """
    for required in ("Open", "Close"):
        if required not in predicted_df.columns:
            raise ValueError(f"predicted_df missing required column: {required}")

    idx = predicted_df.index
    signals = pd.Series(index=idx, dtype=int)

    holding = False
    # Iterate in index order to maintain stateful entry/exit semantics
    for ts in idx:
        open_px = float(predicted_df.at[ts, "Open"])
        close_px = float(predicted_df.at[ts, "Close"])

        if not holding and close_px > open_px:
            holding = True
            signals.at[ts] = 1
        elif holding and close_px <= open_px:
            holding = False
            signals.at[ts] = 0
        else:
            signals.at[ts] = 1 if holding else 0

    return pd.DataFrame({"signal": signals})
