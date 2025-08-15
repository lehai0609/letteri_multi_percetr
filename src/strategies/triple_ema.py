"""
Triple-EMA based strategy generator.

This module provides:
- evaluate_operator_expression(expr: str, vars: dict) -> bool
    Safely evaluate the operator precedence expression for a single row given numeric variables.
- generate_signals(predicted_df, period=None, operator_precedence=None, cfg_path="configs/default.yaml")
    Compute TEMA on predicted OHLC and produce a stateful signal series (1 = long, 0 = flat)
    using the provided operator precedence expression for entry; the exit condition is the
    symmetric comparison (replace '<' with '>').

Notes
-----
The operator_precedence expression is expected to reference the tokens:
    Low, High, Close, Open, TEMA_Low, TEMA_High, TEMA_Close, TEMA_Open
and to use Python boolean operators `and` / `or` and comparison operators (`<`, `>`).
"""
from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from src.indicators.tema import tema_ohlc
from src.utils.config import load_config

__all__ = ["generate_signals", "evaluate_operator_expression"]


def evaluate_operator_expression(expr: str, vars_map: Dict[str, float]) -> bool:
    """
    Evaluate the operator precedence expression for a single row.

    Parameters
    ----------
    expr : str
        Expression like "((Low<TEMA_Low or High<TEMA_High) and (Close<TEMA_Close or Open<TEMA_Open))"
    vars_map : dict
        Mapping of token names to numeric values, e.g.
        {
            "Low": 10.0,
            "TEMA_Low": 11.2,
            ...
        }

    Returns
    -------
    bool
        Result of the boolean expression. Returns False on any evaluation error.
    """
    # Minimal safety: no builtins, only the provided vars are visible.
    try:
        result = eval(expr, {"__builtins__": None}, vars_map)
        return bool(result)
    except Exception:
        # Any error (NaN comparisons, malformed expr) treated as False for entry/exit logic
        return False


def _build_exit_expression(entry_expr: str) -> str:
    """
    Build an exit expression from entry by swapping '<' -> '>'.
    This is a simple heuristic that mirrors the TDD: entry uses < comparisons, exit uses >.
    """
    return entry_expr.replace("<", ">")


def generate_signals(
    predicted_df: pd.DataFrame,
    period: Optional[int] = None,
    operator_precedence: Optional[str] = None,
    cfg_path: str = "configs/default.yaml",
) -> pd.DataFrame:
    """
    Generate Triple-EMA-based signals from predicted OHLC.

    Parameters
    ----------
    predicted_df : pd.DataFrame
        Predicted OHLC with columns ['Open','High','Low','Close'] and DatetimeIndex.
    period : int, optional
        TEMA period. If None, loaded from configuration.
    operator_precedence : str, optional
        Entry boolean expression. If None, loaded from configuration.
    cfg_path : str
        Path to configuration file for defaults.

    Returns
    -------
    pd.DataFrame
        DataFrame with single column "signal" (0 or 1) indexed like predicted_df.
    """
    # Validate input columns
    for required in ("Open", "High", "Low", "Close"):
        if required not in predicted_df.columns:
            raise ValueError(f"predicted_df missing required column: {required}")

    # Load defaults from config if needed
    cfg = None
    if period is None or operator_precedence is None:
        cfg = load_config(cfg_path)
    if period is None:
        period = int(cfg["model"]["tema_period"])
    if operator_precedence is None:
        operator_precedence = str(cfg["model"]["operator_precedence"])

    # Compute TEMA for OHLC
    tema_df = tema_ohlc(predicted_df, period=period)

    entry_expr = operator_precedence
    exit_expr = _build_exit_expression(entry_expr)

    idx = predicted_df.index
    signals = pd.Series(index=idx, dtype=int)

    holding = False
    for ts in idx:
        row = predicted_df.loc[ts]
        tema_row = tema_df.loc[ts]

        vars_map = {
            "Low": float(row["Low"]),
            "High": float(row["High"]),
            "Close": float(row["Close"]),
            "Open": float(row["Open"]),
            "TEMA_Low": float(tema_row["Low"]) if pd.notna(tema_row["Low"]) else float("nan"),
            "TEMA_High": float(tema_row["High"]) if pd.notna(tema_row["High"]) else float("nan"),
            "TEMA_Close": float(tema_row["Close"]) if pd.notna(tema_row["Close"]) else float("nan"),
            "TEMA_Open": float(tema_row["Open"]) if pd.notna(tema_row["Open"]) else float("nan"),
        }

        entry_cond = evaluate_operator_expression(entry_expr, vars_map)
        exit_cond = evaluate_operator_expression(exit_expr, vars_map)

        if not holding and entry_cond:
            holding = True
            signals.at[ts] = 1
        elif holding and exit_cond:
            holding = False
            signals.at[ts] = 0
        else:
            signals.at[ts] = 1 if holding else 0

    return pd.DataFrame({"signal": signals})
