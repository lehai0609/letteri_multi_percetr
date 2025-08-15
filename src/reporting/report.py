# Reporting helpers for the Letteri replication.
#
# Provides small, dependency-light helpers to persist tables (CSV/JSON)
# and generate a couple of diagnostic plots (matplotlib).
#
# Primary API:
#   - build_report(output_dir, *, pred_df, actual_df, forecast_metrics,
#                  strategy_results: Dict[str, dict], overwrite=True)
#
# Where strategy_results is a mapping from strategy name -> dict with keys:
#   - "trades" : list[dict]   (trade dicts as produced by BacktestEngine)
#   - "equity"  : pd.Series   (equity series indexed by date)
#   - "metrics" : dict        (risk/backtest metrics)
#
# The function will:
#   - save predicted_ohlc.csv and forecast_metrics.json if provided
#   - for each strategy save: equity_{name}.csv, metrics_{name}.json, trades_{name}.json
#   - produce plots: equity_{name}.png and predicted_vs_actual.png (if both dfs available)
#
# The module is intentionally small and avoids heavy third-party dependencies
# beyond pandas and matplotlib (already in project requirements).

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)


def save_json(obj: Any, path: str) -> None:
    _ensure_dir(path)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)


def save_equity_csv(equity: pd.Series, path: str) -> None:
    """Persist an equity series to CSV. The CSV will have a header 'equity'."""
    _ensure_dir(path)
    if isinstance(equity, pd.Series):
        df = equity.rename("equity").to_frame()
    else:
        df = pd.Series(equity).rename("equity").to_frame()
    df.to_csv(path)


def _serialize_trades(trades: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for tr in trades:
        tr2 = dict(tr)
        d = tr2.get("date")
        # Try to convert Timestamp / datetime to ISO-format string
        try:
            tr2["date"] = pd.Timestamp(d).isoformat()
        except Exception:
            tr2["date"] = str(d)
        out.append(tr2)
    return out


def plot_equity_curve(equity: pd.Series, path: str, title: Optional[str] = None) -> None:
    """Simple equity curve plot (PNG)."""
    _ensure_dir(path)
    if equity is None or len(equity.dropna()) == 0:
        logger.debug("Skipping equity plot; empty series for %s", path)
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(equity.index, equity.values, linewidth=2, label="Equity")
    ax.set_ylabel("Portfolio value")
    ax.set_xlabel("Date")
    if title:
        ax.set_title(title)
    ax.grid(True, linestyle=":", linewidth=0.7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_predicted_vs_actual(
    actual_df: Optional[pd.DataFrame], pred_df: Optional[pd.DataFrame], path: str, series: str = "Close"
) -> None:
    """Plot predicted vs actual line chart for a chosen series (default: Close)."""
    _ensure_dir(path)
    if actual_df is None or pred_df is None:
        logger.debug("Skipping predicted-vs-actual plot; missing data")
        return
    # Align on intersection of indices
    idx = actual_df.index.intersection(pred_df.index)
    if len(idx) == 0:
        logger.debug("Skipping predicted-vs-actual plot; no overlapping dates")
        return

    a = actual_df.loc[idx, series].astype(float)
    p = pred_df.loc[idx, series].astype(float)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(a.index, a.values, label=f"actual {series}", color="#222222", linewidth=1.5)
    ax.plot(p.index, p.values, label=f"predicted {series}", color="#d62728", linestyle="--", linewidth=1.5)
    ax.set_ylabel(series)
    ax.set_xlabel("Date")
    ax.set_title(f"Predicted vs Actual ({series})")
    ax.grid(True, linestyle=":", linewidth=0.7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_report(
    output_dir: str,
    *,
    pred_df: Optional[pd.DataFrame] = None,
    actual_df: Optional[pd.DataFrame] = None,
    forecast_metrics: Optional[Dict[str, Any]] = None,
    strategy_results: Optional[Dict[str, Dict[str, Any]]] = None,
    overwrite: bool = True,
) -> Dict[str, Any]:
    """Assemble and persist a report.

    Returns a dictionary mapping file roles -> path for the files created.
    """
    os.makedirs(output_dir, exist_ok=True)
    created: Dict[str, Any] = {"output_dir": os.path.abspath(output_dir)}
    # Save predicted ohlc
    if pred_df is not None:
        ppath = os.path.join(output_dir, "predicted_ohlc.csv")
        if overwrite or not os.path.exists(ppath):
            pred_df.to_csv(ppath)
        created["predicted_ohlc"] = ppath

    # Save forecast metrics
    if forecast_metrics is not None:
        mpath = os.path.join(output_dir, "forecast_metrics.json")
        if overwrite or not os.path.exists(mpath):
            save_json(forecast_metrics, mpath)
        created["forecast_metrics"] = mpath

    # Plot predicted vs actual Close for quick visual
    pva_path = os.path.join(output_dir, "predicted_vs_actual_close.png")
    try:
        plot_predicted_vs_actual(actual_df, pred_df, pva_path, series="Close")
        created["predicted_vs_actual_close"] = pva_path
    except Exception as exc:
        logger.warning("Failed to create predicted-vs-actual plot: %s", exc)

    # Strategy-specific outputs
    strategy_results = strategy_results or {}
    for strat_name, res in strategy_results.items():
        safe_name = strat_name.replace(" ", "_").lower()
        # Equity CSV + PNG
        equity = res.get("equity")
        equity_csv = os.path.join(output_dir, f"equity_{safe_name}.csv")
        try:
            if equity is not None:
                save_equity_csv(equity, equity_csv)
                created[f"equity_csv_{safe_name}"] = equity_csv
                plot_equity_curve(equity, os.path.join(output_dir, f"equity_{safe_name}.png"), title=f"Equity - {strat_name}")
                created[f"equity_png_{safe_name}"] = os.path.join(output_dir, f"equity_{safe_name}.png")
        except Exception as exc:
            logger.warning("Failed to save/plot equity for %s: %s", strat_name, exc)

        # Metrics JSON
        metrics = res.get("metrics")
        if metrics is not None:
            metrics_path = os.path.join(output_dir, f"metrics_{safe_name}.json")
            try:
                save_json(metrics, metrics_path)
                created[f"metrics_json_{safe_name}"] = metrics_path
            except Exception as exc:
                logger.warning("Failed to save metrics for %s: %s", strat_name, exc)

        # Trades JSON
        trades = res.get("trades")
        if trades is not None:
            trades_path = os.path.join(output_dir, f"trades_{safe_name}.json")
            try:
                save_json(_serialize_trades(trades), trades_path)
                created[f"trades_json_{safe_name}"] = trades_path
            except Exception as exc:
                logger.warning("Failed to save trades for %s: %s", strat_name, exc)

    return created
