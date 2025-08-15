#!/usr/bin/env python3
# "Train DNN MLPs (Milestone 7 + 9).
#
# Enhancements added for Milestone 9:
# - Optional time-series-aware grid search over hyperparameters using
#   sklearn.model_selection.TimeSeriesSplit (run with --grid).
# - EarlyStopping callbacks are used during fold training (restore_best_weights=True).
# - Best hyperparameters are persisted to the configured grid_search_path (models/grid_search.json).
#
# Usage:
#     python scripts/train.py --config configs/default.yaml
#     python scripts/train.py --all
#     python scripts/train.py --grid      # run grid search (may be slow)
#
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import List, Optional

# Ensure repo root is on sys.path so `src` imports work when running the script directly.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import gc
import itertools
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit

from src.utils.config import load_config
from src.data.scale import ScalerManager
from src.features.window import build_xy
from src.models.dnn import set_seeds, DNNRegressor, build_model

# optional split helper (we fall back to positional split if unavailable)
try:
    from src.data.split import split_by_cutoff  # type: ignore
except Exception:
    split_by_cutoff = None  # type: ignore

# Attempt to import Keras backend for clearing the session between folds.
try:
    import tensorflow as _tf

    from tensorflow.keras import backend as K  # type: ignore
except Exception:
    _tf = None  # type: ignore
    K = None  # type: ignore

OHLC = ["Open", "High", "Low", "Close"]


def generate_synthetic_ohlc(n_days: int = 120, start: str = "2020-01-01", seed: int = 1234) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_days)
    # generate log returns with small drift
    rets = rng.normal(loc=0.0002, scale=0.01, size=n_days)
    close = 100.0 * np.exp(np.cumsum(rets))
    open_ = np.empty_like(close)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    highs = np.maximum(open_, close) + np.abs(rng.normal(0, 0.2, size=n_days))
    lows = np.minimum(open_, close) - np.abs(rng.normal(0, 0.2, size=n_days))
    adj_close = close.copy()
    volume = rng.integers(100_000, 1_000_000, size=n_days)

    df = pd.DataFrame(
        {
            "Open": open_,
            "High": highs,
            "Low": lows,
            "Close": close,
            "Adj Close": adj_close,
            "Volume": volume,
        },
        index=idx,
    )
    return df.astype(float)


def train_one_series(
    train_df: pd.DataFrame,
    target: str,
    cfg: dict,
    scaler_mgr: ScalerManager,
    out_dir: str,
    best_params: Optional[dict] = None,
) -> dict:
    """
    Train a single DNN on `target` series and save the model to out_dir.
    If `best_params` is provided, it overrides the default hyperparameters.
    Returns a small summary dict.
    """
    t = int(cfg["data"]["lag_t"])
    dnn_cfg = cfg["dnn"]

    # Resolve hyperparameters (best_params overrides config)
    if best_params is not None:
        n = int(best_params.get("n", dnn_cfg.get("n_grid", [1])[0]))
        dropout_rate = float(best_params.get("dropout", cfg["model"]["dropout"]))
        l2 = float(best_params.get("l2", dnn_cfg.get("l2")))
        lr = float(best_params.get("lr", dnn_cfg.get("lr")))
        epochs = int(best_params.get("epochs", dnn_cfg.get("epochs")))
        batch_size = int(best_params.get("batch_size", dnn_cfg.get("batch_size")))
    else:
        n_grid = dnn_cfg.get("n_grid", [1])
        n = int(n_grid[0]) if isinstance(n_grid, (list, tuple)) else int(n_grid)
        dropout_rate = float(cfg["model"]["dropout"])
        l2 = float(dnn_cfg.get("l2"))
        lr = float(dnn_cfg.get("lr"))
        epochs = int(dnn_cfg.get("epochs"))
        batch_size = int(dnn_cfg.get("batch_size"))

    bs = int(dnn_cfg.get("bs"))
    early_stopping = bool(dnn_cfg.get("early_stopping"))
    patience = int(dnn_cfg.get("patience"))
    loss = str(dnn_cfg.get("loss"))
    optimizer = str(dnn_cfg.get("optimizer"))

    # Fit scaler on training series (caller is responsible for not leaking future)
    series_train = train_df[target]
    series_scaled = scaler_mgr.fit_transform(series_train, name=target)

    X, y = build_xy(series_scaled, t)
    if X.shape[0] == 0:
        raise RuntimeError(f"Not enough training samples for target {target} with t={t}")

    # create small validation split (10%) if possible
    val_size = max(1, int(0.1 * X.shape[0])) if X.shape[0] > 1 else 1
    if X.shape[0] > val_size:
        X_train, X_val = X[:-val_size], X[-val_size:]
        y_train, y_val = y[:-val_size], y[-val_size:]
        val_data = (X_val, y_val)
    else:
        X_train, y_train = X, y
        val_data = None

    # Set seeds for reproducibility per-series training
    seeds = cfg["model"]["seeds"]
    set_seeds(python_seed=int(seeds["python"]), numpy_seed=int(seeds["numpy"]), tf_seed=int(seeds["tensorflow"]))

    # Build regressor
    reg = DNNRegressor(
        build_fn=build_model,
        input_dim=t,
        n=n,
        t=t,
        bs=bs,
        dropout_rate=float(dropout_rate),
        lr=float(lr),
        l2=float(l2),
        epochs=int(epochs),
        batch_size=int(batch_size),
        early_stopping=bool(early_stopping),
        patience=int(patience),
        loss=loss,
        optimizer=optimizer,
    )

    logging.info("Training model for %s: input_dim=%d n=%d bs=%d epochs=%d batch_size=%d lr=%s l2=%s dropout=%s",
                 target, t, n, bs, reg.epochs, reg.batch_size, lr, l2, dropout_rate)
    history = reg.fit(X_train, y_train, validation_data=val_data, verbose=1)
    model_path = os.path.join(out_dir, f"dnn_{target}.keras")
    reg.save(model_path)

    # Clear Keras session to free memory
    try:
        if K is not None:
            K.clear_session()
    except Exception:
        pass
    gc.collect()

    summary = {
        "target": target,
        "n_samples": int(X.shape[0]),
        "history_keys": list(history.keys()) if isinstance(history, dict) else [],
        "model_path": model_path,
        "params": {"n": n, "bs": bs, "dropout": dropout_rate, "lr": lr, "l2": l2, "epochs": epochs, "batch_size": batch_size},
    }
    return summary


def run_grid_search(train_df: pd.DataFrame, cfg: dict, scaler_mgr: ScalerManager, save_path: str) -> dict:
    """
    Perform a manual grid search using TimeSeriesSplit on the 'Close' series.
    - Fits scalers only on the training data provided (no leakage).
    - Uses TimeSeriesSplit(cv_splits) to create training/validation folds.
    - Returns and persists the best parameter set (minimizing mean MAE across folds).
    """
    logging.info("Starting TimeSeries grid search (may take a while)...")
    start_time = time.time()
    dnn_cfg = cfg["dnn"]
    grid_cfg = dnn_cfg.get("grid", {})
    cv_splits = int(dnn_cfg.get("cv_splits", 3))
    t = int(cfg["data"]["lag_t"])

    # Resolve candidate lists (with fallbacks)
    n_list = list(grid_cfg.get("n", dnn_cfg.get("n_grid", [1])))
    dropout_list = list(grid_cfg.get("dropout", [cfg["model"]["dropout"]]))
    l2_list = list(grid_cfg.get("l2", [dnn_cfg.get("l2")]))
    lr_list = list(grid_cfg.get("lr", [dnn_cfg.get("lr")]))
    epochs_list = list(grid_cfg.get("epochs", [dnn_cfg.get("epochs")]))
    batch_list = list(grid_cfg.get("batch_size", [dnn_cfg.get("batch_size")]))

    combos = list(itertools.product(n_list, dropout_list, l2_list, lr_list, epochs_list, batch_list))
    total_combos = len(combos)
    logging.info("Grid search combos to evaluate: %d", total_combos)

    # Prepare data (Close series only for hyperparameter selection)
    target = "Close"
    series_scaled = scaler_mgr.fit_transform(train_df[target], name=target)
    X, y = build_xy(series_scaled, t)
    if X.shape[0] == 0:
        raise RuntimeError("Not enough data for grid search (after windowing).")

    # Ensure we have at least cv_splits folds possible
    if X.shape[0] <= cv_splits:
        raise RuntimeError(f"Not enough samples ({X.shape[0]}) for cv_splits={cv_splits}.")

    cv = TimeSeriesSplit(n_splits=cv_splits)
    seeds = cfg["model"]["seeds"]

    best_score = float("inf")
    best_params: Optional[dict] = None
    evaluations = []

    for idx, (n, dropout, l2, lr, epochs, batch_size) in enumerate(combos, start=1):
        combo_start = time.time()
        fold_scores = []
        logging.info("Evaluating combo %d/%d: n=%s dropout=%s l2=%s lr=%s epochs=%s batch_size=%s",
                     idx, total_combos, n, dropout, l2, lr, epochs, batch_size)
        combo_ok = True
        try:
            for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X)):
                # Deterministic-ish seeds per combo/fold to keep runs reproducible
                set_seeds(
                    python_seed=int(seeds["python"]) + idx + fold_idx,
                    numpy_seed=int(seeds["numpy"]) + idx + fold_idx,
                    tf_seed=int(seeds["tensorflow"]) + idx + fold_idx,
                )

                X_train, y_train = X[train_idx], y[train_idx]
                X_val, y_val = X[test_idx], y[test_idx]

                # Build a fresh regressor for each fold to avoid weight leakage
                reg = DNNRegressor(
                    build_fn=build_model,
                    input_dim=t,
                    n=int(n),
                    t=t,
                    bs=int(dnn_cfg.get("bs")),
                    dropout_rate=float(dropout),
                    lr=float(lr),
                    l2=float(l2),
                    epochs=int(epochs),
                    batch_size=int(batch_size),
                    early_stopping=True,  # use early stopping during CV folds
                    patience=int(dnn_cfg.get("patience")),
                    loss=str(dnn_cfg.get("loss")),
                    optimizer=str(dnn_cfg.get("optimizer")),
                )

                # Fit on fold train, validate on fold test (no leakage)
                try:
                    reg.fit(X_train, y_train, validation_data=(X_val, y_val), verbose=0)
                except Exception as e:
                    logging.exception("Training failed for combo %s on fold %d: %s", (n, dropout, l2, lr, epochs, batch_size), fold_idx, e)
                    combo_ok = False
                    break

                # Predict and score
                try:
                    y_pred = reg.predict(X_val)
                    fold_mae = float(np.mean(np.abs(y_pred - y_val)))
                    fold_scores.append(fold_mae)
                except Exception as e:
                    logging.exception("Prediction failed for combo %s on fold %d: %s", (n, dropout, l2, lr, epochs, batch_size), fold_idx, e)
                    combo_ok = False
                    break

                # Clear session / free memory before next fold
                try:
                    if K is not None:
                        K.clear_session()
                except Exception:
                    pass
                gc.collect()

            if not combo_ok or len(fold_scores) == 0:
                logging.warning("Skipping combo %d due to training/prediction errors.", idx)
                evaluations.append({"combo": (n, dropout, l2, lr, epochs, batch_size), "mean_score": None, "fold_scores": None})
                continue

            mean_score = float(np.mean(fold_scores))
            evaluations.append({"combo": (n, dropout, l2, lr, epochs, batch_size), "mean_score": mean_score, "fold_scores": fold_scores})
            logging.info("Combo %d mean MAE=%.6f (folds=%d)", idx, mean_score, len(fold_scores))

            if mean_score < best_score:
                best_score = mean_score
                best_params = {
                    "n": int(n),
                    "dropout": float(dropout),
                    "l2": float(l2),
                    "lr": float(lr),
                    "epochs": int(epochs),
                    "batch_size": int(batch_size),
                    "cv_splits": int(cv_splits),
                }
                logging.info("New best params (MAE=%.6f): %s", best_score, best_params)

        except Exception as e:
            logging.exception("Unexpected error while evaluating combo %d: %s", idx, e)
        finally:
            combo_time = time.time() - combo_start
            logging.info("Combo %d/%d evaluated in %.1fs", idx, total_combos, combo_time)

    total_time = time.time() - start_time
    result = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "duration_seconds": total_time,
        "evaluations": evaluations,
        "best_score": None if best_score == float("inf") else float(best_score),
        "best_params": best_params,
    }

    # Persist results to disk
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)
    logging.info("Saved grid search results to %s (best_score=%s)", save_path, result["best_score"])
    return result


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    p.add_argument("--all", action="store_true", help="Train all OHLC series (default trains only Close)")
    p.add_argument("--out", default=None, help="Output models directory (overrides config.io.models_dir)")
    p.add_argument("--grid", action="store_true", help="Run time-series grid search (and persist best params)")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    # set seeds
    seeds = cfg["model"]["seeds"]
    set_seeds(python_seed=int(seeds["python"]), numpy_seed=int(seeds["numpy"]), tf_seed=int(seeds["tensorflow"]))

    models_dir = args.out or cfg["io"]["models_dir"]
    os.makedirs(models_dir, exist_ok=True)

    # prepare data
    df = generate_synthetic_ohlc(n_days=120, start=cfg["data"].get("start_date", "2020-01-01"), seed=1234)
    logging.info("Generated synthetic OHLC with %d rows", len(df))

    # Attempt to split via split_by_cutoff if available, else fallback to positional split
    cutoff_idx = 89
    cutoff_ts = df.index[cutoff_idx]
    try:
        if split_by_cutoff is not None:
            train_df, test_df = split_by_cutoff(df, cutoff_ts.isoformat(), horizon=int(cfg["data"]["horizon_days"]))
        else:
            raise RuntimeError("split_by_cutoff unavailable")
    except Exception:
        train_df = df.iloc[: cutoff_idx + 1].copy()
        test_df = df.iloc[cutoff_idx + 1 : cutoff_idx + 1 + int(cfg["data"]["horizon_days"])].copy()
        if len(test_df) != int(cfg["data"]["horizon_days"]):
            raise RuntimeError("Not enough rows for test window in fallback split.")

    logging.info("Train rows: %d, Test rows: %d", len(train_df), len(test_df))

    scaler_mgr = ScalerManager(scaler_type=cfg["model"]["scaler_type"])

    # Grid search (optional)
    grid_search_path = cfg["io"].get("grid_search_path", os.path.join(models_dir, "grid_search.json"))
    best_params = None
    if args.grid:
        grid_result = run_grid_search(train_df, cfg, scaler_mgr, grid_search_path)
        best_params = grid_result.get("best_params")
    else:
        # If a previous grid search exists, load it and apply best params automatically
        if os.path.exists(grid_search_path):
            try:
                with open(grid_search_path, "r", encoding="utf-8") as fh:
                    grid_result = json.load(fh)
                    best_params = grid_result.get("best_params")
                    logging.info("Loaded existing grid search best params: %s", best_params)
            except Exception:
                logging.exception("Failed to load existing grid search file %s", grid_search_path)
                best_params = None
        else:
            logging.info("Grid search not requested and no existing grid_search.json found; training will use defaults")

    targets = OHLC if args.all else ["Close"]
    results = []
    for target in targets:
        try:
            res = train_one_series(train_df, target, cfg, scaler_mgr, models_dir, best_params=best_params)
            logging.info("Trained %s -> %s", target, res["model_path"])
            results.append(res)
        except Exception as exc:
            logging.exception("Training failed for %s: %s", target, exc)

    # Persist scalers
    scaler_path = os.path.join(models_dir, "scalers.joblib")
    scaler_mgr.save(scaler_path)
    logging.info("Saved scalers to %s", scaler_path)

    # Save summary
    summary_path = os.path.join(models_dir, "train_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"results": results, "used_best_params": best_params}, f, indent=2, default=str)
    logging.info("Wrote training summary to %s", summary_path)

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
