"""
Tests for Milestone 9: TimeSeries-aware grid search and CV leakage checks.

- test_timeseries_split_no_leakage: ensures TimeSeriesSplit produces folds where training
  indices precede validation indices (no future leakage).
- test_run_grid_search_minimal: (requires tensorflow) runs the run_grid_search routine
  on a very small grid to verify it returns/persists best_params.
"""
from __future__ import annotations

import importlib.util
import json
import os
import copy

import pytest
from sklearn.model_selection import TimeSeriesSplit

from src.utils.config import load_config
from src.features.window import build_xy
from src.data.scale import ScalerManager


def test_timeseries_split_no_leakage(ohlc_random_walk):
    """
    Sanity-check TimeSeriesSplit on windowed data: for every fold,
    the maximum training index should be strictly less than the minimum test index.
    """
    cfg = load_config("configs/default.yaml")
    t = int(cfg["data"]["lag_t"])
    X, y = build_xy(ohlc_random_walk["Close"], t)
    cv_splits = int(cfg["dnn"].get("cv_splits", 3))

    # Basic precondition
    assert X.shape[0] > cv_splits

    cv = TimeSeriesSplit(n_splits=cv_splits)
    for train_idx, test_idx in cv.split(X):
        assert len(train_idx) > 0 and len(test_idx) > 0
        assert max(train_idx) < min(test_idx), "TimeSeriesSplit produced overlapping/peeking indices"


def _load_train_module():
    """
    Load scripts/train.py as a module via its file path so tests can call run_grid_search().
    This avoids needing 'scripts' to be a package.
    """
    script_path = os.path.join(os.getcwd(), "scripts", "train.py")
    spec = importlib.util.spec_from_file_location("train_mod", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module


def test_run_grid_search_minimal(ohlc_random_walk, tmp_path):
    """
    Quick end-to-end run of run_grid_search using a very small hyperparameter grid.
    Requires TensorFlow; will be skipped otherwise.

    The test asserts that:
    - run_grid_search completes without error on the small grid
    - the returned/resulting JSON contains a non-null 'best_params' entry
    - the saved grid JSON file exists
    """
    pytest.importorskip("tensorflow")

    # Load base config and make a deep copy to modify safely for a tiny grid
    base_cfg = load_config("configs/default.yaml")
    cfg = copy.deepcopy(base_cfg)

    # Make the search tiny so the test runs quickly
    cfg["dnn"]["grid"] = {
        "n": [1, 2],
        "dropout": [cfg["model"]["dropout"]],
        "l2": [cfg["dnn"]["l2"]],
        "lr": [cfg["dnn"]["lr"]],
        "epochs": [3],  # keep epochs very small for test speed
        "batch_size": [16],
    }
    cfg["dnn"]["cv_splits"] = 2
    cfg["dnn"]["patience"] = 2

    # Prepare a scaler manager and call run_grid_search from scripts/train.py
    scaler_mgr = ScalerManager(scaler_type=cfg["model"]["scaler_type"])
    train_mod = _load_train_module()
    save_path = str(tmp_path / "grid_search.json")

    result = train_mod.run_grid_search(ohlc_random_walk, cfg, scaler_mgr, save_path)

    assert result is not None
    assert "best_params" in result and result["best_params"] is not None
    assert os.path.exists(save_path)

    # Basic sanity of the persisted file
    with open(save_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    assert data.get("best_params") is not None
    # ensure expected keys present in best_params
    for key in ["n", "dropout", "l2", "lr", "epochs", "batch_size"]:
        assert key in data["best_params"]
