"""
Config utilities: load and validate YAML configuration for the Letteri replication.

Milestone 0 scope:
- Load configs/default.yaml (or a provided path)
- Validate required sections/keys and basic types
- Sanity-check the TEMA operator precedence expression
"""
from __future__ import annotations

import io
import os
from typing import Any, Dict, List

try:
    import yaml  # type: ignore
except ImportError:
    # Deferred import error; tests and runtime will surface this if environment is missing PyYAML.
    yaml = None  # type: ignore


class ConfigError(Exception):
    """Raised when configuration is invalid."""


ALLOWED_TOP_LEVEL_KEYS = {"data", "model", "dnn", "backtest", "io", "logging", "contracts"}


def _require_module(module, name: str) -> None:
    if module is None:
        raise ConfigError(
            f"Missing dependency: {name}. Please add it to requirements and install the environment."
        )


def load_yaml(path: str) -> Dict[str, Any]:
    """Load YAML from file path."""
    _require_module(yaml, "PyYAML")
    if not os.path.exists(path):
        raise ConfigError(f"Config file not found: {path}")
    with io.open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ConfigError("Top-level YAML must be a mapping/object.")
    return data


def _expect_key(d: Dict[str, Any], key: str) -> Any:
    if key not in d:
        raise ConfigError(f"Missing required key: {key}")
    return d[key]


def _expect_type(val: Any, type_, path: str) -> None:
    if not isinstance(val, type_):
        raise ConfigError(f"Invalid type for {path}: expected {type_.__name__}, got {type(val).__name__}")


def _expect_iterable_of(val: Any, elem_type, path: str) -> None:
    if not isinstance(val, (list, tuple)):
        raise ConfigError(f"Invalid type for {path}: expected list/tuple, got {type(val).__name__}")
    for i, x in enumerate(val):
        if not isinstance(x, elem_type):
            raise ConfigError(f"Invalid element type for {path}[{i}]: expected {elem_type.__name__}, got {type(x).__name__}")


def parse_operator_precedence(expr: str) -> Dict[str, Any]:
    """
    Sanity-check the operator precedence expression string for Triple EMA strategy.

    Requirements:
    - Must reference Open, High, Low, Close and corresponding TEMA_* tokens.
    - Balanced parentheses (basic check).
    """
    required_tokens = ["Low", "High", "Close", "Open", "TEMA_Low", "TEMA_High", "TEMA_Close", "TEMA_Open"]
    missing: List[str] = [tok for tok in required_tokens if tok not in expr]
    if missing:
        raise ConfigError(f"operator_precedence missing tokens: {missing}")

    # Basic parentheses balance check
    bal = 0
    for ch in expr:
        if ch == "(":
            bal += 1
        elif ch == ")":
            bal -= 1
        if bal < 0:
            raise ConfigError("operator_precedence has mismatched parentheses (early close).")
    if bal != 0:
        raise ConfigError("operator_precedence has unbalanced parentheses.")

    return {"raw": expr, "required_tokens_present": True}


def validate_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    # Top-level keys
    for k in ALLOWED_TOP_LEVEL_KEYS:
        if k not in cfg:
            raise ConfigError(f"Missing top-level section: {k}")

    # data
    data = _expect_key(cfg, "data")
    _expect_type(data, dict, "data")
    for k in ["source", "ticker", "start_date", "cutoff_date", "end_date"]:
        _expect_type(_expect_key(data, k), str, f"data.{k}")
    _expect_type(_expect_key(data, "horizon_days"), int, "data.horizon_days")
    _expect_type(_expect_key(data, "lag_t"), int, "data.lag_t")
    cache = _expect_key(data, "cache")
    _expect_type(cache, dict, "data.cache")
    for k in ["raw_dir", "processed_dir"]:
        _expect_type(_expect_key(cache, k), str, f"data.cache.{k}")

    # model
    model = _expect_key(cfg, "model")
    _expect_type(model, dict, "model")
    _expect_type(_expect_key(model, "tema_period"), int, "model.tema_period")
    op = _expect_key(model, "operator_precedence")
    _expect_type(op, str, "model.operator_precedence")
    # Will raise if invalid
    cfg["_parsed_operator_precedence"] = parse_operator_precedence(op)
    scaler_type = _expect_key(model, "scaler_type")
    _expect_type(scaler_type, str, "model.scaler_type")
    seeds = _expect_key(model, "seeds")
    _expect_type(seeds, dict, "model.seeds")
    for k in ["python", "numpy", "tensorflow"]:
        _expect_type(_expect_key(seeds, k), int, f"model.seeds.{k}")
    for k in ["dropout", "dropout_alt"]:
        _expect_type(_expect_key(model, k), (int, float), f"model.{k}")

    # dnn
    dnn = _expect_key(cfg, "dnn")
    _expect_type(dnn, dict, "dnn")
    _expect_iterable_of(_expect_key(dnn, "n_grid"), int, "dnn.n_grid")
    for k, typ in [
        ("bs", int),
        ("l2", (int, float)),
        ("lr", (int, float)),
        ("epochs", int),
        ("batch_size", int),
        ("cv_splits", int),
        ("early_stopping", bool),
        ("patience", int),
        ("input_dim_from_t", bool),
        ("loss", str),
        ("optimizer", str),
    ]:
        _expect_type(_expect_key(dnn, k), typ, f"dnn.{k}")

    # backtest
    backtest = _expect_key(cfg, "backtest")
    _expect_type(backtest, dict, "backtest")
    for k, typ in [
        ("initial_cash", (int, float)),
        ("execution_price", str),
        ("cost_bps", (int, float)),
        ("slippage_bps", (int, float)),
        ("allow_fractional", bool),
        ("mark_to_market", str),
    ]:
        _expect_type(_expect_key(backtest, k), typ, f"backtest.{k}")

    # io
    io_cfg = _expect_key(cfg, "io")
    _expect_type(io_cfg, dict, "io")
    for k in ["models_dir", "forecasts_dir", "reports_dir", "logs_dir", "grid_search_path"]:
        _expect_type(_expect_key(io_cfg, k), str, f"io.{k}")

    # logging
    log_cfg = _expect_key(cfg, "logging")
    _expect_type(log_cfg, dict, "logging")
    for k, typ in [("level", str), ("fmt", str), ("datefmt", str)]:
        _expect_type(_expect_key(log_cfg, k), typ, f"logging.{k}")

    # contracts
    contracts = _expect_key(cfg, "contracts")
    _expect_type(contracts, dict, "contracts")
    data_schema = _expect_key(contracts, "data_schema")
    _expect_type(data_schema, dict, "contracts.data_schema")
    _expect_type(_expect_key(data_schema, "index"), str, "contracts.data_schema.index")
    _expect_iterable_of(_expect_key(data_schema, "columns"), str, "contracts.data_schema.columns")

    pred_schema = _expect_key(contracts, "predicted_schema")
    _expect_type(pred_schema, dict, "contracts.predicted_schema")
    _expect_type(_expect_key(pred_schema, "index"), str, "contracts.predicted_schema.index")
    _expect_iterable_of(_expect_key(pred_schema, "columns"), str, "contracts.predicted_schema.columns")

    sig_schema = _expect_key(contracts, "signals_schema")
    _expect_type(sig_schema, dict, "contracts.signals_schema")
    _expect_type(_expect_key(sig_schema, "index"), str, "contracts.signals_schema.index")
    _expect_iterable_of(_expect_key(sig_schema, "columns"), str, "contracts.signals_schema.columns")

    return cfg


def load_config(path: str = "configs/default.yaml") -> Dict[str, Any]:
    """
    Public API: Load and validate configuration.
    Returns the validated config dictionary with an added _parsed_operator_precedence helper.
    """
    cfg = load_yaml(path)
    cfg = validate_config(cfg)
    return cfg
