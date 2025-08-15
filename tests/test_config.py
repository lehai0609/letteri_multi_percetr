import pytest

from src.utils.config import load_config, ConfigError


def test_load_config_default():
    cfg = load_config("configs/default.yaml")
    assert isinstance(cfg, dict)
    # Top-level sections required by validator
    for k in ("data", "model", "dnn", "backtest", "io", "logging", "contracts"):
        assert k in cfg
    # parsed operator precedence helper must be present
    assert "_parsed_operator_precedence" in cfg
    parsed = cfg["_parsed_operator_precedence"]
    assert parsed.get("required_tokens_present") is True
