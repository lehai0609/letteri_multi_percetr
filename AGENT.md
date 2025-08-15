# Agent Instructions for Letteri Multi-Perceptron Trading System

## Commands
- **Test**: `python -m pytest` (all tests), `python -m pytest tests/test_models_dnn.py` (single file)
- **Lint**: `python -m flake8 src tests` (style), `python -m black src tests` (format), `python -m isort src tests` (imports)
- **Type check**: Use mypy if needed (not configured yet)

## Architecture
- **Python 3.13+** trading system implementing Letteri et al. DNN forecasting paper
- **src/**: Main modules (data/, models/, features/, strategies/, forecasting/, backtest/, utils/, metrics/, indicators/)
- **tests/**: pytest-based tests with `pytest.importorskip` for optional dependencies
- **configs/**: YAML configuration (default.yaml) with data contracts and hyperparameters
- **TensorFlow/Keras**: Optional dependency for DNN models (graceful ImportError handling)

## Code Style
- **Black** formatting (line length 88), **isort** imports (profile="black")
- **Type hints**: Use `from __future__ import annotations` and typing module
- **Imports**: Optional deps handled with try/except blocks, use `pytest.importorskip` in tests
- **Docstrings**: NumPy style with Parameters/Returns sections
- **Error handling**: Custom exceptions inherit from Exception, descriptive error messages
- **Naming**: snake_case for functions/vars, PascalCase for classes, descriptive variable names
- **Constants**: ALL_CAPS for module-level constants
- **Logging**: Use `logging.getLogger(__name__)`, INFO level default
