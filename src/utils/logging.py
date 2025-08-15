"""
Project logging utilities.

Important:
- This file is named logging.py inside src/utils/. When importing, avoid shadowing
  the standard library 'logging' module in your files. Prefer:
    from src.utils import logging as log_utils
    logger = log_utils.setup_logging(...)
- Inside this module we import the stdlib logging as 'py_logging' to avoid confusion.

Features:
- setup_logging(): human or JSON logs, console + timed-rotating file, UTF-8.
- setup_logging_from_config(): configure from a dict (e.g., loaded YAML).
- set_log_context()/clear_log_context(): inject consistent context (run_id, seed, ticker, ...).
- silence_third_party(): reduce noise from chatty libs; TensorFlow verbosity control.
- install_excepthook(): route uncaught exceptions to your logger.

Example:
    from src.utils import logging as log_utils
    logger = log_utils.setup_logging(
        name="app",
        level="INFO",
        log_dir="logs",
        console=True,
        file=True,
        json=False,
        rotate_when="D",
        backup_count=7,
    )
    log_utils.set_log_context(run_id="2025-08-15T12-00-00Z", seed=42, ticker="ANF")
    logger.info("Pipeline start")
"""

from __future__ import annotations

import json
import logging as py_logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler, RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

# ---------------------------------------------------------------------------
# Constants and helpers
# ---------------------------------------------------------------------------

LOG_LEVELS: Dict[str, int] = {
    "CRITICAL": py_logging.CRITICAL,
    "ERROR": py_logging.ERROR,
    "WARNING": py_logging.WARNING,
    "WARN": py_logging.WARNING,
    "INFO": py_logging.INFO,
    "DEBUG": py_logging.DEBUG,
    "NOTSET": py_logging.NOTSET,
}

DEFAULT_HUMAN_FORMAT = (
    "%(asctime)s.%(msecs)03d | %(levelname)s | %(name)s:%(lineno)d | %(ctx)s | %(message)s"
)
DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"

# Global context filter instance shared across handlers
class ContextFilter(py_logging.Filter):
    """
    Adds contextual key/values to each LogRecord as a single 'ctx' string.
    Use set() to update global context; values persist until cleared.
    """

    def __init__(self, initial: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self._context: Dict[str, Any] = dict(initial or {})

    def set(self, **kwargs: Any) -> None:
        self._context.update({k: v for k, v in kwargs.items() if v is not None})

    def clear(self, *keys: str) -> None:
        if not keys:
            self._context.clear()
            return
        for k in keys:
            self._context.pop(k, None)

    def filter(self, record: py_logging.LogRecord) -> bool:
        # Build a compact "k=v" string; stable key order
        if self._context:
            parts = [f"{k}={self._context[k]}" for k in sorted(self._context.keys())]
            setattr(record, "ctx", " ".join(parts))
        else:
            setattr(record, "ctx", "-")
        return True


_context_filter = ContextFilter()


class JsonFormatter(py_logging.Formatter):
    """
    JSON log formatter suitable for structured logs.

    Fields:
      time, level, logger, line, msg, ctx (dict), process, thread, file, func
      plus exception info if present.
    """

    def format(self, record: py_logging.LogRecord) -> str:
        base = {
            "time": datetime.fromtimestamp(record.created).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "line": record.lineno,
            "msg": record.getMessage(),
            "ctx": getattr(record, "ctx", None),
            "process": record.process,
            "thread": record.thread,
            "file": record.pathname,
            "func": record.funcName,
        }
        if record.exc_info:
            base["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(base, ensure_ascii=False)


def _ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _coerce_level(level: Union[str, int]) -> int:
    if isinstance(level, int):
        return level
    return LOG_LEVELS.get(str(level).upper(), py_logging.INFO)


def _resolve_logfile_name(name: str, log_dir: Union[str, Path], filename: Optional[str]) -> Path:
    log_dir_path = _ensure_dir(log_dir)
    if filename:
        return log_dir_path / filename
    ts = datetime.now().strftime("%Y-%m-%d")
    return log_dir_path / f"{name}_{ts}.log"


@dataclass
class LoggingConfig:
    name: str = "app"
    level: Union[str, int] = "INFO"
    log_dir: Union[str, Path] = "logs"
    filename: Optional[str] = None
    console: bool = True
    file: bool = True
    json: bool = False
    rotate_when: Optional[str] = "D"  # 'S','M','H','D','W0'-'W6','midnight' or None
    backup_count: int = 7
    max_bytes: Optional[int] = None  # If set, use RotatingFileHandler instead of Timed
    reset: bool = True  # Remove existing handlers on this logger
    propagate: bool = False  # Prevent double logging if using root
    silence_libs: bool = True
    silence_level: Union[str, int] = "WARNING"
    tf_cpp_min_log_level: Optional[int] = 2  # 0..3; 2 hides INFO, shows WARNING+
    capture_warnings: bool = True  # route warnings.warn to logging


def setup_logging(
    name: str = "app",
    level: Union[str, int] = "INFO",
    log_dir: Union[str, Path] = "logs",
    filename: Optional[str] = None,
    *,
    console: bool = True,
    file: bool = True,
    json: bool = False,
    rotate_when: Optional[str] = "D",
    backup_count: int = 7,
    max_bytes: Optional[int] = None,
    reset: bool = True,
    propagate: bool = False,
    silence_libs: bool = True,
    silence_level: Union[str, int] = "WARNING",
    tf_cpp_min_log_level: Optional[int] = 2,
    capture_warnings: bool = True,
) -> py_logging.Logger:
    """
    Configure and return a logger.

    Parameters mirror LoggingConfig; see dataclass docs above.
    """
    cfg = LoggingConfig(
        name=name,
        level=level,
        log_dir=log_dir,
        filename=filename,
        console=console,
        file=file,
        json=json,
        rotate_when=rotate_when,
        backup_count=backup_count,
        max_bytes=max_bytes,
        reset=reset,
        propagate=propagate,
        silence_libs=silence_libs,
        silence_level=silence_level,
        tf_cpp_min_log_level=tf_cpp_min_log_level,
        capture_warnings=capture_warnings,
    )
    return _apply_logging_config(cfg)


def setup_logging_from_config(config: Dict[str, Any]) -> py_logging.Logger:
    """
    Configure logging from a config dict (e.g., loaded from YAML).

    Expected shape (all keys optional):
      logging:
        name: "app"
        level: "INFO"
        dir: "logs"
        filename: null
        console: true
        file: true
        json: false
        rotate_when: "D"
        backup_count: 7
        max_bytes: null
        reset: true
        propagate: false
        silence_libs: true
        silence_level: "WARNING"
        tf_cpp_min_log_level: 2
        capture_warnings: true
    """
    section = dict(config.get("logging", {}))
    cfg = LoggingConfig(
        name=section.get("name", "app"),
        level=section.get("level", "INFO"),
        log_dir=section.get("dir", "logs"),
        filename=section.get("filename"),
        console=section.get("console", True),
        file=section.get("file", True),
        json=section.get("json", False),
        rotate_when=section.get("rotate_when", "D"),
        backup_count=int(section.get("backup_count", 7)),
        max_bytes=section.get("max_bytes"),
        reset=section.get("reset", True),
        propagate=section.get("propagate", False),
        silence_libs=section.get("silence_libs", True),
        silence_level=section.get("silence_level", "WARNING"),
        tf_cpp_min_log_level=section.get("tf_cpp_min_log_level", 2),
        capture_warnings=section.get("capture_warnings", True),
    )
    return _apply_logging_config(cfg)


def _apply_logging_config(cfg: LoggingConfig) -> py_logging.Logger:
    # TensorFlow C++ logs: 0=all, 1=filter INFO, 2=WARNING, 3=ERROR
    if cfg.tf_cpp_min_log_level is not None:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(cfg.tf_cpp_min_log_level)

    if cfg.capture_warnings:
        py_logging.captureWarnings(True)

    logger = py_logging.getLogger(cfg.name)
    logger.setLevel(_coerce_level(cfg.level))
    logger.propagate = cfg.propagate

    if cfg.reset:
        for h in list(logger.handlers):
            logger.removeHandler(h)

    # Always attach the context filter to the logger (once)
    if all(not isinstance(f, ContextFilter) for f in logger.filters):
        logger.addFilter(_context_filter)

    # Console handler
    if cfg.console:
        ch = py_logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(_coerce_level(cfg.level))
        ch.addFilter(_context_filter)
        if cfg.json:
            ch.setFormatter(JsonFormatter())
        else:
            ch.setFormatter(py_logging.Formatter(DEFAULT_HUMAN_FORMAT, datefmt=DEFAULT_DATEFMT))
        logger.addHandler(ch)

    # File handler
    if cfg.file:
        logfile = _resolve_logfile_name(cfg.name, cfg.log_dir, cfg.filename)
        logfile.parent.mkdir(parents=True, exist_ok=True)
        if cfg.max_bytes:
            fh = RotatingFileHandler(
                logfile, maxBytes=int(cfg.max_bytes), backupCount=int(cfg.backup_count), encoding="utf-8"
            )
        else:
            when = cfg.rotate_when or "D"
            fh = TimedRotatingFileHandler(
                logfile, when=when, backupCount=int(cfg.backup_count), encoding="utf-8", utc=False
            )
        fh.setLevel(_coerce_level(cfg.level))
        fh.addFilter(_context_filter)
        if cfg.json:
            fh.setFormatter(JsonFormatter())
        else:
            fh.setFormatter(py_logging.Formatter(DEFAULT_HUMAN_FORMAT, datefmt=DEFAULT_DATEFMT))
        logger.addHandler(fh)

    if cfg.silence_libs:
        silence_third_party(level=cfg.silence_level)

    return logger


def get_logger(name: Optional[str] = None) -> py_logging.Logger:
    """
    Get a configured logger by name. If name is None, returns the root logger.
    """
    return py_logging.getLogger(name)


def set_log_context(**kwargs: Any) -> None:
    """
    Set or update contextual fields for logs, e.g.:
      set_log_context(run_id="2025-08-15T12-00", seed=42, ticker="ANF")
    """
    _context_filter.set(**kwargs)


def clear_log_context(*keys: str) -> None:
    """
    Clear contextual fields. With no args, clears all.
    """
    _context_filter.clear(*keys)


def silence_third_party(level: Union[str, int] = "WARNING") -> None:
    """
    Reduce noise from common libraries.
    """
    lvl = _coerce_level(level)
    noisy = [
        "matplotlib",
        "tensorflow",
        "absl",
        "urllib3",
        "pmdarima",
        "PIL",
        "numexpr",
        "yfinance",
        "fsspec",
        "asyncio",
    ]
    for name in noisy:
        py_logging.getLogger(name).setLevel(lvl)


def install_excepthook(logger: Optional[py_logging.Logger] = None) -> None:
    """
    Route uncaught exceptions to the provided logger (or root if None).
    """
    target = logger or py_logging.getLogger()

    def _hook(exc_type, exc, tb):
        target.exception("Uncaught exception", exc_info=(exc_type, exc, tb))

    sys.excepthook = _hook


# Convenience: if this module is run directly, demonstrate configuration
if __name__ == "__main__":
    lg = setup_logging(name="demo", level="DEBUG", json=False)
    set_log_context(run_id="demo-run", seed=123)
    lg.debug("Debug message with context")
    lg.info("Info message")
    try:
        1 / 0
    except ZeroDivisionError:
        lg.exception("An error occurred")