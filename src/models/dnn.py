# DNN MLP builder and light regressor wrapper used for Milestone 7.

# This module provides:
# - set_seeds(...) helper to fix RNG seeds (best-effort across python/numpy/tensorflow)
# - build_model(...) that returns a compiled `tf.keras.Model` MLP matching the
#   architecture described in the planning docs:
#     Input(t) -> Dense(n*t, ReLU, l2) -> Dropout -> Dense(n*bs, ReLU, l2) ->
#     Dropout -> Dense(1, linear)
# - DNNRegressor: a small scikit-like wrapper with fit/predict/save/load that
#   works even when SciKeras is not available. Tests may import and importorskip
#   tensorflow when running.

# Notes:
# - TensorFlow is optional at import time; functions that require TF will raise a
#   clear ImportError if TF is missing. Tests should use pytest.importorskip("tensorflow")
#   or the project requirements should include TensorFlow for model tests.
# - The wrapper uses Keras callbacks for early stopping when requested.

from __future__ import annotations

import logging
import os
import random
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

try:
    import tensorflow as tf
    from tensorflow.keras import backend as K
    from tensorflow.keras import callbacks as keras_callbacks
    from tensorflow.keras import layers, models, optimizers, regularizers
except Exception:  # pragma: no cover - exercised only in environments lacking TF
    tf = None  # type: ignore

logger = logging.getLogger(__name__)


def _require_tf() -> None:
    if tf is None:
        raise ImportError(
            "TensorFlow is required for src.models.dnn but is not installed. "
            "Install tensorflow (e.g. `pip install tensorflow`) to use the DNN modules."
        )


def set_seeds(python_seed: int = 42, numpy_seed: int = 42, tf_seed: Optional[int] = None) -> None:
    """
    Best-effort determinism:
    - sets python random seed, numpy seed, and tf random seed (if TF available).
    - note: full determinism in TF may require environment variables (TF_DETERMINISTIC_OPS=1)
      and matching hardware/software; this helper aims for reproducibility in tests.
    """
    random.seed(python_seed)
    os.environ["PYTHONHASHSEED"] = str(python_seed)
    np.random.seed(numpy_seed)
    if tf is not None:
        if tf_seed is None:
            tf_seed = numpy_seed
        try:
            tf.random.set_seed(int(tf_seed))
            # try to enable deterministic ops if available (best-effort)
            os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
        except Exception:
            logger.debug("Could not set tf random seed deterministically.")


def build_model(
    input_dim: int,
    n: int,
    t: int,
    bs: int,
    dropout_rate: float = 0.2,
    lr: float = 1e-3,
    l2: float = 1e-4,
    loss: str = "mae",
    optimizer: str = "adam",
) -> "tf.keras.Model":
    """
    Build and compile a Keras MLP according to the TDD.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input (should equal lag t).
    n : int
        Size multiplier; first hidden layer size = n * t.
    t : int
        Lag window length (kept for clarity in naming).
    bs : int
        Secondary multiplier; second hidden layer size = n * bs.
    dropout_rate : float
        Dropout rate applied after each dense layer.
    lr : float
        Learning rate for Adam optimizer.
    l2 : float
        L2 weight decay coefficient.
    loss : str
        Loss function name (e.g., "mae", "mse").
    optimizer : str
        Optimizer name; currently supports "adam" else falls back to string.

    Returns
    -------
    model : tf.keras.Model
    """
    _require_tf()
    if input_dim <= 0:
        raise ValueError("input_dim must be > 0")
    if n <= 0 or bs <= 0:
        raise ValueError("n and bs must be positive integers")

    reg = regularizers.l2(float(l2)) if l2 and l2 > 0 else None

    inputs = layers.Input(shape=(input_dim,), name="input")
    x = layers.Dense(n * t, activation="relu", kernel_regularizer=reg, name="dense_1")(inputs)
    if dropout_rate and dropout_rate > 0.0:
        x = layers.Dropout(rate=float(dropout_rate), name="dropout_1")(x)
    x = layers.Dense(n * bs, activation="relu", kernel_regularizer=reg, name="dense_2")(x)
    if dropout_rate and dropout_rate > 0.0:
        x = layers.Dropout(rate=float(dropout_rate), name="dropout_2")(x)
    outputs = layers.Dense(1, activation="linear", name="output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=f"mlp_t{input_dim}_n{n}_bs{bs}")

    # optimizer selection
    if optimizer.lower() == "adam":
        opt = optimizers.Adam(learning_rate=float(lr))
    else:
        # allow passing custom optimizer names; Keras will resolve strings for common ones
        try:
            opt = optimizer  # type: ignore
        except Exception:
            opt = optimizers.Adam(learning_rate=float(lr))

    model.compile(optimizer=opt, loss=loss, metrics=["mse"])
    return model


class DNNRegressor:
    """
    Lightweight scikit-like regressor wrapper around a compiled Keras model.

    Usage:
        reg = DNNRegressor(build_fn=build_model, input_dim=5, n=1, t=5, bs=5, ...)
        reg.fit(X, y, validation_data=(X_val, y_val))
        y_pred = reg.predict(X_new)
        reg.save(path)
        reg2 = DNNRegressor.load(path)
    """

    def __init__(
        self,
        build_fn: Callable[..., "tf.keras.Model"],
        *,
        input_dim: int,
        n: int,
        t: int,
        bs: int,
        dropout_rate: float = 0.2,
        lr: float = 1e-3,
        l2: float = 1e-4,
        epochs: int = 100,
        batch_size: int = 32,
        early_stopping: bool = True,
        patience: int = 10,
        loss: str = "mae",
        optimizer: str = "adam",
    ) -> None:
        _require_tf()
        self.build_fn = build_fn
        self.input_dim = int(input_dim)
        self.n = int(n)
        self.t = int(t)
        self.bs = int(bs)
        self.dropout_rate = float(dropout_rate)
        self.lr = float(lr)
        self.l2 = float(l2)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.early_stopping = bool(early_stopping)
        self.patience = int(patience)
        self.loss = loss
        self.optimizer = optimizer

        self.model: Optional["tf.keras.Model"] = None

    def build(self) -> "tf.keras.Model":
        """Construct the Keras model (and store it)."""
        self.model = self.build_fn(
            input_dim=self.input_dim,
            n=self.n,
            t=self.t,
            bs=self.bs,
            dropout_rate=self.dropout_rate,
            lr=self.lr,
            l2=self.l2,
            loss=self.loss,
            optimizer=self.optimizer,
        )
        return self.model

    def fit(
        self,
        X: "np.ndarray",
        y: "np.ndarray",
        validation_data: Optional[Tuple["np.ndarray", "np.ndarray"]] = None,
        verbose: int = 0,
    ) -> Dict[str, Any]:
        """
        Fit the model. Returns the Keras history.history dict.

        If the internal model is not yet built, build it first.
        """
        _require_tf()
        if self.model is None:
            self.build()
        cb = []
        if self.early_stopping:
            # monitor val_loss if validation_data provided, otherwise monitor loss
            monitor = "val_loss" if validation_data is not None else "loss"
            cb.append(
                keras_callbacks.EarlyStopping(monitor=monitor, patience=self.patience, restore_best_weights=True)
            )
        history = self.model.fit(
            X,
            y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=validation_data,
            callbacks=cb,
            verbose=verbose,
            shuffle=False,
        )
        return history.history

    def predict(self, X: "np.ndarray", verbose: int = 0) -> "np.ndarray":
        """Return 1-D array of predictions."""
        _require_tf()
        if self.model is None:
            raise RuntimeError("Model has not been built/loaded. Call fit() or load() first.")
        preds = self.model.predict(X, batch_size=self.batch_size, verbose=verbose)
        return np.asarray(preds).reshape(-1)

    def save(self, path: str) -> None:
        """
        Save the Keras model to the given path.

        Behavior:
        - If `path` has a recognized file extension (.keras, .h5, .hdf5), delegate to
          Keras `model.save(path)`.
        - Otherwise treat `path` as a directory: create it (if needed) and save the model
          to `<path>/model.keras` (native Keras archive). This matches the common
          expectation of passing a directory when running training scripts.
        """
        _require_tf()
        if self.model is None:
            raise RuntimeError("No model to save.")

        p_str = str(path)
        lower = p_str.lower()

        # If the user passed a filename with supported extension, save directly there.
        if lower.endswith(".keras") or lower.endswith(".h5") or lower.endswith(".hdf5"):
            parent = os.path.dirname(p_str) or "."
            os.makedirs(parent, exist_ok=True)
            self.model.save(p_str)
            logger.info("Saved model to %s", p_str)
            return

        # Otherwise treat path as a directory and save the model inside it as model.keras
        os.makedirs(p_str, exist_ok=True)
        target = os.path.join(p_str, "model.keras")
        self.model.save(target)
        logger.info("Saved model to %s (inside dir %s)", target, p_str)

    @classmethod
    def load(cls, path: str) -> "DNNRegressor":
        """
        Load a saved Keras model and wrap it in a DNNRegressor instance.

        The loader will attempt multiple candidate locations/formats:
        - path as provided
        - path + ".keras"
        - path + ".h5"
        - if path is a directory: path/"model.keras", path/"model.h5"
        """
        _require_tf()

        candidates = [str(path), f"{str(path)}.keras", f"{str(path)}.h5", f"{str(path)}.hdf5"]

        # If path is a directory, prefer files inside it
        try:
            if os.path.isdir(path):
                candidates.insert(0, os.path.join(str(path), "model.keras"))
                candidates.insert(1, os.path.join(str(path), "model.h5"))
        except Exception:
            # ignore any os errors and proceed with candidate list
            pass

        last_exc = None
        model = None
        for candidate in candidates:
            try:
                model = models.load_model(candidate)
                logger.info("Loaded Keras model from %s", candidate)
                break
            except Exception as exc:
                last_exc = exc
                continue

        if model is None:
            raise RuntimeError(f"Failed to load Keras model from any candidate path. Last error: {last_exc}")

        # Try to infer input_dim from model
        try:
            input_shape = model.input_shape
            input_dim = int(input_shape[1])
        except Exception:
            input_dim = 0
        wrapper = cls(build_fn=lambda **kw: model, input_dim=input_dim, n=1, t=input_dim, bs=1)
        wrapper.model = model
        return wrapper


__all__ = ["set_seeds", "build_model", "DNNRegressor"]
