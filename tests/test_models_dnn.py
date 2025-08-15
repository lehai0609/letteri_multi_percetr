import numpy as np
import pytest

# Skip these tests if TensorFlow is not available in the environment.
tf = pytest.importorskip("tensorflow")

from src.models.dnn import set_seeds, build_model, DNNRegressor


def test_build_model_layer_sizes():
    """Model layers should match the sizing rules: dense_1 -> n * t, dense_2 -> n * bs, output -> 1."""
    input_dim = 5
    n = 2
    t = 5
    bs = 3
    model = build_model(input_dim=input_dim, n=n, t=t, bs=bs, dropout_rate=0.1, lr=1e-3, l2=1e-4)
    # layer units are accessible as `units` on Dense layers
    dense1 = model.get_layer("dense_1")
    dense2 = model.get_layer("dense_2")
    output = model.get_layer("output")
    assert getattr(dense1, "units", None) == n * t
    assert getattr(dense2, "units", None) == n * bs
    assert getattr(output, "units", None) == 1
    # dropout layers present
    names = [ly.name for ly in model.layers]
    assert "dropout_1" in names
    assert "dropout_2" in names


def test_set_seeds_reproducibility_of_initial_weights():
    """
    Setting identical seeds before building models should produce identical initial weights.
    This is a best-effort test — full TF determinism can depend on environment — but
    it validates that the helper at least sets the RNG seeds.
    """
    set_seeds(python_seed=1, numpy_seed=1, tf_seed=1)
    m1 = build_model(input_dim=5, n=1, t=5, bs=5, dropout_rate=0.0)
    w1 = [w.copy() for w in m1.get_weights()]

    set_seeds(python_seed=1, numpy_seed=1, tf_seed=1)
    m2 = build_model(input_dim=5, n=1, t=5, bs=5, dropout_rate=0.0)
    w2 = [w.copy() for w in m2.get_weights()]

    assert len(w1) == len(w2)
    for a, b in zip(w1, w2):
        assert a.shape == b.shape
        assert np.allclose(a, b)


def test_dnnregressor_fit_predict_save_load(tmp_path):
    """
    Train a tiny network on synthetic data, save it, reload it, and verify predict shape
    and parity between original and loaded model predictions.
    """
    rng = np.random.RandomState(0)
    X = rng.normal(size=(40, 5)).astype(float)
    # synthetic target: linear combination + noise
    coeffs = np.array([0.5, -0.3, 0.2, 0.0, 0.1])
    y = X.dot(coeffs) + 0.01 * rng.normal(size=(X.shape[0],))

    set_seeds(python_seed=0, numpy_seed=0, tf_seed=0)
    reg = DNNRegressor(
        build_fn=build_model,
        input_dim=5,
        n=1,
        t=5,
        bs=5,
        dropout_rate=0.0,
        lr=1e-3,
        l2=1e-6,
        epochs=5,
        batch_size=8,
        early_stopping=False,
        patience=2,
        loss="mae",
        optimizer="adam",
    )

    # Fit (small number of epochs to keep test fast)
    history = reg.fit(X, y, validation_data=None, verbose=0)
    preds = reg.predict(X)
    assert preds.shape == (X.shape[0],)

    model_dir = tmp_path / "dnn_model"
    model_dir_str = str(model_dir)
    reg.save(model_dir_str)

    # Load and predict
    loaded = DNNRegressor.load(model_dir_str)
    preds2 = loaded.predict(X)
    assert preds2.shape == preds.shape
    # Predictions should be extremely close after save/load
    assert np.allclose(preds, preds2, atol=1e-6)
