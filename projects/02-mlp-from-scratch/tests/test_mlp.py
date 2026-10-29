"""Black-box tests for the multi-layer perceptron.

Test the behavior — MLP solves XOR (perceptron cannot), separates multiclass
Gaussian blobs, gradients are correct — not internal weight values. Refactoring
the training loop, activation, or gradient math should not break these tests.

The gradient-correctness test is the load-bearing one: it verifies analytical
backprop against finite-difference numerical gradients. If backprop is wrong,
training may still "look right" for a while but the model won't generalize.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_PROJECT_DIR = Path(__file__).resolve().parents[1]
_MODULE_PATH = _PROJECT_DIR / "mlp.py"
_spec = importlib.util.spec_from_file_location("mlp_mod", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
mlp_mod = importlib.util.module_from_spec(_spec)
sys.modules["mlp_mod"] = mlp_mod
_spec.loader.exec_module(mlp_mod)

MLP = mlp_mod.MLP
make_xor = mlp_mod.make_xor
make_gaussian_blobs = mlp_mod.make_gaussian_blobs


class TestMLPSolvesXOR:
    """The perceptron's failure case. This is the point of adding a hidden layer."""

    def test_solves_xor_with_hidden_layer(self) -> None:
        X, y = make_xor()
        model = MLP(layer_sizes=[2, 8, 2], learning_rate=0.5, max_epochs=5000, seed=0)
        model.fit(X, y)
        assert model.score(X, y) == 1.0, "MLP with hidden layer must solve XOR"

    def test_deeper_network_also_solves_xor(self) -> None:
        X, y = make_xor()
        model = MLP(layer_sizes=[2, 8, 8, 2], learning_rate=0.5, max_epochs=5000, seed=0)
        model.fit(X, y)
        assert model.score(X, y) == 1.0


class TestMLPMulticlass:
    """Softmax + cross-entropy for K > 2 classes."""

    def test_classifies_three_gaussian_blobs(self) -> None:
        X_train, y_train = make_gaussian_blobs(n_per_class=100, n_classes=3, seed=1)
        X_test, y_test = make_gaussian_blobs(n_per_class=100, n_classes=3, seed=2)
        model = MLP(layer_sizes=[2, 16, 3], learning_rate=0.1, max_epochs=2000, seed=0)
        model.fit(X_train, y_train)
        assert model.score(X_test, y_test) > 0.90, "must generalize to a fresh sample"

    def test_classifies_five_gaussian_blobs(self) -> None:
        X_train, y_train = make_gaussian_blobs(n_per_class=80, n_classes=5, seed=1)
        X_test, y_test = make_gaussian_blobs(n_per_class=80, n_classes=5, seed=2)
        model = MLP(layer_sizes=[2, 32, 5], learning_rate=0.1, max_epochs=3000, seed=0)
        model.fit(X_train, y_train)
        assert model.score(X_test, y_test) > 0.85


class TestPredictionContract:
    """Output shape, label domain, and probability validity."""

    def test_predict_returns_class_indices(self) -> None:
        X, y = make_gaussian_blobs(n_per_class=50, n_classes=3, seed=0)
        model = MLP(layer_sizes=[2, 8, 3], learning_rate=0.1, max_epochs=500, seed=0)
        model.fit(X, y)
        predictions = model.predict(X)
        assert predictions.shape == (len(y),)
        assert predictions.dtype.kind in ("i", "u")
        assert set(np.unique(predictions).tolist()) <= {0, 1, 2}

    def test_predict_proba_returns_valid_distribution(self) -> None:
        X, y = make_gaussian_blobs(n_per_class=50, n_classes=3, seed=0)
        model = MLP(layer_sizes=[2, 8, 3], learning_rate=0.1, max_epochs=500, seed=0)
        model.fit(X, y)
        probs = model.predict_proba(X)
        assert probs.shape == (len(y), 3)
        assert np.all(probs >= 0), "probabilities cannot be negative"
        assert np.all(probs <= 1), "probabilities cannot exceed 1"
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-6)


class TestGradientCorrectness:
    """Load-bearing: analytical backprop must match numerical gradients.

    If this test fails, backprop is wrong. Every downstream training result is
    coincidence. This is the single most important test in the project.
    """

    def test_analytical_gradient_matches_numerical(self) -> None:
        X, y = make_gaussian_blobs(n_per_class=10, n_classes=3, seed=0)
        model = MLP(layer_sizes=[2, 4, 3], learning_rate=0.01, max_epochs=1, seed=0)

        analytical_grads = model._compute_gradients(X, y)  # type: ignore[attr-defined]

        eps = 1e-5
        for layer_idx, (dW, db) in enumerate(analytical_grads):
            W = model.weights[layer_idx]  # type: ignore[attr-defined]
            b = model.biases[layer_idx]  # type: ignore[attr-defined]

            for i in range(min(W.shape[0], 3)):
                for j in range(min(W.shape[1], 3)):
                    original = W[i, j]
                    W[i, j] = original + eps
                    loss_plus = model._loss(X, y)  # type: ignore[attr-defined]
                    W[i, j] = original - eps
                    loss_minus = model._loss(X, y)  # type: ignore[attr-defined]
                    W[i, j] = original

                    numerical = (loss_plus - loss_minus) / (2 * eps)
                    np.testing.assert_allclose(
                        dW[i, j],
                        numerical,
                        rtol=1e-3,
                        atol=1e-5,
                        err_msg=f"grad mismatch at layer {layer_idx} W[{i},{j}]",
                    )

            for i in range(min(b.shape[0], 3)):
                original = b[i]
                b[i] = original + eps
                loss_plus = model._loss(X, y)  # type: ignore[attr-defined]
                b[i] = original - eps
                loss_minus = model._loss(X, y)  # type: ignore[attr-defined]
                b[i] = original

                numerical = (loss_plus - loss_minus) / (2 * eps)
                np.testing.assert_allclose(
                    db[i],
                    numerical,
                    rtol=1e-3,
                    atol=1e-5,
                    err_msg=f"grad mismatch at layer {layer_idx} b[{i}]",
                )


class TestPreconditions:
    """Assertions fire before any work happens."""

    def test_rejects_fewer_than_two_layers(self) -> None:
        with pytest.raises(AssertionError, match="at least 2 layers"):
            MLP(layer_sizes=[5])

    def test_rejects_non_positive_layer_size(self) -> None:
        with pytest.raises(AssertionError, match="positive"):
            MLP(layer_sizes=[2, 0, 3])

    def test_rejects_non_positive_learning_rate(self) -> None:
        with pytest.raises(AssertionError, match="learning_rate must be positive"):
            MLP(layer_sizes=[2, 4, 2], learning_rate=0.0)

    def test_rejects_wrong_input_dimensionality(self) -> None:
        model = MLP(layer_sizes=[2, 4, 3], learning_rate=0.1, max_epochs=10)
        X = np.zeros((5, 4))
        with pytest.raises(AssertionError, match="expected 2 features"):
            model.predict(X)

    def test_rejects_shape_mismatch_between_data_and_labels(self) -> None:
        model = MLP(layer_sizes=[2, 4, 3], learning_rate=0.1, max_epochs=10)
        X = np.zeros((5, 2))
        y = np.zeros(3, dtype=np.int64)
        with pytest.raises(AssertionError, match=r"rows.*labels"):
            model.fit(X, y)

    def test_rejects_labels_outside_valid_range(self) -> None:
        model = MLP(layer_sizes=[2, 4, 3], learning_rate=0.1, max_epochs=10)
        X = np.zeros((3, 2))
        y = np.array([0, 1, 99], dtype=np.int64)
        with pytest.raises(AssertionError, match="labels must be in"):
            model.fit(X, y)


class TestDatasets:
    """Sanity checks on the toy datasets."""

    def test_xor_dataset(self) -> None:
        X, y = make_xor()
        assert X.shape == (4, 2)
        assert y.shape == (4,)
        assert set(np.unique(y).tolist()) == {0, 1}

    def test_gaussian_blobs_shape(self) -> None:
        X, y = make_gaussian_blobs(n_per_class=25, n_classes=4)
        assert X.shape == (100, 2)
        assert y.shape == (100,)
        assert set(np.unique(y).tolist()) == {0, 1, 2, 3}
