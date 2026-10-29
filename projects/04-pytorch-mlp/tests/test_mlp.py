"""Black-box tests for the PyTorch MLP.

Same behavioral contract as project 02 (XOR solvable, multiclass classification,
prediction shape, precondition assertions) — the tests are almost identical.
Plus PyTorch-specific tests:

- Parameter count matches the manual math (n_in * n_out + n_out per layer)
- Trained model round-trips through state_dict save/load without accuracy loss

The "same contract, same behavior" property is the whole point of project 04.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

_PROJECT_DIR = Path(__file__).resolve().parents[1]
_MODULE_PATH = _PROJECT_DIR / "mlp.py"
_spec = importlib.util.spec_from_file_location("pt_mlp_mod", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
mlp_mod = importlib.util.module_from_spec(_spec)
sys.modules["pt_mlp_mod"] = mlp_mod
_spec.loader.exec_module(mlp_mod)

MLP = mlp_mod.MLP
make_xor = mlp_mod.make_xor
make_gaussian_blobs = mlp_mod.make_gaussian_blobs


class TestMLPSolvesXOR:
    """Same test that project 02 passes — MLP handles XOR."""

    def test_solves_xor(self) -> None:
        # Arrange
        X, y = make_xor()
        model = MLP(layer_sizes=[2, 8, 2], learning_rate=0.05, max_epochs=2000, seed=0)

        # Act
        model.fit(X, y)

        # Assert
        assert model.score(X, y) == 1.0

    def test_deeper_network_also_solves_xor(self) -> None:
        # Arrange
        X, y = make_xor()
        model = MLP(layer_sizes=[2, 8, 8, 2], learning_rate=0.05, max_epochs=2000, seed=0)

        # Act
        model.fit(X, y)

        # Assert
        assert model.score(X, y) == 1.0


class TestMLPMulticlass:
    def test_classifies_three_gaussian_blobs(self) -> None:
        # Arrange
        X_train, y_train = make_gaussian_blobs(n_per_class=100, n_classes=3, seed=1)
        X_test, y_test = make_gaussian_blobs(n_per_class=100, n_classes=3, seed=2)
        model = MLP(layer_sizes=[2, 16, 3], learning_rate=0.01, max_epochs=1000, seed=0)

        # Act
        model.fit(X_train, y_train)

        # Assert
        assert model.score(X_test, y_test) > 0.90

    def test_classifies_five_gaussian_blobs(self) -> None:
        # Arrange
        X_train, y_train = make_gaussian_blobs(n_per_class=80, n_classes=5, seed=1)
        X_test, y_test = make_gaussian_blobs(n_per_class=80, n_classes=5, seed=2)
        model = MLP(layer_sizes=[2, 32, 5], learning_rate=0.01, max_epochs=1500, seed=0)

        # Act
        model.fit(X_train, y_train)

        # Assert
        assert model.score(X_test, y_test) > 0.85


class TestPredictionContract:
    def test_predict_returns_class_indices(self) -> None:
        # Arrange
        X, y = make_gaussian_blobs(n_per_class=50, n_classes=3, seed=0)
        model = MLP(layer_sizes=[2, 8, 3], learning_rate=0.01, max_epochs=200, seed=0)
        model.fit(X, y)

        # Act
        predictions = model.predict(X)

        # Assert
        assert predictions.shape == (len(y),)
        assert predictions.dtype.kind in ("i", "u")
        assert set(np.unique(predictions).tolist()) <= {0, 1, 2}

    def test_predict_proba_returns_valid_distribution(self) -> None:
        # Arrange
        X, y = make_gaussian_blobs(n_per_class=50, n_classes=3, seed=0)
        model = MLP(layer_sizes=[2, 8, 3], learning_rate=0.01, max_epochs=200, seed=0)
        model.fit(X, y)

        # Act
        probs = model.predict_proba(X)

        # Assert
        assert probs.shape == (len(y), 3)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)


class TestArchitecture:
    """PyTorch-specific: the parameter count is derivable from layer_sizes."""

    def test_parameter_count_matches_manual_math(self) -> None:
        # For layer_sizes [2, 8, 3]: Linear(2, 8) has 2*8 + 8 = 24 params,
        # Linear(8, 3) has 8*3 + 3 = 27 params. Total = 51.

        # Arrange
        model = MLP(layer_sizes=[2, 8, 3], learning_rate=0.01, max_epochs=1, seed=0)
        expected = 2 * 8 + 8 + 8 * 3 + 3

        # Act
        actual = model.n_parameters()

        # Assert
        assert actual == expected

    def test_parameter_count_scales_with_depth(self) -> None:
        # Arrange
        model = MLP(layer_sizes=[2, 8, 8, 8, 3], learning_rate=0.01, max_epochs=1, seed=0)
        expected = (2 * 8 + 8) + (8 * 8 + 8) * 2 + (8 * 3 + 3)

        # Act
        actual = model.n_parameters()

        # Assert
        assert actual == expected


class TestSaveLoad:
    """Trained weights round-trip through save/load without prediction change."""

    def test_state_dict_save_and_reload_preserves_predictions(self) -> None:
        # Arrange
        import torch

        X, y = make_gaussian_blobs(n_per_class=50, n_classes=3, seed=0)
        original = MLP(layer_sizes=[2, 8, 3], learning_rate=0.01, max_epochs=200, seed=0)
        original.fit(X, y)
        original_preds = original.predict(X)

        # Act
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            torch.save(original.net.state_dict(), tmp.name)  # type: ignore[attr-defined]
            reloaded = MLP(layer_sizes=[2, 8, 3], learning_rate=0.01, max_epochs=1, seed=999)
            reloaded.net.load_state_dict(torch.load(tmp.name, weights_only=True))  # type: ignore[attr-defined]
        reloaded_preds = reloaded.predict(X)

        # Assert
        np.testing.assert_array_equal(original_preds, reloaded_preds)


class TestPreconditions:
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
        # Arrange
        model = MLP(layer_sizes=[2, 4, 3], learning_rate=0.01, max_epochs=10)
        X = np.zeros((5, 4))

        # Act / Assert
        with pytest.raises(AssertionError, match="expected 2 features"):
            model.predict(X)

    def test_rejects_shape_mismatch_between_data_and_labels(self) -> None:
        # Arrange
        model = MLP(layer_sizes=[2, 4, 3], learning_rate=0.01, max_epochs=10)
        X = np.zeros((5, 2))
        y = np.zeros(3, dtype=np.int64)

        # Act / Assert
        with pytest.raises(AssertionError, match=r"rows.*labels"):
            model.fit(X, y)

    def test_rejects_labels_outside_valid_range(self) -> None:
        # Arrange
        model = MLP(layer_sizes=[2, 4, 3], learning_rate=0.01, max_epochs=10)
        X = np.zeros((3, 2))
        y = np.array([0, 1, 99], dtype=np.int64)

        # Act / Assert
        with pytest.raises(AssertionError, match="labels must be in"):
            model.fit(X, y)


class TestDatasets:
    def test_xor_dataset(self) -> None:
        # Act
        X, y = make_xor()

        # Assert
        assert X.shape == (4, 2)
        assert y.shape == (4,)
        assert set(np.unique(y).tolist()) == {0, 1}

    def test_gaussian_blobs_shape(self) -> None:
        # Act
        X, y = make_gaussian_blobs(n_per_class=25, n_classes=4)

        # Assert
        assert X.shape == (100, 2)
        assert y.shape == (100,)
        assert set(np.unique(y).tolist()) == {0, 1, 2, 3}
