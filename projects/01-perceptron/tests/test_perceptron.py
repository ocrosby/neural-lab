"""Black-box tests for the Rosenblatt perceptron.

Following `rules/black-box-testing.md`: assert on observable outcomes
(accuracy, convergence, prediction shape), never on internal weight values
or update-step counts. Refactoring the training loop should not break these.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_PROJECT_DIR = Path(__file__).resolve().parents[1]
_MODULE_PATH = _PROJECT_DIR / "perceptron.py"
_spec = importlib.util.spec_from_file_location("perceptron_mod", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
perceptron_mod = importlib.util.module_from_spec(_spec)
sys.modules["perceptron_mod"] = perceptron_mod
_spec.loader.exec_module(perceptron_mod)

Perceptron = perceptron_mod.Perceptron
make_linearly_separable = perceptron_mod.make_linearly_separable
make_xor = perceptron_mod.make_xor


class TestPerceptronConvergence:
    """Behavioral property: converges on separable data, fails on XOR."""

    def test_converges_on_linearly_separable_data(self) -> None:
        X, y = make_linearly_separable(n_per_class=50, seed=42)
        model = Perceptron(n_features=2)
        epochs = model.fit(X, y)
        assert epochs < model.max_epochs, "should converge before hitting max_epochs"
        assert model.score(X, y) == 1.0, "should achieve perfect accuracy on separable data"

    def test_fails_to_converge_on_xor(self) -> None:
        X, y = make_xor()
        model = Perceptron(n_features=2, max_epochs=100)
        epochs = model.fit(X, y)
        assert epochs == model.max_epochs, "must exhaust max_epochs on non-separable XOR"
        assert model.score(X, y) < 1.0, "cannot classify XOR perfectly"

    def test_generalizes_to_held_out_data(self) -> None:
        X_train, y_train = make_linearly_separable(n_per_class=100, seed=1)
        X_test, y_test = make_linearly_separable(n_per_class=100, seed=2)
        model = Perceptron(n_features=2)
        model.fit(X_train, y_train)
        assert model.score(X_test, y_test) > 0.95, "should generalize to a fresh sample"


class TestPerceptronPredictionContract:
    """Prediction API: shape, label domain, and dimensionality checks."""

    def test_predict_returns_correct_shape(self) -> None:
        X, y = make_linearly_separable(n_per_class=10)
        model = Perceptron(n_features=2)
        model.fit(X, y)
        predictions = model.predict(X)
        assert predictions.shape == (len(y),)

    def test_predict_returns_labels_in_valid_set(self) -> None:
        X, y = make_linearly_separable(n_per_class=10)
        model = Perceptron(n_features=2)
        model.fit(X, y)
        predictions = model.predict(X)
        assert set(np.unique(predictions).tolist()) <= {-1, 1}


class TestPerceptronPreconditions:
    """Precondition assertions fire before doing any work."""

    def test_rejects_non_positive_n_features(self) -> None:
        with pytest.raises(AssertionError, match="n_features must be positive"):
            Perceptron(n_features=0)

    def test_rejects_non_positive_learning_rate(self) -> None:
        with pytest.raises(AssertionError, match="learning_rate must be positive"):
            Perceptron(n_features=2, learning_rate=0.0)

    def test_rejects_shape_mismatch_between_data_and_labels(self) -> None:
        model = Perceptron(n_features=2)
        X = np.zeros((5, 2))
        y = np.ones(3, dtype=np.int64)
        with pytest.raises(AssertionError, match=r"rows.*labels"):
            model.fit(X, y)

    def test_rejects_labels_outside_valid_set(self) -> None:
        model = Perceptron(n_features=2)
        X = np.zeros((3, 2))
        y = np.array([0, 1, 2], dtype=np.int64)
        with pytest.raises(AssertionError, match="labels must be in"):
            model.fit(X, y)

    def test_rejects_wrong_feature_count_at_predict(self) -> None:
        model = Perceptron(n_features=2)
        X_train, y_train = make_linearly_separable(n_per_class=10)
        model.fit(X_train, y_train)
        X_wrong = np.zeros((5, 3))
        with pytest.raises(AssertionError, match="expected 2 features"):
            model.predict(X_wrong)


class TestDatasets:
    """Sanity checks on the toy datasets."""

    def test_linearly_separable_dataset_shape(self) -> None:
        X, y = make_linearly_separable(n_per_class=25)
        assert X.shape == (50, 2)
        assert y.shape == (50,)
        assert set(np.unique(y).tolist()) == {-1, 1}

    def test_xor_dataset(self) -> None:
        X, y = make_xor()
        assert X.shape == (4, 2)
        assert y.shape == (4,)
        assert set(np.unique(y).tolist()) == {-1, 1}
