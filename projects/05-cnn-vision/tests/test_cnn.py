"""Black-box tests for the CNN.

Tests are fast — they train on a small synthetic image dataset (bright square
in one of 4 quadrants + noise). Real MNIST training lives in demo.py, not
tests, because CI shouldn't wait 30 seconds per epoch for a slow test.

The teaching contract:
- Architecture: given (N, C, H, W) input, forward produces (N, n_classes) logits
- Training: loss decreases across epochs, model achieves >80% on the synthetic task
- Filters: first-layer weights are inspectable and have expected shape
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

_PROJECT_DIR = Path(__file__).resolve().parents[1]

_cnn_spec = importlib.util.spec_from_file_location("cnn_mod", _PROJECT_DIR / "cnn.py")
assert _cnn_spec is not None and _cnn_spec.loader is not None
cnn_mod = importlib.util.module_from_spec(_cnn_spec)
sys.modules["cnn_mod"] = cnn_mod
_cnn_spec.loader.exec_module(cnn_mod)

_ds_spec = importlib.util.spec_from_file_location("cnn_datasets", _PROJECT_DIR / "datasets.py")
assert _ds_spec is not None and _ds_spec.loader is not None
ds_mod = importlib.util.module_from_spec(_ds_spec)
sys.modules["cnn_datasets"] = ds_mod
_ds_spec.loader.exec_module(ds_mod)

CNN = cnn_mod.CNN
make_quadrant_dataset = ds_mod.make_quadrant_dataset


def _loader(X: torch.Tensor, y: torch.Tensor, batch_size: int = 16) -> DataLoader:
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)


class TestArchitecture:
    def test_grayscale_forward_shape(self) -> None:
        # Arrange
        model = CNN(in_channels=1, image_size=28, n_classes=10, seed=0)
        X = torch.randn(4, 1, 28, 28)

        # Act
        probs = model.predict_proba(X)

        # Assert
        assert probs.shape == (4, 10)

    def test_rgb_forward_shape(self) -> None:
        # Arrange
        model = CNN(in_channels=3, image_size=32, n_classes=10, seed=0)
        X = torch.randn(2, 3, 32, 32)

        # Act
        probs = model.predict_proba(X)

        # Assert
        assert probs.shape == (2, 10)

    def test_first_conv_filters_shape(self) -> None:
        # Arrange
        model = CNN(in_channels=1, image_size=28, n_classes=10, conv_channels=[16, 32], seed=0)

        # Act
        filters = model.first_conv_filters()

        # Assert — 16 filters, each 1 input channel (grayscale), 3x3 kernel
        assert filters.shape == (16, 1, 3, 3)

    def test_n_parameters_positive(self) -> None:
        # Arrange
        model = CNN(in_channels=1, image_size=28, n_classes=10, seed=0)

        # Act
        n = model.n_parameters()

        # Assert
        assert n > 0


class TestTraining:
    def test_trains_on_quadrant_task(self) -> None:
        """A CNN with two conv layers should easily learn 'bright square in quadrant K'."""
        # Arrange
        torch.manual_seed(0)
        X_train, y_train = make_quadrant_dataset(n_per_class=64, n_classes=4, image_size=28, seed=1)
        X_test, y_test = make_quadrant_dataset(n_per_class=32, n_classes=4, image_size=28, seed=2)
        train_loader = _loader(X_train, y_train, batch_size=32)
        test_loader = _loader(X_test, y_test, batch_size=32)

        model = CNN(
            in_channels=1,
            image_size=28,
            n_classes=4,
            conv_channels=[8, 16],
            fc_features=32,
            learning_rate=0.005,
            seed=0,
        )

        # Act
        losses = model.fit(train_loader, n_epochs=5)
        accuracy = model.evaluate(test_loader)

        # Assert
        assert len(losses) == 5, "must return one loss per epoch"
        assert losses[-1] < losses[0], "loss should decrease across epochs"
        assert accuracy > 0.80, f"quadrant task should be easy, got {accuracy:.2%}"

    def test_predict_returns_class_indices(self) -> None:
        # Arrange
        torch.manual_seed(0)
        X, y = make_quadrant_dataset(n_per_class=16, n_classes=4, image_size=28, seed=1)
        model = CNN(
            in_channels=1,
            image_size=28,
            n_classes=4,
            conv_channels=[8, 16],
            fc_features=32,
            learning_rate=0.005,
            seed=0,
        )
        model.fit(_loader(X, y), n_epochs=2)

        # Act
        preds = model.predict(X)

        # Assert
        assert preds.shape == (len(y),)
        assert preds.dtype.kind in ("i", "u")
        assert set(np.unique(preds).tolist()) <= {0, 1, 2, 3}


class TestPredictionContract:
    def test_predict_proba_is_valid_distribution(self) -> None:
        # Arrange
        model = CNN(in_channels=1, image_size=28, n_classes=10, seed=0)
        X = torch.randn(5, 1, 28, 28)

        # Act
        probs = model.predict_proba(X)

        # Assert
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)


class TestPreconditions:
    def test_rejects_non_positive_in_channels(self) -> None:
        with pytest.raises(AssertionError, match="in_channels"):
            CNN(in_channels=0, image_size=28, n_classes=10)

    def test_rejects_non_positive_image_size(self) -> None:
        with pytest.raises(AssertionError, match="image_size"):
            CNN(in_channels=1, image_size=0, n_classes=10)

    def test_rejects_non_positive_n_classes(self) -> None:
        with pytest.raises(AssertionError, match="n_classes"):
            CNN(in_channels=1, image_size=28, n_classes=0)

    def test_rejects_non_positive_learning_rate(self) -> None:
        with pytest.raises(AssertionError, match="learning_rate"):
            CNN(in_channels=1, image_size=28, n_classes=10, learning_rate=0.0)

    def test_rejects_wrong_number_of_channels_at_predict(self) -> None:
        # Arrange
        model = CNN(in_channels=1, image_size=28, n_classes=10, seed=0)
        X_bad = torch.randn(2, 3, 28, 28)  # 3 channels, model expects 1

        # Act / Assert
        with pytest.raises(AssertionError, match=r"expected 1 channels"):
            model.predict(X_bad)

    def test_rejects_wrong_image_size_at_predict(self) -> None:
        # Arrange
        model = CNN(in_channels=1, image_size=28, n_classes=10, seed=0)
        X_bad = torch.randn(2, 1, 32, 32)

        # Act / Assert
        with pytest.raises(AssertionError, match=r"expected image size 28"):
            model.predict(X_bad)


class TestQuadrantDataset:
    def test_shape_and_labels(self) -> None:
        # Act
        X, y = make_quadrant_dataset(n_per_class=10, n_classes=4, image_size=28)

        # Assert
        assert X.shape == (40, 1, 28, 28)
        assert y.shape == (40,)
        assert set(y.unique().tolist()) == {0, 1, 2, 3}

    def test_dataset_is_learnable(self) -> None:
        """Signal-to-noise ratio must be high enough that a CNN can learn it."""
        # Arrange
        X, y = make_quadrant_dataset(n_per_class=50, n_classes=4, image_size=28, seed=0)

        # Assert — each class's mean image should have highest energy in its
        # designated quadrant. Otherwise the task isn't learnable and the
        # training test above is meaningless.
        for cls in range(4):
            mean_image = X[y == cls].mean(dim=0).squeeze()  # (H, W)
            h, w = mean_image.shape
            row = cls // 2
            col = cls % 2
            r_start, r_end = (0, h // 2) if row == 0 else (h // 2, h)
            c_start, c_end = (0, w // 2) if col == 0 else (w // 2, w)
            target_quadrant_energy = mean_image[r_start:r_end, c_start:c_end].sum()
            total_energy = mean_image.sum()
            assert target_quadrant_energy > total_energy * 0.5, (
                f"class {cls}'s signal should dominate its quadrant"
            )
