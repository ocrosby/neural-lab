"""Same MLP architecture as project 02, rebuilt in PyTorch.

Compare this file to `projects/02-mlp-from-scratch/mlp.py`. Same math, same
behavior, ~1/4 the code — because PyTorch pre-builds:

- The autograd engine (which project 03 built by hand)
- Standard layers (`nn.Linear` = weight + bias with He init by default)
- Optimizers (`torch.optim.Adam` is a drop-in replacement for the manual
  weight update loop)
- Loss functions (`nn.CrossEntropyLoss` fuses log-softmax + NLL for numerical
  stability and gives you the `p - y_onehot` gradient shortcut for free)

The API — layer_sizes, fit/predict/predict_proba/score — is identical to
project 02, so the tests are almost identical too.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class MLP:
    """PyTorch MLP with ReLU hidden layers and softmax output.

    Attributes:
        layer_sizes: Widths of each layer, input to output.
        learning_rate: Adam step size.
        max_epochs: Full-batch training passes.
        net: The underlying `nn.Sequential` module.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        learning_rate: float = 0.01,
        max_epochs: int = 1000,
        seed: int = 42,
    ) -> None:
        assert len(layer_sizes) >= 2, (
            f"MLP needs at least 2 layers (input + output), got {layer_sizes}"
        )
        assert all(n > 0 for n in layer_sizes), (
            f"every layer size must be positive, got {layer_sizes}"
        )
        assert learning_rate > 0, f"learning_rate must be positive, got {learning_rate}"
        assert max_epochs > 0, f"max_epochs must be positive, got {max_epochs}"

        torch.manual_seed(seed)
        self.layer_sizes = list(layer_sizes)
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.n_classes = layer_sizes[-1]

        layers: list[nn.Module] = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.net.parameters())

    def fit(self, X: np.ndarray, y: np.ndarray) -> int:
        assert X.ndim == 2, f"X must be 2D, got shape {X.shape}"
        assert y.ndim == 1, f"y must be 1D, got shape {y.shape}"
        assert X.shape[0] == y.shape[0], f"X has {X.shape[0]} rows but y has {y.shape[0]} labels"
        assert X.shape[1] == self.layer_sizes[0], (
            f"expected {self.layer_sizes[0]} features, got {X.shape[1]}"
        )
        assert y.dtype.kind in ("i", "u"), f"y must be integer labels, got dtype {y.dtype}"
        assert y.min() >= 0 and y.max() < self.n_classes, (
            f"labels must be in [0, {self.n_classes - 1}], got min={y.min()}, max={y.max()}"
        )

        X_t = torch.from_numpy(X).float()
        y_t = torch.from_numpy(y).long()

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)

        self.net.train()
        for _epoch in range(self.max_epochs):
            optimizer.zero_grad()
            logits = self.net(X_t)
            loss = loss_fn(logits, y_t)
            loss.backward()
            optimizer.step()

        return self.max_epochs

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert X.ndim == 2, f"X must be 2D, got shape {X.shape}"
        assert X.shape[1] == self.layer_sizes[0], (
            f"expected {self.layer_sizes[0]} features, got {X.shape[1]}"
        )
        X_t = torch.from_numpy(X).float()
        self.net.eval()
        with torch.no_grad():
            logits = self.net(X_t)
            probs = torch.softmax(logits, dim=-1)
        return probs.numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1).astype(np.int64)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float((self.predict(X) == y).mean())


def make_xor() -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    y = np.array([0, 1, 1, 0], dtype=np.int64)
    return X, y


def make_gaussian_blobs(
    n_per_class: int = 100, n_classes: int = 3, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Well-separated 2D Gaussian blobs, one per class, evenly spaced on a ring.

    Identical to `projects/02-mlp-from-scratch/mlp.py`'s dataset — the point is
    to exercise the same problem with the same data, so any difference in
    accuracy is attributable to the framework, not the input distribution.
    """
    assert n_per_class > 0, f"n_per_class must be positive, got {n_per_class}"
    assert n_classes >= 2, f"n_classes must be >= 2, got {n_classes}"

    rng = np.random.default_rng(seed)
    centers = np.stack(
        [
            5.0 * np.array([np.cos(2 * np.pi * k / n_classes), np.sin(2 * np.pi * k / n_classes)])
            for k in range(n_classes)
        ]
    )
    X_parts, y_parts = [], []
    for k in range(n_classes):
        X_parts.append(rng.normal(loc=centers[k], scale=0.7, size=(n_per_class, 2)))
        y_parts.append(np.full(n_per_class, k, dtype=np.int64))
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]
