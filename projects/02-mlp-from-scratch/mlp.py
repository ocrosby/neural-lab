"""Multi-layer perceptron with hand-derived backprop. Pure NumPy, no autograd.

The MLP overcomes the perceptron's fundamental limitation: it can represent
non-linearly-separable functions like XOR. The trick is a hidden layer with a
non-linear activation between two linear transforms. Without the non-linearity,
stacked linear layers collapse into a single linear layer (matrix multiplication
is associative) and gain nothing.

Architecture:
- Hidden layers: linear + ReLU
- Output layer: linear + softmax
- Loss: cross-entropy
- Optimizer: full-batch gradient descent
- Init: He initialization (W ~ N(0, sqrt(2/n_in))) for ReLU

The gradient for softmax + cross-entropy simplifies to `p - y_onehot`, which is
why every framework combines the two into one op — the intermediate softmax
values do not need to be differentiated through.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np


def _relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def _relu_grad(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(z.dtype)


def _softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    z_shifted = z - z.max(axis=-1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / exp_z.sum(axis=-1, keepdims=True)


def _one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((y.shape[0], n_classes), dtype=np.float64)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


class MLP:
    """Multi-layer perceptron with ReLU hidden layers and softmax output.

    Attributes:
        layer_sizes: Widths of each layer, input to output. `[2, 8, 3]` means
            2-dim input, one hidden layer of 8 units, 3-class output.
        learning_rate: Step size for SGD updates.
        max_epochs: Full-batch gradient-descent passes.
        weights: Per-layer weight matrices; `weights[l]` has shape
            `(layer_sizes[l], layer_sizes[l+1])`.
        biases: Per-layer bias vectors; `biases[l]` has shape `(layer_sizes[l+1],)`.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        learning_rate: float = 0.1,
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

        self.layer_sizes = list(layer_sizes)
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.n_classes = layer_sizes[-1]

        rng = np.random.default_rng(seed)
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        for n_in, n_out in pairwise(layer_sizes):
            scale = np.sqrt(2.0 / n_in)
            self.weights.append(rng.normal(0.0, scale, size=(n_in, n_out)))
            self.biases.append(np.zeros(n_out, dtype=np.float64))

    def _forward(self, X: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Return (activations, pre_activations) per layer.

        `activations[0]` is the input; `activations[-1]` is the softmax output.
        `pre_activations[l]` is `z = a_{l-1} @ W_l + b_l` before the activation.
        """
        activations = [X]
        pre_activations: list[np.ndarray] = []
        n_layers = len(self.weights)
        for layer_idx, (W, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            z = activations[-1] @ W + b
            pre_activations.append(z)
            is_output = layer_idx == n_layers - 1
            a = _softmax(z) if is_output else _relu(z)
            activations.append(a)
        return activations, pre_activations

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Mean cross-entropy loss over the batch."""
        activations, _ = self._forward(X)
        probs = activations[-1]
        eps = 1e-12
        picked = probs[np.arange(len(y)), y]
        return float(-np.log(np.clip(picked, eps, 1.0)).mean())

    def _compute_gradients(
        self, X: np.ndarray, y: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Backprop: return list of (dW, db) tuples, one per layer.

        For softmax + cross-entropy, the output-layer pre-activation gradient
        simplifies to `p - y_onehot`, dividing by N for the mean loss. Every
        hidden layer's gradient is the upstream gradient times `relu'(z)`.
        """
        n_samples = X.shape[0]
        activations, pre_activations = self._forward(X)
        y_onehot = _one_hot(y, self.n_classes)

        dz = (activations[-1] - y_onehot) / n_samples

        grads: list[tuple[np.ndarray, np.ndarray]] = []
        for layer_idx in reversed(range(len(self.weights))):
            a_prev = activations[layer_idx]
            dW = a_prev.T @ dz
            db = dz.sum(axis=0)
            grads.append((dW, db))
            if layer_idx > 0:
                da_prev = dz @ self.weights[layer_idx].T
                dz = da_prev * _relu_grad(pre_activations[layer_idx - 1])

        grads.reverse()
        return grads

    def fit(self, X: np.ndarray, y: np.ndarray) -> int:
        """Full-batch gradient descent for `max_epochs`. Returns epochs run."""
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

        for _epoch in range(self.max_epochs):
            grads = self._compute_gradients(X, y)
            for layer_idx, (dW, db) in enumerate(grads):
                self.weights[layer_idx] -= self.learning_rate * dW
                self.biases[layer_idx] -= self.learning_rate * db
        return self.max_epochs

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Softmax class probabilities. Shape `(n_samples, n_classes)`."""
        assert X.ndim == 2, f"X must be 2D, got shape {X.shape}"
        assert X.shape[1] == self.layer_sizes[0], (
            f"expected {self.layer_sizes[0]} features, got {X.shape[1]}"
        )
        activations, _ = self._forward(X)
        return activations[-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Class indices via argmax of softmax output."""
        return np.argmax(self.predict_proba(X), axis=1).astype(np.int64)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Fraction of correct predictions on (X, y)."""
        return float((self.predict(X) == y).mean())


def make_xor() -> tuple[np.ndarray, np.ndarray]:
    """The XOR dataset. Not linearly separable — MLP with hidden layer solves it."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    y = np.array([0, 1, 1, 0], dtype=np.int64)
    return X, y


def make_gaussian_blobs(
    n_per_class: int = 100, n_classes: int = 3, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Well-separated 2D Gaussian blobs, one per class, evenly spaced on a ring."""
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


def main() -> None:
    """Demo: MLP solves XOR (which the perceptron couldn't), then multiclass."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from shared.seeds import set_seed

    set_seed(0)

    print("=== XOR (perceptron cannot; MLP can) ===")
    X, y = make_xor()
    model = MLP(layer_sizes=[2, 8, 2], learning_rate=0.5, max_epochs=5000, seed=0)
    model.fit(X, y)
    print(f"accuracy = {model.score(X, y):.2%}")
    print(f"predictions = {model.predict(X)}, true = {y}")

    print("\n=== 3-class Gaussian blobs ===")
    X_train, y_train = make_gaussian_blobs(n_per_class=100, n_classes=3, seed=1)
    X_test, y_test = make_gaussian_blobs(n_per_class=100, n_classes=3, seed=2)
    model = MLP(layer_sizes=[2, 16, 3], learning_rate=0.1, max_epochs=2000, seed=0)
    model.fit(X_train, y_train)
    print(f"train accuracy = {model.score(X_train, y_train):.2%}")
    print(f"test accuracy  = {model.score(X_test, y_test):.2%}")

    print("\n=== 5-class Gaussian blobs ===")
    X_train, y_train = make_gaussian_blobs(n_per_class=80, n_classes=5, seed=1)
    X_test, y_test = make_gaussian_blobs(n_per_class=80, n_classes=5, seed=2)
    model = MLP(layer_sizes=[2, 32, 5], learning_rate=0.1, max_epochs=3000, seed=0)
    model.fit(X_train, y_train)
    print(f"train accuracy = {model.score(X_train, y_train):.2%}")
    print(f"test accuracy  = {model.score(X_test, y_test):.2%}")


if __name__ == "__main__":
    main()
