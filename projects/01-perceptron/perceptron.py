"""Rosenblatt's 1958 perceptron in pure NumPy.

The perceptron is a linear binary classifier. Given inputs x in R^d and
weights w in R^d plus a bias b, it predicts:

    y_hat = sign(w . x + b)

The learning rule, iterated over training examples (x_i, y_i in {-1, +1}):

    if y_i * (w . x_i + b) <= 0:      # misclassified or on the boundary
        w <- w + eta * y_i * x_i
        b <- b + eta * y_i

Rosenblatt proved this converges to a separating hyperplane in finite steps
if one exists. It cannot solve XOR — that limitation motivated the MLP
(project 02).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Perceptron:
    """Single-layer perceptron.

    Attributes:
        n_features: Dimensionality of input vectors.
        learning_rate: Step size for the weight update. Rosenblatt's original
            proof holds for any positive eta; larger eta converges faster on
            linearly-separable data but oscillates more near the boundary.
        max_epochs: Cap on training passes. Bounds runtime on non-separable
            data where the update rule never terminates.
    """

    n_features: int
    learning_rate: float = 1.0
    max_epochs: int = 100
    weights: np.ndarray = field(init=False)
    bias: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        assert self.n_features > 0, f"n_features must be positive, got {self.n_features}"
        assert self.learning_rate > 0, f"learning_rate must be positive, got {self.learning_rate}"
        assert self.max_epochs > 0, f"max_epochs must be positive, got {self.max_epochs}"
        self.weights = np.zeros(self.n_features, dtype=np.float64)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels in {-1, +1} for a batch of inputs."""
        assert X.ndim == 2, f"X must be 2D (n_samples, n_features), got shape {X.shape}"
        assert X.shape[1] == self.n_features, (
            f"expected {self.n_features} features, got {X.shape[1]}"
        )
        scores = X @ self.weights + self.bias
        return np.where(scores >= 0, 1, -1).astype(np.int64)

    def fit(self, X: np.ndarray, y: np.ndarray) -> int:
        """Train until convergence or max_epochs. Returns epochs used.

        Args:
            X: Shape (n_samples, n_features).
            y: Shape (n_samples,), labels in {-1, +1}.

        Returns:
            Number of epochs actually run. Equal to max_epochs if the data is
            not linearly separable within that cap.
        """
        assert X.ndim == 2, f"X must be 2D, got shape {X.shape}"
        assert y.ndim == 1, f"y must be 1D, got shape {y.shape}"
        assert X.shape[0] == y.shape[0], f"X has {X.shape[0]} rows but y has {y.shape[0]} labels"
        assert set(np.unique(y).tolist()) <= {-1, 1}, (
            f"labels must be in {{-1, +1}}, got {np.unique(y).tolist()}"
        )

        for epoch in range(1, self.max_epochs + 1):
            errors = 0
            for x_i, y_i in zip(X, y, strict=True):
                margin = float(y_i) * (float(x_i @ self.weights) + self.bias)
                if margin <= 0:
                    self.weights += self.learning_rate * y_i * x_i
                    self.bias += self.learning_rate * float(y_i)
                    errors += 1
            if errors == 0:
                return epoch
        return self.max_epochs

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Fraction of correct predictions on (X, y)."""
        return float((self.predict(X) == y).mean())


def make_linearly_separable(n_per_class: int = 50, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Two 2D Gaussian blobs, clearly separable. For sanity checks and demos."""
    rng = np.random.default_rng(seed)
    pos = rng.normal(loc=[2.0, 2.0], scale=0.5, size=(n_per_class, 2))
    neg = rng.normal(loc=[-2.0, -2.0], scale=0.5, size=(n_per_class, 2))
    X = np.vstack([pos, neg])
    y = np.concatenate(
        [np.ones(n_per_class, dtype=np.int64), -np.ones(n_per_class, dtype=np.int64)]
    )
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def make_xor() -> tuple[np.ndarray, np.ndarray]:
    """The XOR dataset. Not linearly separable — perceptron cannot solve it.

    This is the whole point of the exercise. Try `Perceptron(2).fit(*make_xor())`
    and watch it fail to converge.
    """
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    y = np.array([-1, 1, 1, -1], dtype=np.int64)
    return X, y


def main() -> None:
    """Demo: train on a linearly-separable blob, then show XOR failure."""
    import sys

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    from shared.seeds import set_seed

    set_seed(42)

    print("=== Linearly-separable blobs ===")
    X, y = make_linearly_separable(n_per_class=50)
    model = Perceptron(n_features=2)
    epochs = model.fit(X, y)
    accuracy = model.score(X, y)
    print(f"converged in {epochs} epoch(s), accuracy = {accuracy:.2%}")
    print(f"weights = {model.weights}, bias = {model.bias:.4f}")

    print("\n=== XOR (should fail) ===")
    X_xor, y_xor = make_xor()
    model_xor = Perceptron(n_features=2, max_epochs=1000)
    epochs = model_xor.fit(X_xor, y_xor)
    accuracy = model_xor.score(X_xor, y_xor)
    print(f"stopped after {epochs} epoch(s), accuracy = {accuracy:.2%}")
    print("XOR is not linearly separable — this is the motivation for the MLP (project 02)")


if __name__ == "__main__":
    main()
