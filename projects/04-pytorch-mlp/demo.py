"""Demo: the same three problems from project 02, solved in PyTorch.

Same architecture, same data, same behavior — measured by the same test
contract. The runtime is ~10x shorter than project 02 because Adam converges
faster than full-batch GD, and PyTorch's fused kernels beat NumPy for the
sizes involved.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_HERE = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("pt_mlp", _HERE / "mlp.py")
assert _spec is not None and _spec.loader is not None
mlp_mod = importlib.util.module_from_spec(_spec)
sys.modules["pt_mlp"] = mlp_mod
_spec.loader.exec_module(mlp_mod)

MLP = mlp_mod.MLP
make_xor = mlp_mod.make_xor
make_gaussian_blobs = mlp_mod.make_gaussian_blobs


def main() -> None:
    sys.path.insert(0, str(_HERE.parents[1]))
    from shared.seeds import set_seed

    set_seed(0)

    print("=== XOR ===")
    X, y = make_xor()
    model = MLP(layer_sizes=[2, 8, 2], learning_rate=0.05, max_epochs=2000, seed=0)
    print(f"parameters: {model.n_parameters()}")
    model.fit(X, y)
    print(f"accuracy = {model.score(X, y):.2%}")
    print(f"predictions = {model.predict(X)}, true = {y}")

    print("\n=== 3-class Gaussian blobs ===")
    X_train, y_train = make_gaussian_blobs(n_per_class=100, n_classes=3, seed=1)
    X_test, y_test = make_gaussian_blobs(n_per_class=100, n_classes=3, seed=2)
    model = MLP(layer_sizes=[2, 16, 3], learning_rate=0.01, max_epochs=1000, seed=0)
    print(f"parameters: {model.n_parameters()}")
    model.fit(X_train, y_train)
    print(f"train accuracy = {model.score(X_train, y_train):.2%}")
    print(f"test accuracy  = {model.score(X_test, y_test):.2%}")

    print("\n=== 5-class Gaussian blobs ===")
    X_train, y_train = make_gaussian_blobs(n_per_class=80, n_classes=5, seed=1)
    X_test, y_test = make_gaussian_blobs(n_per_class=80, n_classes=5, seed=2)
    model = MLP(layer_sizes=[2, 32, 5], learning_rate=0.01, max_epochs=1500, seed=0)
    print(f"parameters: {model.n_parameters()}")
    model.fit(X_train, y_train)
    print(f"train accuracy = {model.score(X_train, y_train):.2%}")
    print(f"test accuracy  = {model.score(X_test, y_test):.2%}")


if __name__ == "__main__":
    main()
