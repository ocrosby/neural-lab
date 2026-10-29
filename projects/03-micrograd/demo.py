"""Demo: use the scalar autograd engine to train an MLP on XOR.

Every gradient in this training loop was computed by `Value._backward`
closures walking a DAG of scalar ops. Not one numpy vector, no PyTorch,
no autograd magic — just the chain rule executed in reverse topological
order over ~1300 scalar operations per forward pass.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_HERE = Path(__file__).parent

_ENGINE_KEY = "micrograd_engine"
_engine_spec = importlib.util.spec_from_file_location(_ENGINE_KEY, _HERE / "engine.py")
assert _engine_spec is not None and _engine_spec.loader is not None
engine_mod = importlib.util.module_from_spec(_engine_spec)
sys.modules[_ENGINE_KEY] = engine_mod
_engine_spec.loader.exec_module(engine_mod)

_NN_KEY = "micrograd_nn"
_nn_spec = importlib.util.spec_from_file_location(_NN_KEY, _HERE / "nn.py")
assert _nn_spec is not None and _nn_spec.loader is not None
nn_mod = importlib.util.module_from_spec(_nn_spec)
sys.modules[_NN_KEY] = nn_mod
_nn_spec.loader.exec_module(nn_mod)

Value = engine_mod.Value
MLP = nn_mod.MLP


def main() -> None:
    sys.path.insert(0, str(_HERE.parents[1]))
    from shared.seeds import set_seed

    set_seed(0)

    X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    y_targets = [-1.0, 1.0, 1.0, -1.0]

    print("=== Simple scalar autograd example ===")
    a = Value(2.0)
    b = Value(3.0)
    c = a * b + a**2 - b / 2
    c.backward()
    print(f"c = a*b + a^2 - b/2  where a={a.data}, b={b.data}")
    print(f"c.data = {c.data:.4f}")
    print(f"dc/da  = {a.grad:.4f}  (analytical: b + 2a = 3 + 4 = 7)")
    print(f"dc/db  = {b.grad:.4f}  (analytical: a - 0.5 = 1.5)")

    print("\n=== Train an MLP on XOR through the engine ===")
    mlp = MLP(n_in=2, n_outs=[16, 16, 1], seed=1)
    print(f"parameters: {len(mlp.parameters())}")

    learning_rate = 0.05
    n_steps = 500
    for step in range(n_steps):
        preds = [mlp([Value(xi) for xi in x]) for x in X]
        loss = sum(((p - t) ** 2 for p, t in zip(preds, y_targets, strict=True)), Value(0.0))
        mlp.zero_grad()
        loss.backward()
        for p in mlp.parameters():
            p.data -= learning_rate * p.grad
        if step % 100 == 0 or step == n_steps - 1:
            print(f"step {step:4d}: loss = {loss.data:.6f}")

    print("\nfinal predictions vs targets:")
    for x, t in zip(X, y_targets, strict=True):
        p = mlp([Value(xi) for xi in x]).data  # type: ignore[union-attr]
        marker = "OK" if p * t > 0 else "FAIL"
        print(f"  x={x}  target={t:+.1f}  pred={p:+.4f}  [{marker}]")


if __name__ == "__main__":
    main()
