"""Neural-network primitives built on the autograd engine.

Nothing new happens here — every operation bottoms out in `Value` ops, and
gradients flow through the same DAG that `engine.py` already knows how to
walk. The whole point is to show that PyTorch's `nn.Module` is a thin wrapper
over its autograd engine, and so is this. Once you have `Value`, the neural
network layer is bookkeeping.

Loading modules by file path (via importlib) keeps this project independent
of Python-package layout — matches how the tests load `engine`.
"""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

_ENGINE_KEY = "micrograd_engine"
if _ENGINE_KEY in sys.modules:
    _engine_mod = sys.modules[_ENGINE_KEY]
else:
    _engine_spec = importlib.util.spec_from_file_location(
        _ENGINE_KEY, Path(__file__).parent / "engine.py"
    )
    assert _engine_spec is not None and _engine_spec.loader is not None
    _engine_mod = importlib.util.module_from_spec(_engine_spec)
    sys.modules[_ENGINE_KEY] = _engine_mod
    _engine_spec.loader.exec_module(_engine_mod)
Value = _engine_mod.Value


class Module:
    """Base class. Subclasses implement `parameters()` and are callable."""

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self) -> list:  # type: ignore[type-arg]
        return []


class Neuron(Module):
    """A single artificial neuron: linear combination + optional ReLU."""

    def __init__(self, n_in: int, nonlin: bool = True, seed: int = 42) -> None:
        assert n_in > 0, f"n_in must be positive, got {n_in}"
        rng = random.Random(seed)
        self.w = [Value(rng.uniform(-1.0, 1.0)) for _ in range(n_in)]
        self.b = Value(0.0)
        self.nonlin = nonlin
        self.n_in = n_in

    def __call__(self, x: list) -> Value:  # type: ignore[type-arg]
        assert len(x) == self.n_in, f"expected {self.n_in} inputs, got {len(x)}"
        act = self.b
        for wi, xi in zip(self.w, x, strict=True):
            act = act + wi * xi
        return act.relu() if self.nonlin else act

    def parameters(self) -> list:  # type: ignore[type-arg]
        return [*self.w, self.b]


class Layer(Module):
    """A row of `n_out` neurons, each taking the same `n_in`-dim input."""

    def __init__(self, n_in: int, n_out: int, nonlin: bool = True, seed: int = 42) -> None:
        assert n_out > 0, f"n_out must be positive, got {n_out}"
        self.neurons = [Neuron(n_in, nonlin=nonlin, seed=seed + i) for i in range(n_out)]

    def __call__(self, x: list) -> list:  # type: ignore[type-arg]
        return [n(x) for n in self.neurons]

    def parameters(self) -> list:  # type: ignore[type-arg]
        return [p for n in self.neurons for p in n.parameters()]


class MLP(Module):
    """Stack of `Layer`s. Output layer is linear (no ReLU); hidden layers use ReLU."""

    def __init__(self, n_in: int, n_outs: list[int], seed: int = 42) -> None:
        assert len(n_outs) >= 1, "MLP needs at least one output layer"
        sizes = [n_in, *n_outs]
        n_layers = len(n_outs)
        self.layers = [
            Layer(
                sizes[i],
                sizes[i + 1],
                nonlin=(i < n_layers - 1),
                seed=seed + i * 1000,
            )
            for i in range(n_layers)
        ]

    def __call__(self, x: list):  # type: ignore[type-arg,no-untyped-def]
        current: list = [xi if isinstance(xi, Value) else Value(xi) for xi in x]
        for layer in self.layers:
            current = layer(current)
        return current[0] if len(current) == 1 else current

    def parameters(self) -> list:  # type: ignore[type-arg]
        return [p for layer in self.layers for p in layer.parameters()]
