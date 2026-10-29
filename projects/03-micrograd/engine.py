"""A scalar-valued autograd engine — the guts of PyTorch, in ~150 lines.

Every `Value` wraps a scalar and records every operation into a DAG. Each op
attaches a `_backward` closure that knows how to compute gradients w.r.t. its
children given the gradient w.r.t. itself. On `.backward()`, we walk the DAG
in reverse topological order and let each closure run.

The chain rule shows up as one line inside each op's `_backward`:

    self.grad += (local_derivative) * out.grad

`out.grad` is the gradient from downstream (upstream in reverse). Multiplying
by the local derivative gives the gradient contribution to `self`. Summation
across paths (a variable used multiple times) happens automatically via `+=`.

Scale this from scalar to n-dim tensor and swap Python for CUDA kernels and
you have PyTorch. The algorithm is identical.
"""

from __future__ import annotations

import math


class Value:
    """A node in a scalar autograd graph.

    Attributes:
        data: The forward-pass value.
        grad: The gradient of some downstream scalar w.r.t. `self`, accumulated
            by `.backward()`. Zero until `.backward()` is called.
    """

    def __init__(self, data: float, _children: tuple[Value, ...] = (), _op: str = "") -> None:
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(_children)
        self._backward: callable = lambda: None  # type: ignore[valid-type]
        self._op = _op

    def __add__(self, other: object) -> Value:
        other_v = other if isinstance(other, Value) else Value(float(other))  # type: ignore[arg-type]
        out = Value(self.data + other_v.data, (self, other_v), "+")

        def _backward() -> None:
            self.grad += out.grad
            other_v.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: object) -> Value:
        other_v = other if isinstance(other, Value) else Value(float(other))  # type: ignore[arg-type]
        out = Value(self.data * other_v.data, (self, other_v), "*")

        def _backward() -> None:
            self.grad += other_v.data * out.grad
            other_v.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, exponent: object) -> Value:
        assert isinstance(exponent, int | float), (
            f"only int/float exponents are supported, got {type(exponent).__name__}"
        )
        exp_f = float(exponent)
        out = Value(self.data**exp_f, (self,), f"**{exp_f}")

        def _backward() -> None:
            self.grad += exp_f * (self.data ** (exp_f - 1)) * out.grad

        out._backward = _backward
        return out

    def __neg__(self) -> Value:
        return self * -1.0

    def __sub__(self, other: object) -> Value:
        return self + (-other if isinstance(other, Value) else -float(other))  # type: ignore[arg-type]

    def __truediv__(self, other: object) -> Value:
        if isinstance(other, Value):
            return self * (other**-1.0)
        return self * (1.0 / float(other))  # type: ignore[arg-type]

    def __radd__(self, other: object) -> Value:
        return self + other

    def __rmul__(self, other: object) -> Value:
        return self * other

    def __rsub__(self, other: object) -> Value:
        return (-self) + other

    def __rtruediv__(self, other: object) -> Value:
        return (self**-1.0) * other

    def relu(self) -> Value:
        out = Value(max(0.0, self.data), (self,), "ReLU")

        def _backward() -> None:
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> Value:
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward() -> None:
            self.grad += (1.0 - t * t) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> Value:
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")

        def _backward() -> None:
            self.grad += e * out.grad

        out._backward = _backward
        return out

    def log(self) -> Value:
        assert self.data > 0.0, f"log requires positive input, got {self.data}"
        out = Value(math.log(self.data), (self,), "log")

        def _backward() -> None:
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def backward(self) -> None:
        """Reverse-mode autodiff: propagate gradients from self back to leaves."""
        topo: list[Value] = []
        visited: set[Value] = set()

        def build(v: Value) -> None:
            if v in visited:
                return
            visited.add(v)
            for child in v._prev:
                build(child)
            topo.append(v)

        build(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def __repr__(self) -> str:
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
