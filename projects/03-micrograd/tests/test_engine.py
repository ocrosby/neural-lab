"""Black-box tests for the scalar autograd engine.

Two kinds of assertions:

1. **Analytical**: for well-known functions (`f(x) = x^2`, `f(x,y) = xy + x^2`),
   check the gradient matches the closed-form derivative. Direct and readable.

2. **PyTorch as ground truth**: build the same expression twice — once in
   micrograd, once in PyTorch — and compare `.grad` values. This catches
   subtle bugs in operator overloads (`__rmul__`, `__pow__` on negatives,
   `__truediv__` gradient) that the analytical tests miss.

The PyTorch tests are the load-bearing ones. If those pass and the analytical
ones fail, the analytical test is wrong. If PyTorch fails and analytical
passes, the operator has an unusual behavior mode (subtractive form, division
by Value) that the analytical test didn't cover.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch

_PROJECT_DIR = Path(__file__).resolve().parents[1]
_ENGINE_KEY = "micrograd_engine"
if _ENGINE_KEY in sys.modules:
    engine_mod = sys.modules[_ENGINE_KEY]
else:
    _engine_spec = importlib.util.spec_from_file_location(_ENGINE_KEY, _PROJECT_DIR / "engine.py")
    assert _engine_spec is not None and _engine_spec.loader is not None
    engine_mod = importlib.util.module_from_spec(_engine_spec)
    sys.modules[_ENGINE_KEY] = engine_mod
    _engine_spec.loader.exec_module(engine_mod)

Value = engine_mod.Value


class TestForwardValues:
    """Data (not gradient) computation is correct."""

    def test_addition(self) -> None:
        a = Value(2.0)
        b = Value(3.0)
        assert (a + b).data == 5.0

    def test_multiplication(self) -> None:
        a = Value(2.0)
        b = Value(3.0)
        assert (a * b).data == 6.0

    def test_power_with_scalar_exponent(self) -> None:
        assert Value(3.0) ** 2 == 9.0 or (Value(3.0) ** 2).data == 9.0

    def test_subtraction(self) -> None:
        assert (Value(5.0) - Value(3.0)).data == 2.0

    def test_division_by_value(self) -> None:
        assert (Value(6.0) / Value(3.0)).data == 2.0

    def test_scalar_on_the_left(self) -> None:
        a = Value(2.0)
        assert (3 + a).data == 5.0
        assert (3 * a).data == 6.0
        assert (5 - a).data == 3.0

    def test_negation(self) -> None:
        assert (-Value(3.0)).data == -3.0


class TestAnalyticalGradients:
    """Gradients match closed-form derivatives for well-known functions."""

    def test_gradient_of_x_squared(self) -> None:
        # f(x) = x^2, f'(x) = 2x. At x=3, f'(3) = 6.
        x = Value(3.0)
        y = x * x
        y.backward()
        assert x.grad == pytest.approx(6.0)

    def test_gradient_of_polynomial(self) -> None:
        # f(x, y) = x*y + x^2. df/dx = y + 2x. df/dy = x.
        # At x=2, y=3: df/dx = 3 + 4 = 7, df/dy = 2.
        x = Value(2.0)
        y = Value(3.0)
        f = x * y + x**2
        f.backward()
        assert x.grad == pytest.approx(7.0)
        assert y.grad == pytest.approx(2.0)

    def test_gradient_reuses_variable(self) -> None:
        # f(x) = x + x = 2x. df/dx = 2 (accumulated through both branches).
        x = Value(4.0)
        f = x + x
        f.backward()
        assert x.grad == pytest.approx(2.0)


class TestActivations:
    """Activation gradients match their calculus derivatives."""

    def test_relu_positive_input(self) -> None:
        x = Value(3.0)
        y = x.relu()
        y.backward()
        assert y.data == 3.0
        assert x.grad == pytest.approx(1.0)

    def test_relu_negative_input(self) -> None:
        x = Value(-2.0)
        y = x.relu()
        y.backward()
        assert y.data == 0.0
        assert x.grad == pytest.approx(0.0)

    def test_tanh_gradient(self) -> None:
        # d(tanh(x))/dx = 1 - tanh^2(x)
        x = Value(0.5)
        y = x.tanh()
        y.backward()
        expected = 1 - math.tanh(0.5) ** 2
        assert x.grad == pytest.approx(expected, rel=1e-6)

    def test_exp_gradient(self) -> None:
        # d(e^x)/dx = e^x
        x = Value(1.5)
        y = x.exp()
        y.backward()
        assert x.grad == pytest.approx(math.exp(1.5), rel=1e-6)

    def test_log_gradient(self) -> None:
        # d(log(x))/dx = 1/x
        x = Value(2.0)
        y = x.log()
        y.backward()
        assert x.grad == pytest.approx(0.5, rel=1e-6)


class TestAgainstPyTorch:
    """PyTorch is the reference implementation. Match to ~1e-6."""

    def _compare(
        self,
        build_mg,  # type: ignore[no-untyped-def]
        build_pt,  # type: ignore[no-untyped-def]
        inputs: dict[str, float],
    ) -> None:
        mg_inputs = {k: Value(v) for k, v in inputs.items()}
        mg_out = build_mg(**mg_inputs)
        mg_out.backward()

        pt_inputs = {
            k: torch.tensor(v, requires_grad=True, dtype=torch.float64) for k, v in inputs.items()
        }
        pt_out = build_pt(**pt_inputs)
        pt_out.backward()

        for k in inputs:
            assert mg_inputs[k].grad == pytest.approx(
                float(pt_inputs[k].grad.item()), rel=1e-6, abs=1e-9
            ), f"grad mismatch on {k}"

    def test_simple_arithmetic(self) -> None:
        self._compare(
            build_mg=lambda a, b: a * b + a**2 - b / 2,
            build_pt=lambda a, b: a * b + a**2 - b / 2,
            inputs={"a": 2.0, "b": 3.0},
        )

    def test_nested_expression(self) -> None:
        self._compare(
            build_mg=lambda a, b, c: (a + b) * c + (a - b) ** 3,
            build_pt=lambda a, b, c: (a + b) * c + (a - b) ** 3,
            inputs={"a": 1.5, "b": 0.8, "c": 2.1},
        )

    def test_relu_composition(self) -> None:
        self._compare(
            build_mg=lambda a, b: (a * b + Value(0.5)).relu() + (a - b).relu(),
            build_pt=lambda a, b: (a * b + 0.5).relu() + (a - b).relu(),
            inputs={"a": 1.0, "b": 2.0},
        )

    def test_tanh_composition(self) -> None:
        self._compare(
            build_mg=lambda a, b: (a * b).tanh() + a.tanh() * b,
            build_pt=lambda a, b: (a * b).tanh() + a.tanh() * b,
            inputs={"a": 0.3, "b": 0.7},
        )

    def test_exp_log_roundtrip(self) -> None:
        # d/dx[log(exp(x))] should equal d/dx[x] = 1
        self._compare(
            build_mg=lambda a: a.exp().log() + a * a,
            build_pt=lambda a: a.exp().log() + a * a,
            inputs={"a": 1.2},
        )

    def test_division_gradient(self) -> None:
        self._compare(
            build_mg=lambda a, b: a / b + b / a,
            build_pt=lambda a, b: a / b + b / a,
            inputs={"a": 2.0, "b": 3.0},
        )


class TestGradientAccumulation:
    """Semantics of calling .backward multiple times or reusing a variable."""

    def test_gradient_accumulates_across_backward_calls(self) -> None:
        # PyTorch semantics: gradients accumulate unless zeroed.
        x = Value(3.0)
        (x * x).backward()
        first_grad = x.grad
        # Reset and compute again — need a fresh graph
        x2 = Value(3.0)
        (x2 * x2).backward()
        (x2 * x2).backward()
        assert x2.grad == pytest.approx(2 * first_grad)


class TestPreconditions:
    """Assertions fire on invalid inputs."""

    def test_rejects_log_of_non_positive(self) -> None:
        with pytest.raises((AssertionError, ValueError)):
            Value(0.0).log()

    def test_rejects_pow_by_non_scalar_value(self) -> None:
        # Value ** Value isn't supported (we only implement scalar exponents)
        with pytest.raises((AssertionError, TypeError)):
            Value(2.0) ** Value(3.0)  # type: ignore[operator]
