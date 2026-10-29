"""Black-box tests for the NN primitives built on the autograd engine.

If these pass, the whole abstraction is validated end-to-end: Value ops back-
propagate gradients through Module/Neuron/Layer/MLP correctly, and the
resulting parameters actually learn XOR (which the perceptron couldn't).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_PROJECT_DIR = Path(__file__).resolve().parents[1]
_ENGINE_KEY = "micrograd_engine"
_NN_KEY = "micrograd_nn"

if _ENGINE_KEY in sys.modules:
    engine_mod = sys.modules[_ENGINE_KEY]
else:
    _engine_spec = importlib.util.spec_from_file_location(_ENGINE_KEY, _PROJECT_DIR / "engine.py")
    assert _engine_spec is not None and _engine_spec.loader is not None
    engine_mod = importlib.util.module_from_spec(_engine_spec)
    sys.modules[_ENGINE_KEY] = engine_mod
    _engine_spec.loader.exec_module(engine_mod)

if _NN_KEY in sys.modules:
    nn_mod = sys.modules[_NN_KEY]
else:
    _nn_spec = importlib.util.spec_from_file_location(_NN_KEY, _PROJECT_DIR / "nn.py")
    assert _nn_spec is not None and _nn_spec.loader is not None
    nn_mod = importlib.util.module_from_spec(_nn_spec)
    sys.modules[_NN_KEY] = nn_mod
    _nn_spec.loader.exec_module(nn_mod)

Value = engine_mod.Value
Neuron = nn_mod.Neuron
Layer = nn_mod.Layer
MLP = nn_mod.MLP


class TestNeuron:
    """A single neuron is n_in weights + one bias, applied to an input vector."""

    def test_neuron_produces_value_output(self) -> None:
        n = Neuron(n_in=3, seed=0)
        y = n([Value(1.0), Value(2.0), Value(3.0)])
        assert isinstance(y, Value)

    def test_neuron_parameters_count(self) -> None:
        # A neuron with n_in inputs has n_in weights + 1 bias.
        n = Neuron(n_in=5, seed=0)
        assert len(n.parameters()) == 6

    def test_nonlin_neuron_output_nonnegative(self) -> None:
        # ReLU neuron: output is always >= 0
        n = Neuron(n_in=3, nonlin=True, seed=0)
        y = n([Value(-10.0), Value(-10.0), Value(-10.0)])
        assert y.data >= 0.0


class TestLayer:
    def test_layer_output_shape(self) -> None:
        layer = Layer(n_in=3, n_out=4, seed=0)
        out = layer([Value(1.0), Value(1.0), Value(1.0)])
        assert isinstance(out, list)
        assert len(out) == 4
        assert all(isinstance(o, Value) for o in out)

    def test_layer_parameters_count(self) -> None:
        # 4 neurons, each with 3 weights + 1 bias = 16 params
        layer = Layer(n_in=3, n_out=4, seed=0)
        assert len(layer.parameters()) == 16


class TestMLP:
    def test_mlp_forward_shape_single_output(self) -> None:
        mlp = MLP(n_in=2, n_outs=[8, 1], seed=0)
        y = mlp([Value(0.5), Value(-0.5)])
        assert isinstance(y, Value)

    def test_mlp_forward_shape_multiple_outputs(self) -> None:
        mlp = MLP(n_in=2, n_outs=[8, 3], seed=0)
        y = mlp([Value(0.5), Value(-0.5)])
        assert isinstance(y, list)
        assert len(y) == 3

    def test_mlp_parameters_aggregates_all_layers(self) -> None:
        # Layer(2, 8): 8 * (2+1) = 24
        # Layer(8, 1): 1 * (8+1) = 9
        # Total: 33
        mlp = MLP(n_in=2, n_outs=[8, 1], seed=0)
        assert len(mlp.parameters()) == 33

    def test_zero_grad_resets_all_parameters(self) -> None:
        mlp = MLP(n_in=2, n_outs=[8, 1], seed=0)
        for p in mlp.parameters():
            p.grad = 5.0
        mlp.zero_grad()
        assert all(p.grad == 0.0 for p in mlp.parameters())

    def test_mlp_trains_on_xor(self) -> None:
        """The whole point: micrograd is enough to learn XOR."""
        X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
        y_targets = [-1.0, 1.0, 1.0, -1.0]

        mlp = MLP(n_in=2, n_outs=[16, 16, 1], seed=1)
        learning_rate = 0.05
        for _step in range(500):
            preds = [mlp([Value(xi) for xi in x]) for x in X]
            loss = sum(((p - t) ** 2 for p, t in zip(preds, y_targets, strict=True)), Value(0.0))
            mlp.zero_grad()
            loss.backward()
            for p in mlp.parameters():
                p.data -= learning_rate * p.grad

        final_preds = [
            mlp([Value(xi) for xi in x]).data  # type: ignore[union-attr]
            for x in X
        ]
        correct = sum(
            1 for pred, target in zip(final_preds, y_targets, strict=True) if pred * target > 0
        )
        assert correct == 4, f"MLP should solve XOR, got {correct}/4 correct: {final_preds}"


class TestPreconditions:
    def test_neuron_rejects_wrong_input_length(self) -> None:
        n = Neuron(n_in=3, seed=0)
        with pytest.raises(AssertionError, match="expected 3"):
            n([Value(1.0), Value(2.0)])  # only 2 inputs

    def test_neuron_rejects_non_positive_n_in(self) -> None:
        with pytest.raises(AssertionError, match="positive"):
            Neuron(n_in=0)

    def test_mlp_rejects_empty_layer_list(self) -> None:
        with pytest.raises(AssertionError, match="at least one"):
            MLP(n_in=2, n_outs=[])
