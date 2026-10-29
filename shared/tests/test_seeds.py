"""Tests that seeding actually produces reproducible sequences.

Black-box: assert on the observable outcome (two runs after `set_seed(N)`
produce identical sequences), not on internal RNG state.
"""

from __future__ import annotations

import random

import numpy as np

from shared.seeds import set_seed


def test_python_random_reproducible() -> None:
    set_seed(42)
    first = [random.random() for _ in range(5)]
    set_seed(42)
    second = [random.random() for _ in range(5)]
    assert first == second


def test_numpy_random_reproducible() -> None:
    set_seed(42)
    first = np.random.rand(10)
    set_seed(42)
    second = np.random.rand(10)
    np.testing.assert_array_equal(first, second)


def test_different_seeds_produce_different_sequences() -> None:
    set_seed(42)
    a = np.random.rand(10)
    set_seed(43)
    b = np.random.rand(10)
    assert not np.array_equal(a, b)
