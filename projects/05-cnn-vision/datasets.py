"""Synthetic image datasets for fast, offline CNN tests.

The `quadrant` task: each image has a small bright square in one of `n_classes`
positions (top-left, top-right, bottom-left, bottom-right for 4 classes) plus
Gaussian noise. Structurally trivial for a CNN — a single 3x3 conv filter can
localize a bright patch — and impossible for a flat MLP to solve at the same
sample efficiency, because the MLP has to learn the pixel-position → class
mapping from scratch for every pixel independently.

Uses stdlib + torch only, no torchvision, no network I/O.
"""

from __future__ import annotations

import torch


def make_quadrant_dataset(
    n_per_class: int,
    n_classes: int = 4,
    image_size: int = 28,
    in_channels: int = 1,
    noise_std: float = 0.1,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bright-square-in-quadrant dataset.

    Args:
        n_per_class: Number of samples per class.
        n_classes: 2 (top / bottom), 4 (quadrants), or higher (grid positions).
        image_size: H = W = this value.
        in_channels: 1 for grayscale, 3 for pseudo-RGB (bright patch replicated).
        noise_std: Gaussian noise added on top of the signal. Higher = harder.
        seed: RNG seed.

    Returns:
        X: shape (n_per_class * n_classes, in_channels, image_size, image_size).
        y: shape (n_per_class * n_classes,), integer class labels.
    """
    assert n_per_class > 0, f"n_per_class must be positive, got {n_per_class}"
    assert n_classes >= 2, f"n_classes must be >= 2, got {n_classes}"
    assert image_size >= 8, f"image_size must be >= 8, got {image_size}"
    assert in_channels in (1, 3), f"in_channels must be 1 or 3, got {in_channels}"

    generator = torch.Generator().manual_seed(seed)
    n_total = n_per_class * n_classes
    X = torch.randn(n_total, in_channels, image_size, image_size, generator=generator) * noise_std
    y = torch.zeros(n_total, dtype=torch.long)

    grid = _grid_size(n_classes)
    patch_size = max(2, image_size // (grid * 2))

    for cls in range(n_classes):
        row = cls // grid
        col = cls % grid
        cell_h = image_size // grid
        cell_w = image_size // grid
        center_r = row * cell_h + cell_h // 2
        center_c = col * cell_w + cell_w // 2
        half = patch_size // 2
        r_slice = slice(max(0, center_r - half), min(image_size, center_r + half))
        c_slice = slice(max(0, center_c - half), min(image_size, center_c + half))

        start = cls * n_per_class
        end = start + n_per_class
        X[start:end, :, r_slice, c_slice] += 1.0
        y[start:end] = cls

    perm = torch.randperm(n_total, generator=generator)
    return X[perm], y[perm]


def _grid_size(n_classes: int) -> int:
    """Smallest integer g such that g*g >= n_classes."""
    g = 1
    while g * g < n_classes:
        g += 1
    return g
