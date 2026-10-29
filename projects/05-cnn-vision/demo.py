"""Demo: train the CNN on real MNIST.

Downloads MNIST to `./data/` on first run (~11 MB). Trains for a few epochs
and prints train/test accuracy per epoch. Saves the learned first-conv-layer
filters to `first_conv_filters.png` — usually you can see clear edge / corner
detectors emerge from the noise.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

_HERE = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("cnn_mod", _HERE / "cnn.py")
assert _spec is not None and _spec.loader is not None
cnn_mod = importlib.util.module_from_spec(_spec)
sys.modules["cnn_mod"] = cnn_mod
_spec.loader.exec_module(cnn_mod)
CNN = cnn_mod.CNN


def main() -> None:
    sys.path.insert(0, str(_HERE.parents[1]))
    from shared.seeds import set_seed

    set_seed(0)
    torch.set_num_threads(4)

    transform = transforms.Compose([transforms.ToTensor()])
    data_dir = _HERE.parents[1] / "data"
    print(f"Loading MNIST from {data_dir}...")
    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    print(f"  train: {len(train_ds)}, test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    print("\n=== CNN on MNIST ===")
    model = CNN(
        in_channels=1,
        image_size=28,
        n_classes=10,
        conv_channels=[16, 32],
        fc_features=128,
        learning_rate=0.001,
        seed=0,
    )
    print(f"parameters: {model.n_parameters():,}")

    n_epochs = 3
    for epoch in range(1, n_epochs + 1):
        losses = model.fit(train_loader, n_epochs=1)
        test_acc = model.evaluate(test_loader)
        print(f"epoch {epoch}: train loss = {losses[0]:.4f}, test accuracy = {test_acc:.2%}")

    filters = model.first_conv_filters()  # (16, 1, 3, 3)
    print(f"\nfirst-conv filter shape: {filters.shape}")
    _save_filters(filters, _HERE / "first_conv_filters.png")
    print(f"filters saved to {_HERE / 'first_conv_filters.png'}")


def _save_filters(filters: np.ndarray, path: Path) -> None:
    out_c = filters.shape[0]
    cols = 8
    rows = (out_c + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))
    for i, ax in enumerate(axes.flat):
        if i < out_c:
            f = filters[i, 0]
            ax.imshow(f, cmap="gray")
            ax.set_title(f"#{i}", fontsize=8)
        ax.axis("off")
    fig.suptitle("First-layer conv filters", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
