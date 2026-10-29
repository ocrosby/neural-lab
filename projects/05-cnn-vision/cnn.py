"""CNN for image classification — LeNet-style, ~2 conv layers + 2 fully-connected.

Architecture pattern (identical across LeNet-5, AlexNet, VGG, ResNet at the
"stem" level — differences are depth and residual connections):

    (N, C, H, W)
      │
      ▼
    Conv2d(C -> C1, kernel=3, padding=1) -> ReLU -> MaxPool2d(2)   # H, W halved
      │
      ▼
    Conv2d(C1 -> C2, kernel=3, padding=1) -> ReLU -> MaxPool2d(2)  # halved again
      │
      ▼
    Flatten
      │
      ▼
    Linear(C2 * (H/4) * (W/4) -> fc_features) -> ReLU
      │
      ▼
    Linear(fc_features -> n_classes)                                # logits

Why this beats an MLP on images: a Conv2d layer applies the SAME small filter
at every spatial position. The MLP learns a separate weight for every input
pixel. When the object of interest (a digit, a face, a cat) can appear at
different positions, the CNN's shared weights generalize with a fraction of
the data an MLP would need. This is the "translation equivariance" property
that makes convolutions the right prior for pixel data.

Padding=1 with kernel=3 preserves H/W through the conv; only MaxPool halves them.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


class CNN:
    """LeNet-style CNN with `len(conv_channels)` conv blocks and 2 FC layers."""

    def __init__(
        self,
        in_channels: int = 1,
        image_size: int = 28,
        n_classes: int = 10,
        conv_channels: list[int] | None = None,
        fc_features: int = 128,
        learning_rate: float = 0.001,
        seed: int = 42,
    ) -> None:
        assert in_channels > 0, f"in_channels must be positive, got {in_channels}"
        assert image_size > 0, f"image_size must be positive, got {image_size}"
        assert n_classes > 0, f"n_classes must be positive, got {n_classes}"
        assert learning_rate > 0, f"learning_rate must be positive, got {learning_rate}"

        conv_channels = conv_channels or [16, 32]
        assert len(conv_channels) >= 1, f"need at least one conv layer, got {conv_channels}"
        assert all(c > 0 for c in conv_channels), (
            f"every conv channel count must be positive, got {conv_channels}"
        )
        assert image_size % (2 ** len(conv_channels)) == 0, (
            f"image_size {image_size} must be divisible by 2^{len(conv_channels)} "
            f"= {2 ** len(conv_channels)} (one halving per pool)"
        )

        torch.manual_seed(seed)
        self.in_channels = in_channels
        self.image_size = image_size
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self._conv_channels = list(conv_channels)

        conv_layers: list[nn.Module] = []
        prev = in_channels
        for c in conv_channels:
            conv_layers.append(nn.Conv2d(prev, c, kernel_size=3, padding=1))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool2d(2))
            prev = c
        self.features = nn.Sequential(*conv_layers)

        flat_size = conv_channels[-1] * (image_size // (2 ** len(conv_channels))) ** 2
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, fc_features),
            nn.ReLU(),
            nn.Linear(fc_features, n_classes),
        )
        self.net = nn.Sequential(self.features, self.classifier)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.net.parameters())

    def first_conv_filters(self) -> np.ndarray:
        """Weights of the first Conv2d layer, shape (out_channels, in_channels, kH, kW)."""
        first_conv = self.features[0]
        assert isinstance(first_conv, nn.Conv2d)
        return first_conv.weight.detach().cpu().numpy()

    def fit(self, train_loader: DataLoader, n_epochs: int = 5) -> list[float]:
        assert n_epochs > 0, f"n_epochs must be positive, got {n_epochs}"
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)

        epoch_losses: list[float] = []
        self.net.train()
        for _epoch in range(n_epochs):
            total_loss = 0.0
            n_batches = 0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                logits = self.net(batch_x)
                loss = loss_fn(logits, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
                n_batches += 1
            epoch_losses.append(total_loss / max(1, n_batches))
        return epoch_losses

    def predict_proba(self, X: torch.Tensor) -> np.ndarray:
        self._check_input(X)
        self.net.eval()
        with torch.no_grad():
            logits = self.net(X)
            probs = torch.softmax(logits, dim=-1)
        return probs.detach().cpu().numpy()

    def predict(self, X: torch.Tensor) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1).astype(np.int64)

    def evaluate(self, loader: DataLoader) -> float:
        self.net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in loader:
                logits = self.net(batch_x)
                predictions = torch.argmax(logits, dim=-1)
                correct += int((predictions == batch_y).sum().item())
                total += int(batch_y.shape[0])
        return correct / max(1, total)

    def _check_input(self, X: torch.Tensor) -> None:
        assert X.ndim == 4, f"expected (N, C, H, W) tensor, got shape {tuple(X.shape)}"
        assert X.shape[1] == self.in_channels, (
            f"expected {self.in_channels} channels, got {X.shape[1]}"
        )
        assert X.shape[2] == self.image_size and X.shape[3] == self.image_size, (
            f"expected image size {self.image_size}x{self.image_size}, "
            f"got {X.shape[2]}x{X.shape[3]}"
        )
