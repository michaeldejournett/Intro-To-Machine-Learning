"""Two MLP classifiers (shallow vs deeper + batch norm)."""

from __future__ import annotations

import torch
from torch import nn


class ShallowMLP(nn.Module):
    """2 hidden layers; fewer params than the deep model."""

    def __init__(self, n_features: int, n_classes: int, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepBatchNormMLP(nn.Module):
    """More layers + batch norm (see widths in code)."""

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        dims: tuple[int, ...] = (256, 128, 64, 32),
        dropout: float = 0.25,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_features
        for i, d in enumerate(dims):
            layers.append(nn.Linear(prev, d))
            layers.append(nn.BatchNorm1d(d))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout if i < len(dims) - 1 else dropout * 0.5))
            prev = d
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
