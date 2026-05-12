"""Model interface + registry + the baseline architecture.

A `Model` consumes the dict produced by some `Encoder` and returns
`(policy_logits, value)`. Each implementation reads whatever keys it needs
from `inputs`; mismatched encoder/model pairs fail loudly with KeyError on
the first forward.
"""

from __future__ import annotations

from typing import Final

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encode import Encoder

MODELS: dict[str, type[Model]] = {}

_NEG_INF: Final[float] = -1e9


def register_model(name: str):
    def deco(cls: type[Model]) -> type[Model]:
        if name in MODELS:
            raise ValueError(f"model {name!r} already registered")
        MODELS[name] = cls
        return cls

    return deco


def build_model(name: str, encoder: Encoder, params: dict) -> Model:
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; registered: {sorted(MODELS)}")
    return MODELS[name](encoder=encoder, **params)


class Model(nn.Module):
    """`forward(inputs)` reads tensors by key from `inputs` and returns
    `(policy_logits, value)`.

    `policy_logits`: (B, max_n) with illegal cells filled with a large
    negative, so a downstream `softmax` zeros them.
    `value`: (B,) in [-1, 1], from the *current player's* perspective.
    """

    def forward(self, inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


# ---------- SimpleConv -------------------------------------------------------


class _ResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        # GroupNorm avoids the zero-padded-cell skew that BatchNorm hits on
        # this variable-width input.
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.relu(x + h)


@register_model("simple_conv")
class SimpleConv(Model):
    """Residual stack of 1D conv blocks. Schedule context becomes an additive
    per-channel bias on the conv features. Value head averages over valid
    cells and runs through a small MLP.

    Reads `board`, `ctx`, `legal_mask`, `valid_mask` from `inputs`."""

    def __init__(
        self,
        encoder: Encoder,
        channels: int = 64,
        n_blocks: int = 4,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.stem = nn.Conv1d(
            encoder.board_channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.stem_norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.ctx_proj = nn.Sequential(
            nn.Linear(encoder.ctx_dim, channels),
            nn.ReLU(),
            nn.Linear(channels, channels),
        )
        self.blocks = nn.ModuleList(
            [_ResBlock(channels, kernel_size) for _ in range(n_blocks)]
        )
        self.policy_head = nn.Conv1d(channels, 1, kernel_size=1)
        self.value_head = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, 1),
        )

    def forward(
        self, inputs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        board = inputs["board"]
        legal_mask = inputs["legal_mask"]
        valid_mask = inputs["valid_mask"].to(board.dtype)

        x = F.relu(self.stem_norm(self.stem(board)))
        x = x + self.ctx_proj(inputs["ctx"]).unsqueeze(-1)
        for block in self.blocks:
            x = block(x)

        logits = self.policy_head(x).squeeze(1).masked_fill(~legal_mask, _NEG_INF)

        valid_sum = valid_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        pooled = (x * valid_mask.unsqueeze(1)).sum(dim=-1) / valid_sum
        value = torch.tanh(self.value_head(pooled).squeeze(-1))

        return logits, value
