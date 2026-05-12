"""Encoders: registry + base class + default implementation.

An encoder turns a list of `GameState`s into a dict of batched tensors. Each
encoder publishes named attributes (e.g. `board_channels`, `ctx_dim`) that a
paired model reads at construction to size its input layers.

By convention every encoder produces these keys so models can mask generically:
    legal_mask: (B, max_n) bool — True for cells legal to play now.
    valid_mask: (B, max_n) bool — True for cells that exist (i < state.spec.n).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Final

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field

from ..core import EMPTY, GameState, O, X

# Scales signed-mark entries in the ctx vector into a roughly unit range.
_MARKS_SCALE: Final[float] = 12.0


ENCODERS: dict[str, type[Encoder]] = {}


def register_encoder(name: str):
    def deco(cls: type[Encoder]) -> type[Encoder]:
        if name in ENCODERS:
            raise ValueError(f"encoder {name!r} already registered")
        ENCODERS[name] = cls
        return cls

    return deco


class EncoderConfig(BaseModel):
    """Selects a registered encoder and passes through its params. `max_n` and
    `max_turns` are universal padding limits; encoder-specific knobs live in
    `params`."""

    model_config = ConfigDict(frozen=True)

    name: str = "default"
    max_n: int = Field(default=24, gt=0)
    max_turns: int = Field(default=12, gt=0)
    params: dict[str, Any] = Field(default_factory=dict)


def build_encoder(cfg: EncoderConfig) -> Encoder:
    if cfg.name not in ENCODERS:
        raise KeyError(
            f"unknown encoder {cfg.name!r}; registered: {sorted(ENCODERS)}"
        )
    return ENCODERS[cfg.name](max_n=cfg.max_n, max_turns=cfg.max_turns, **cfg.params)


class Encoder(ABC):
    """All encoders pad to `max_n` / `max_turns` and produce `legal_mask` +
    `valid_mask` in their output dict. Each encoder also exposes the integer
    attributes (e.g. `board_channels`) that its paired model reads."""

    def __init__(self, max_n: int, max_turns: int) -> None:
        self.max_n = max_n
        self.max_turns = max_turns

    @abstractmethod
    def encode(self, states: list[GameState]) -> dict[str, torch.Tensor]:
        """Pad + canonicalize a batch of states. Output dict is the model's input."""

    def __call__(self, states: list[GameState]) -> dict[str, torch.Tensor]:
        return self.encode(states)


@register_encoder("default")
class DefaultEncoder(Encoder):
    """Mine/opp/empty/valid/position board planes + signed-marks schedule context.

    Canonicalization. Board is encoded from the current player's view: channel
    0 is "mine", channel 1 is "opp". Cell indices stay absolute since the
    game's rightmost tie-break is positional. Schedule context is expressed in
    the same frame — "my" future turns carry positive marks, opponent turns
    negative.

    Output shapes (B = batch):
        board      (B, 5, max_n)     float32 — mine, opp, empty, valid, position
        ctx        (B, ctx_dim)      float32 — schedule + turn state
        legal_mask (B, max_n)        bool
        valid_mask (B, max_n)        bool

    The ctx vector layout (length `ctx_dim = 2 + 2*max_turns`):
        [0]                        — fraction of this turn's marks still to place
        [1]                        — turn_idx / len(schedule)
        [2 .. 2+max_turns)         — signed marks per remaining turn slot
        [2+max_turns .. 2+2*mt)    — mask: 1 where the slot is a real turn
    """

    board_channels: Final[int] = 5

    def __init__(self, max_n: int, max_turns: int) -> None:
        super().__init__(max_n, max_turns)
        self.ctx_dim = 2 + 2 * max_turns

    def encode(self, states: list[GameState]) -> dict[str, torch.Tensor]:
        """Vectorized: gather per-state ints/arrays in a single pass, then all
        feature computation runs as bulk numpy ops on (B, ...) arrays. One
        torch.from_numpy at the end per tensor."""
        B = len(states)
        mn = self.max_n
        mt = self.max_turns

        boards_padded = np.zeros((B, mn), dtype=np.uint8)
        sched_padded = np.zeros((B, mt), dtype=np.int32)
        ns = np.empty(B, dtype=np.int32)
        sched_lens = np.empty(B, dtype=np.int32)
        turn_idxs = np.empty(B, dtype=np.int32)
        placements_left = np.empty(B, dtype=np.int32)
        mes = np.empty(B, dtype=np.uint8)

        for i, state in enumerate(states):
            n = state.spec.n
            L = len(state.spec.schedule)
            if n > mn:
                raise ValueError(f"state with n={n} exceeds encoder max_n={mn}")
            if L > mt:  # noqa: SIM300 — symmetric to the n > mn check above
                raise ValueError(
                    f"schedule length {L} exceeds encoder max_turns={mt}"
                )
            boards_padded[i, :n] = state.board
            sched_padded[i, :L] = state.spec.schedule
            ns[i] = n
            sched_lens[i] = L
            turn_idxs[i] = state.turn_idx
            placements_left[i] = state.placements_left
            mes[i] = int(state.current_player)

        opps = np.where(mes == int(X), int(O), int(X)).astype(np.uint8)
        cell_idx = np.arange(mn, dtype=np.int32)[None, :]
        valid = cell_idx < ns[:, None]  # (B, mn) bool
        mine = (boards_padded == mes[:, None]) & valid
        opp_m = (boards_padded == opps[:, None]) & valid
        empty = (boards_padded == EMPTY) & valid

        # Position channel: j / (n-1) over valid cells, 0.5 for n==1, 0 elsewhere.
        denom = np.where(ns > 1, ns - 1, 1).astype(np.float32)
        positions = (cell_idx.astype(np.float32) / denom[:, None]) * valid
        single = ns == 1
        if single.any():
            positions[single, 0] = 0.5

        board_np = np.zeros((B, self.board_channels, mn), dtype=np.float32)
        board_np[:, 0] = mine
        board_np[:, 1] = opp_m
        board_np[:, 2] = empty
        board_np[:, 3] = valid
        board_np[:, 4] = positions

        # Context. plc_frac depends on the (possibly past-end) current turn.
        in_range_now = turn_idxs < sched_lens
        cur_turn = np.where(in_range_now, turn_idxs, 0)
        cur_marks = sched_padded[np.arange(B), cur_turn]
        plc_frac = np.where(
            in_range_now,
            placements_left / np.maximum(1, cur_marks),
            0.0,
        ).astype(np.float32)

        # Per-slot remaining-schedule signed marks (j = 0 uses placements_left).
        j_range = np.arange(mt, dtype=np.int32)[None, :]
        abs_t = turn_idxs[:, None] + j_range
        in_range = abs_t < sched_lens[:, None]
        abs_t_clipped = np.where(in_range, abs_t, 0)
        marks = np.take_along_axis(sched_padded, abs_t_clipped, axis=1).astype(np.float32)
        marks[:, 0] = placements_left
        marks = np.where(in_range, marks, 0.0)
        sign = np.where(
            abs_t % 2 == turn_idxs[:, None] % 2, 1.0, -1.0
        ).astype(np.float32)

        ctx_np = np.zeros((B, self.ctx_dim), dtype=np.float32)
        ctx_np[:, 0] = plc_frac
        ctx_np[:, 1] = turn_idxs.astype(np.float32) / np.maximum(1, sched_lens)
        ctx_np[:, 2 : 2 + mt] = sign * marks / _MARKS_SCALE
        ctx_np[:, 2 + mt : 2 + 2 * mt] = in_range

        return {
            "board": torch.from_numpy(board_np),
            "ctx": torch.from_numpy(ctx_np),
            "legal_mask": torch.from_numpy(empty),
            "valid_mask": torch.from_numpy(valid),
        }
