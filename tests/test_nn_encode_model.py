"""Smoke tests for the encoder registry + SimpleConv."""

from __future__ import annotations

import torch

from main_street.core import GameSpec, GameState, step
from main_street.nn.encode import ENCODERS, EncoderConfig, build_encoder
from main_street.nn.models import MODELS, build_model


def _states_batch() -> list[GameState]:
    a = GameState.initial(GameSpec(n=5, schedule=(2, 3)))
    b = step(GameState.initial(GameSpec(n=8, schedule=(3, 4))), 7)
    c = step(step(GameState.initial(GameSpec(n=10, schedule=(3, 3, 3))), 9), 0)
    return [a, b, c]


def test_default_encoder_attributes_and_output_keys():
    assert "default" in ENCODERS
    cfg = EncoderConfig(name="default", max_n=12, max_turns=6)
    enc = build_encoder(cfg)
    assert enc.max_n == 12
    assert enc.max_turns == 6
    assert enc.board_channels == 5
    assert enc.ctx_dim == 2 + 2 * 6

    out = enc(_states_batch())
    assert set(out.keys()) == {"board", "ctx", "legal_mask", "valid_mask"}
    assert out["board"].shape == (3, enc.board_channels, enc.max_n)
    assert out["ctx"].shape == (3, enc.ctx_dim)
    assert out["legal_mask"].shape == (3, enc.max_n)
    assert out["valid_mask"].shape == (3, enc.max_n)


def test_valid_and_legal_masks_correct():
    enc = build_encoder(EncoderConfig(max_n=12, max_turns=6))
    states = _states_batch()
    out = enc(states)
    for i, s in enumerate(states):
        n = s.spec.n
        assert torch.all(out["valid_mask"][i, :n])
        assert not torch.any(out["valid_mask"][i, n:])
        for j in range(n):
            assert bool(out["legal_mask"][i, j]) == (s.board[j] == 0)


def test_model_forward_shapes_and_masking():
    assert "simple_conv" in MODELS
    enc = build_encoder(EncoderConfig(max_n=12, max_turns=6))
    model = build_model("simple_conv", enc, params={"channels": 16, "n_blocks": 2})
    out = enc(_states_batch())
    logits, value = model(out)
    assert logits.shape == (3, enc.max_n)
    assert value.shape == (3,)
    assert torch.all(logits[~out["legal_mask"]] < -1e6)
    assert torch.all(value.abs() <= 1.0)
