"""PUCT and AlphaZero agent integration."""

from __future__ import annotations

import numpy as np
import torch

from main_street.agents import AlphaZeroAgentSpec, build
from main_street.core import GameSpec, GameState
from main_street.nn.checkpoint import CheckpointMeta, save_checkpoint
from main_street.nn.encode import EncoderConfig, build_encoder
from main_street.nn.mcts import puct_search, select_action, visit_distribution
from main_street.nn.models import build_model


def _untrained_pair():
    cfg = EncoderConfig(max_n=12, max_turns=6)
    encoder = build_encoder(cfg)
    params = {"channels": 16, "n_blocks": 2}
    model = build_model("simple_conv", encoder, params)
    return model, encoder, cfg, params


def test_puct_terminates_and_picks_legal():
    model, encoder, _cfg, _ = _untrained_pair()
    state = GameState.initial(GameSpec(n=6, schedule=(2, 3)))
    root = puct_search(state, model, encoder, n_simulations=32, c_puct=1.5)
    assert root.priors  # network expanded the root
    legal = {int(c) for c in np.flatnonzero(state.board == 0)}
    for a in root.children:
        assert a in legal
    assert any(c.visit_count > 0 for c in root.children.values())


def test_visit_distribution_normalized():
    model, encoder, _cfg, _ = _untrained_pair()
    state = GameState.initial(GameSpec(n=6, schedule=(2, 3)))
    root = puct_search(state, model, encoder, n_simulations=16, c_puct=1.5)
    dist = visit_distribution(root, temperature=1.0)
    assert abs(sum(dist.values()) - 1.0) < 1e-6


def test_alphazero_agent_via_checkpoint(tmp_path):
    model, _encoder, cfg, params = _untrained_pair()
    ckpt = tmp_path / "ckpt.pt"
    save_checkpoint(
        ckpt,
        model=model,
        model_name="simple_conv",
        model_params=params,
        encoder_config=cfg,
        meta=CheckpointMeta(run_id="test", iter=0),
    )

    spec = AlphaZeroAgentSpec(
        checkpoint_path=str(ckpt), n_simulations=8, c_puct=1.5, temperature=0.0
    )
    agent = build(spec)
    state = GameState.initial(GameSpec(n=6, schedule=(2, 3)))
    a = agent.act(state)
    assert 0 <= a < 6
    assert state.board[a] == 0


def test_puct_dirichlet_noise_doesnt_break():
    model, encoder, _cfg, _ = _untrained_pair()
    state = GameState.initial(GameSpec(n=6, schedule=(2, 3)))
    rng = np.random.default_rng(0)
    root = puct_search(
        state,
        model,
        encoder,
        n_simulations=8,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_eps=0.25,
        rng=rng,
    )
    assert abs(sum(root.priors.values()) - 1.0) < 1e-5


def test_select_action_temperature_zero_is_argmax():
    torch.manual_seed(0)
    model, encoder, _cfg, _ = _untrained_pair()
    state = GameState.initial(GameSpec(n=6, schedule=(2, 3)))
    root = puct_search(state, model, encoder, n_simulations=32, c_puct=1.5)
    a = select_action(root, temperature=0.0)
    best = max(root.children.items(), key=lambda kv: (kv[1].visit_count, kv[0]))[0]
    assert a == best
