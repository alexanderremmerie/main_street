"""End-to-end smoke test: tiny training run produces a usable checkpoint.

We don't assert on quality — that's a function of compute. We assert the
pipeline doesn't crash, a checkpoint lands on disk, and the resulting
`AlphaZeroAgentSpec` builds and plays a legal move.
"""

from __future__ import annotations

from pathlib import Path

from main_street.agents import AlphaZeroAgentSpec, build
from main_street.core import GameSpec, GameState
from main_street.eval.positions import (
    SourceSpec,
    build_position_set,
)
from main_street.nn.encode import EncoderConfig
from main_street.nn.sources import SelfPlaySourceConfig, SupervisedSourceConfig
from main_street.nn.train import (
    DataConfig,
    LoopConfig,
    ModelConfig,
    TrainConfig,
    Trainer,
    WandbConfig,
)


def test_trainer_runs_end_to_end(tmp_path, monkeypatch):
    # Build minimum eval sets in a temp dir to keep the test self-contained
    # and avoid mutating the project's data dir.
    eval_root = tmp_path / "eval"
    sources_sup = [
        SourceSpec(spec=GameSpec(n=5, schedule=(2, 3)), mode="all_reachable"),
        SourceSpec(spec=GameSpec(n=6, schedule=(1, 2, 2)), mode="all_reachable"),
    ]
    ps_sup = build_position_set("train_small", sources_sup)
    ps_sup.save(root=eval_root)
    ps_starter = build_position_set(
        "starter",
        [SourceSpec(spec=GameSpec(n=5, schedule=(2, 3)), mode="all_reachable")],
    )
    ps_starter.save(root=eval_root)
    ps_holdout = build_position_set(
        "holdout",
        [SourceSpec(spec=GameSpec(n=6, schedule=(1, 2, 2)), mode="all_reachable")],
    )
    ps_holdout.save(root=eval_root)
    ps_diag = build_position_set(
        "diagnostics",
        [
            SourceSpec(
                spec=GameSpec(n=5, schedule=(2, 3)),
                mode="initial_only",
                label="diag",
            )
        ],
    )
    ps_diag.save(root=eval_root)

    # Patch DEFAULT_ROOT so the trainer's `PositionSet.load` finds them.
    monkeypatch.setattr("main_street.eval.positions.DEFAULT_ROOT", eval_root)

    # Run output goes under the project's `data/` by default; redirect with chdir.
    monkeypatch.chdir(tmp_path)

    cfg = TrainConfig(
        name="t",
        seed=0,
        model=ModelConfig(name="simple_conv", params={"channels": 8, "n_blocks": 1}),
        encoder=EncoderConfig(max_n=6, max_turns=3),
        data=DataConfig(
            sources=[
                SupervisedSourceConfig(weight=0.7, set="train_small"),
                SelfPlaySourceConfig(
                    weight=0.3,
                    specs=[(5, [2, 3]), (6, [1, 2, 2])],
                    games_per_iter=2,
                    n_simulations=4,
                    temperature_moves=2,
                    capacity=200,
                ),
            ]
        ),
        loop=LoopConfig(
            iters=1,
            steps_per_iter=3,
            batch_size=16,
            eval_every=1,
            eval_n_simulations=4,
        ),
        wandb=WandbConfig(mode="disabled"),
    )
    final = Trainer(cfg).run()
    assert Path(final).exists()

    spec = AlphaZeroAgentSpec(
        checkpoint_path=str(final), n_simulations=4, c_puct=1.5, temperature=0.0
    )
    agent = build(spec)
    state = GameState.initial(GameSpec(n=5, schedule=(2, 3)))
    a = agent.act(state)
    assert 0 <= a < 5
    assert state.board[a] == 0  # legal cell
