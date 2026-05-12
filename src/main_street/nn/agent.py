"""Runtime AlphaZero agent: loads a checkpoint, plays via PUCT.

Kept out of `agents.py` so importing the agent registry doesn't pull torch
along for callers that only want classical baselines.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core import GameState
from .checkpoint import load_checkpoint
from .mcts import puct_search, select_action


class _AlphaZero:
    __slots__ = ("_model", "_encoder", "_n_sims", "_c_puct", "_temperature", "_rng")

    def __init__(
        self,
        checkpoint_path: Path,
        n_simulations: int,
        c_puct: float,
        temperature: float,
        seed: int | None,
    ) -> None:
        model, encoder, _meta = load_checkpoint(checkpoint_path)
        self._model = model
        self._encoder = encoder
        self._n_sims = n_simulations
        self._c_puct = c_puct
        self._temperature = temperature
        self._rng = np.random.default_rng(seed)

    def act(self, state: GameState) -> int:
        root = puct_search(
            root_state=state,
            model=self._model,
            encoder=self._encoder,
            n_simulations=self._n_sims,
            c_puct=self._c_puct,
        )
        return select_action(root, temperature=self._temperature, rng=self._rng)


def build_alphazero(
    checkpoint_path: str | Path,
    n_simulations: int,
    c_puct: float,
    temperature: float,
    seed: int | None,
) -> _AlphaZero:
    return _AlphaZero(
        checkpoint_path=Path(checkpoint_path),
        n_simulations=n_simulations,
        c_puct=c_puct,
        temperature=temperature,
        seed=seed,
    )
