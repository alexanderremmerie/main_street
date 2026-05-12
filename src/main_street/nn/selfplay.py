"""One self-play game using PUCT.

If `opponent` is None, the current model plays both sides and every move is
recorded as a training sample (the textbook AlphaZero setup). If `opponent`
is provided, only moves made on `learner_side` are recorded; the opponent
plays its turns silently. This is how "self-play against alphabeta",
"self-play against a frozen prior checkpoint", etc. are expressed.

Move-selection temperature: 1 for the first `temperature_moves` plies
(exploratory), 0 (greedy) afterwards. Dirichlet noise at the root.
"""

from __future__ import annotations

import numpy as np

from ..agents import Agent
from ..core import GameSpec, GameState, outcome, step
from .buffer import Sample
from .encode import Encoder
from .mcts import puct_search, select_action, visit_distribution
from .models import Model


def self_play_game(
    spec: GameSpec,
    model: Model,
    encoder: Encoder,
    n_simulations: int,
    c_puct: float,
    dirichlet_alpha: float,
    dirichlet_eps: float,
    temperature_moves: int,
    rng: np.random.Generator,
    opponent: Agent | None = None,
    learner_side: int | None = None,
) -> list[Sample]:
    if opponent is not None and learner_side is None:
        raise ValueError("opponent requires learner_side to identify whose moves to record")

    state = GameState.initial(spec)
    visited: list[tuple[GameState, dict[int, float]]] = []
    move_idx = 0
    while not state.is_terminal:
        is_learner_turn = opponent is None or int(state.current_player) == learner_side
        if is_learner_turn:
            root = puct_search(
                root_state=state,
                model=model,
                encoder=encoder,
                n_simulations=n_simulations,
                c_puct=c_puct,
                dirichlet_alpha=dirichlet_alpha,
                dirichlet_eps=dirichlet_eps,
                rng=rng,
            )
            train_pi = visit_distribution(root, temperature=1.0)
            visited.append((state, train_pi))
            temp = 1.0 if move_idx < temperature_moves else 0.0
            action = select_action(root, temperature=temp, rng=rng)
        else:
            action = int(opponent.act(state))
        state = step(state, action)
        move_idx += 1

    z_x = float(outcome(state))
    return [Sample(state=s, pi=p, z=z_x) for s, p in visited]
