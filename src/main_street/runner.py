"""Run games. `play` plays a full bot-vs-bot match; `record_from_actions`
builds a persisted record from a client-supplied action sequence (used when
humans are involved); `run_tournament` plays many bot-vs-bot games and persists.
"""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Iterable
from itertools import combinations

from .agents import AgentSpec, build
from .core import GameSpec, GameState, X, final_state, outcome, step
from .records import (
    ComparisonConfig,
    ComparisonRecord,
    ComparisonSummary,
    EvalConfig,
    EvalRecord,
    EvalSummary,
    GameRecord,
    PairResult,
)
from .store import (
    get_player,
    insert_comparison,
    insert_eval,
    insert_game,
    update_comparison,
    update_eval,
)


def _new_id() -> str:
    return uuid.uuid4().hex


def record_from_actions(
    spec: GameSpec,
    x_agent: AgentSpec,
    o_agent: AgentSpec,
    actions: Iterable[int],
    *,
    eval_id: str | None = None,
    seed: int | None = None,
) -> GameRecord:
    """Validate and replay an externally-played action sequence, returning the
    record (not persisted). Raises ValueError if any action is illegal or the
    final state is not terminal.
    """
    seq = tuple(int(a) for a in actions)
    state = final_state(spec, seq)
    if not state.is_terminal:
        done = state.turn_idx
        total = len(spec.schedule)
        raise ValueError(
            f"action sequence does not complete the game ({done}/{total} turns done)"
        )
    return GameRecord(
        id=_new_id(),
        spec=spec,
        x_agent=x_agent,
        o_agent=o_agent,
        actions=seq,
        outcome=outcome(state),
        eval_id=eval_id,
        seed=seed,
    )


def play(
    spec: GameSpec,
    x_agent: AgentSpec,
    o_agent: AgentSpec,
    *,
    eval_id: str | None = None,
    seed: int | None = None,
) -> GameRecord:
    """Play a full bot-vs-bot game and return its record (not persisted).
    Both agents must be buildable on the server (no human).
    """
    x = build(x_agent)
    o = build(o_agent)
    state = GameState.initial(spec)
    actions: list[int] = []
    while not state.is_terminal:
        agent = x if state.current_player == X else o
        cell = int(agent.act(state))
        actions.append(cell)
        state = step(state, cell)
    return GameRecord(
        id=_new_id(),
        spec=spec,
        x_agent=x_agent,
        o_agent=o_agent,
        actions=tuple(actions),
        outcome=outcome(state),
        eval_id=eval_id,
        seed=seed,
    )


def run_comparison(
    conn: sqlite3.Connection, config: ComparisonConfig
) -> ComparisonRecord:
    """Run an N-player × M-spec comparison.

    Each unordered pair `(P_i, P_j)` becomes one `EvalConfig` covering every
    spec in `config.specs`. The underlying tournaments are persisted as
    ordinary `evals` rows; this function only records the comparison's
    metadata and references their IDs.

    Player resolution happens up front so a missing player ID fails the whole
    comparison immediately rather than mid-run.
    """
    if len(set(config.player_ids)) != len(config.player_ids):
        raise ValueError("player_ids must be unique within a comparison")

    resolved: dict[str, AgentSpec] = {}
    for pid in config.player_ids:
        player = get_player(conn, pid)
        if player is None:
            raise ValueError(f"player {pid!r} not found")
        resolved[pid] = player.agent_spec

    comparison_id = _new_id()
    record = ComparisonRecord(id=comparison_id, config=config, status="running")
    with conn:
        insert_comparison(conn, record)

    pairs: list[PairResult] = []
    n_total = 0
    try:
        # Stable, deterministic pair ordering matches what the UI will render.
        # Per-pair seed is offset so re-running with the same seed reproduces
        # exact game sequences.
        for idx, (a_id, b_id) in enumerate(combinations(config.player_ids, 2)):
            eval_cfg = EvalConfig(
                agent_a=resolved[a_id],
                agent_b=resolved[b_id],
                specs=config.specs,
                n_games_per_spec=config.n_games_per_spec,
                swap_sides=config.swap_sides,
                seed=config.seed + idx * 10_000,
            )
            ev = run_tournament(conn, eval_cfg)
            assert ev.summary is not None  # run_tournament raises rather than return summary=None
            pairs.append(
                PairResult(
                    a_player_id=a_id,
                    b_player_id=b_id,
                    eval_id=ev.id,
                    a_winrate=ev.summary.a_winrate,
                    n_games=ev.summary.n_games,
                )
            )
            n_total += ev.summary.n_games

        summary = ComparisonSummary(pairs=tuple(pairs), n_total_games=n_total)
        with conn:
            update_comparison(conn, comparison_id, "done", summary)
        return ComparisonRecord(
            id=comparison_id, config=config, status="done", summary=summary
        )
    except Exception:
        with conn:
            update_comparison(conn, comparison_id, "failed", None)
        raise


def run_tournament(conn: sqlite3.Connection, config: EvalConfig) -> EvalRecord:
    eval_id = _new_id()
    record = EvalRecord(id=eval_id, config=config, status="running")
    with conn:
        insert_eval(conn, record)

    a_wins = b_wins = ties = 0
    n = 0
    try:
        for spec in config.specs:
            for i in range(config.n_games_per_spec):
                a_is_x = not (config.swap_sides and i % 2 == 1)
                x_agent = config.agent_a if a_is_x else config.agent_b
                o_agent = config.agent_b if a_is_x else config.agent_a
                game = play(spec, x_agent, o_agent, eval_id=eval_id, seed=config.seed + n)
                with conn:
                    insert_game(conn, game)
                a_score = game.outcome if a_is_x else -game.outcome
                if a_score > 0:
                    a_wins += 1
                elif a_score < 0:
                    b_wins += 1
                else:
                    ties += 1
                n += 1

        summary = EvalSummary(
            n_games=n,
            a_wins=a_wins,
            b_wins=b_wins,
            ties=ties,
            a_winrate=(a_wins + 0.5 * ties) / max(n, 1),
        )
        with conn:
            update_eval(conn, eval_id, "done", summary)
        return EvalRecord(id=eval_id, config=config, status="done", summary=summary)
    except Exception:
        with conn:
            update_eval(conn, eval_id, "failed", None)
        raise
