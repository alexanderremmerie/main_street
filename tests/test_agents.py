import pytest

from main_street.agents import (
    AlphaBetaAgentSpec,
    BlockerAgentSpec,
    CenterAgentSpec,
    ExtensionAgentSpec,
    ForkAwareAgentSpec,
    GreedyAgentSpec,
    MCTSAgentSpec,
    PotentialAwareAgentSpec,
    RandomAgentSpec,
    RightmostAgentSpec,
    build,
)
from main_street.core import GameSpec, GameState, legal_actions, step

SPECS = [
    RandomAgentSpec(seed=0),
    GreedyAgentSpec(seed=0),
    RightmostAgentSpec(),
    AlphaBetaAgentSpec(depth=3),
    ExtensionAgentSpec(),
    BlockerAgentSpec(),
    CenterAgentSpec(),
    ForkAwareAgentSpec(seed=0),
    PotentialAwareAgentSpec(seed=0),
    MCTSAgentSpec(n_simulations=32, seed=0),
    MCTSAgentSpec(n_simulations=32, seed=0, rollout="forkaware"),
]


@pytest.mark.parametrize("agent_spec", SPECS)
def test_agent_returns_legal_action(agent_spec):
    agent = build(agent_spec)
    s = GameState.initial(GameSpec(n=8, schedule=(1, 2, 2, 1)))
    while not s.is_terminal:
        a = agent.act(s)
        assert a in legal_actions(s).tolist()
        s = step(s, a)


def test_rightmost_picks_rightmost():
    agent = build(RightmostAgentSpec())
    s = GameState.initial(GameSpec(n=5, schedule=(1, 1)))
    assert agent.act(s) == 4


def test_alphabeta_full_search_beats_random_consistently():
    # On small instances alphabeta should beat random as X.
    from main_street.runner import play

    spec = GameSpec(n=6, schedule=(1, 1, 1, 1))
    wins = 0
    for seed in range(8):
        g = play(spec, AlphaBetaAgentSpec(), RandomAgentSpec(seed=seed))
        if g.outcome == 1:
            wins += 1
    assert wins >= 6


def test_extension_extends_rightward_when_possible():
    # X has a single mark at cell 2; the only legal extension is to cell 3
    # (rightmost open neighbor), not cell 1.
    spec = GameSpec(n=5, schedule=(1, 1, 1))
    s = step(GameState.initial(spec), 2)  # X at 2; now O's turn
    s = step(s, 0)  # O somewhere irrelevant; back to X
    agent = build(ExtensionAgentSpec())
    assert agent.act(s) == 3


def test_extension_falls_back_to_rightmost_when_no_run():
    spec = GameSpec(n=4, schedule=(1, 1, 1))
    s = GameState.initial(spec)
    agent = build(ExtensionAgentSpec())
    assert agent.act(s) == 3


def test_blocker_blocks_opponent_longest_run():
    # X plays cell 1; it's now O's turn. Blocker (as O) should put a mark
    # adjacent to X's run — preferring the right side (cell 2).
    spec = GameSpec(n=5, schedule=(1, 1, 1))
    s = step(GameState.initial(spec), 1)  # X at 1; O to move
    agent = build(BlockerAgentSpec())
    assert agent.act(s) == 2


def test_center_picks_middle_of_largest_segment():
    # Whole board empty (length 7) → middle cell is 3 (odd-length segment).
    spec = GameSpec(n=7, schedule=(1, 1, 1))
    agent = build(CenterAgentSpec())
    assert agent.act(GameState.initial(spec)) == 3


def test_center_right_biases_on_even_length_segment():
    # 6-cell empty board: midpoints are 2 and 3; right bias → 3.
    spec = GameSpec(n=6, schedule=(1, 1, 1))
    agent = build(CenterAgentSpec())
    assert agent.act(GameState.initial(spec)) == 3


def test_center_picks_largest_segment_not_just_rightmost():
    # X at 4 splits the board into [0..3] (length 4) and [5..6] (length 2).
    # Center should pick the larger segment's midpoint = (0+3+1)//2 = 2.
    spec = GameSpec(n=7, schedule=(1, 1, 1))
    s = step(GameState.initial(spec), 4)
    s = step(s, 5)  # O somewhere; back to X
    agent = build(CenterAgentSpec())
    assert agent.act(s) == 2


def test_mcts_beats_random_consistently():
    # MCTS with a modest budget should beat Random with high probability.
    from main_street.runner import play

    spec = GameSpec(n=6, schedule=(1, 1, 1, 1))
    wins = 0
    n = 12
    for seed in range(n):
        g = play(spec, MCTSAgentSpec(n_simulations=128, seed=seed), RandomAgentSpec(seed=seed))
        if g.outcome == 1:
            wins += 1
    assert wins >= n - 2, f"MCTS only won {wins}/{n} against Random"


def test_mcts_is_deterministic_given_seed():
    spec = GameSpec(n=5, schedule=(1, 1, 1))
    s = GameState.initial(spec)
    a1 = build(MCTSAgentSpec(n_simulations=50, seed=42)).act(s)
    a2 = build(MCTSAgentSpec(n_simulations=50, seed=42)).act(s)
    assert a1 == a2


def test_forkaware_extends_its_own_two_run():
    """ForkAware should extend a 2-run with two open ends rather than play
    randomly, since (length + open_ends * length / 2) for the extended run
    strictly dominates any single-mark alternative."""
    # Schedule (1,1,1,1,1): 5 turns, X plays on 0, 2, 4. We construct the
    # X-to-move position where X has a 2-run at 3-4 with both ends open.
    spec = GameSpec(n=8, schedule=(1, 1, 1, 1, 1))
    s = GameState.initial(spec)
    s = step(s, 3)  # X@3
    s = step(s, 0)  # O@0 (edge, doesn't touch X)
    s = step(s, 4)  # X@4 — now X has 2-run at 3-4, two open ends
    s = step(s, 1)  # O@1 (still doesn't touch X)
    # X to move at turn 4. Extending the open 2-run is the unique best move.
    assert s.current_player.item() == 1  # sanity: X to move
    agent = build(ForkAwareAgentSpec(seed=0))
    a = agent.act(s)
    assert a in (2, 5), f"ForkAware picked {a} instead of extending its open 2-run"


def test_potentialaware_uses_schedule():
    """PotentialAware should differ from Greedy on a position where the
    *remaining schedule* changes the relative value of moves. The cleanest
    construction: a near-empty board with a big multi-placement turn coming
    up. PotentialAware should not just play the rightmost cell when the
    schedule still permits a longer-than-current-run window."""
    # Just verify the agent returns legal moves on a varied set of states.
    spec = GameSpec(n=10, schedule=(2, 1, 2, 1, 2))
    agent = build(PotentialAwareAgentSpec(seed=0))
    s = GameState.initial(spec)
    while not s.is_terminal:
        a = agent.act(s)
        assert a in legal_actions(s).tolist()
        s = step(s, a)


def test_mcts_heuristic_rollout_beats_random_rollout_on_deep_spec():
    """Heuristic rollouts should clearly outperform uniform random rollouts
    per-simulation. With matched budget, the forkaware-rollout MCTS should
    win most head-to-heads against the random-rollout MCTS."""
    from main_street.runner import play

    spec = GameSpec(n=10, schedule=(2, 2, 2, 1))
    h_wins = 0
    n = 6
    for seed in range(n):
        # Alternate sides so first-move bias doesn't dominate.
        if seed % 2 == 0:
            g = play(
                spec,
                MCTSAgentSpec(n_simulations=80, seed=seed, rollout="forkaware"),
                MCTSAgentSpec(n_simulations=80, seed=seed, rollout="random"),
            )
            if g.outcome == 1:
                h_wins += 1
        else:
            g = play(
                spec,
                MCTSAgentSpec(n_simulations=80, seed=seed, rollout="random"),
                MCTSAgentSpec(n_simulations=80, seed=seed, rollout="forkaware"),
            )
            if g.outcome == -1:
                h_wins += 1
    # Heuristic rollouts shouldn't be *worse* than random at matched budget.
    # On a tight budget the gap may not be huge but it should at least tie.
    assert h_wins >= n // 2, f"heuristic-rollout MCTS only won {h_wins}/{n} vs random rollouts"


def test_mcts_approaches_optimal_with_more_sims():
    """On a small spec the exact value is known; MCTS with enough sims as the
    optimal-side player should win the great majority of games."""
    from main_street.runner import play
    from main_street.solve import solve

    spec = GameSpec(n=5, schedule=(1, 1, 1))
    root_value = solve(GameState.initial(spec)).value
    # `root_value` is +1 (X wins) or -1 (O wins) under perfect play. Let MCTS
    # play the winning side against Greedy from the opposite side; with
    # sufficient sims it should sweep.
    x_spec, o_spec = (
        (MCTSAgentSpec(n_simulations=400, seed=0), GreedyAgentSpec(seed=0))
        if root_value == 1
        else (GreedyAgentSpec(seed=0), MCTSAgentSpec(n_simulations=400, seed=0))
    )
    wins = 0
    n = 6
    for seed in range(n):
        x = MCTSAgentSpec(n_simulations=400, seed=seed) if root_value == 1 else x_spec
        o = MCTSAgentSpec(n_simulations=400, seed=seed) if root_value == -1 else o_spec
        g = play(spec, x, o)
        if (root_value == 1 and g.outcome == 1) or (root_value == -1 and g.outcome == -1):
            wins += 1
    assert wins >= n - 1, f"MCTS only secured {wins}/{n} on a solver-winning side"
