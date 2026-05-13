"""Agents: serializable specs, runtime protocol, and the classical baselines.

`AgentSpec` is the JSON-friendly description; `build(spec)` returns a live
`Agent`. Adding a new baseline is a `Spec` class, a runtime class, and one
`build()` branch. Each baseline's behavior is documented on its runtime class.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Annotated, Literal, Protocol, TypeVar, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .core import EMPTY, GameState, O, X, legal_actions, longest_run, outcome, step
from .llm import (
    append_trace,
    board_to_string,
    parse_json_cell_reply,
    render_prompt,
    request_chat_completion,
)
from .solve import search_with_depth

SQRT_2 = math.sqrt(2.0)


@runtime_checkable
class Agent(Protocol):
    def act(self, state: GameState) -> int: ...


class _Spec(BaseModel):
    model_config = ConfigDict(frozen=True)


class RandomAgentSpec(_Spec):
    kind: Literal["random"] = "random"
    seed: int | None = None


class GreedyAgentSpec(_Spec):
    kind: Literal["greedy"] = "greedy"
    seed: int | None = None


class RightmostAgentSpec(_Spec):
    kind: Literal["rightmost"] = "rightmost"


class AlphaBetaAgentSpec(_Spec):
    kind: Literal["alphabeta"] = "alphabeta"
    depth: int | None = Field(default=None, description="Plies; None = full search.")


class ExtensionAgentSpec(_Spec):
    kind: Literal["extension"] = "extension"


class BlockerAgentSpec(_Spec):
    kind: Literal["blocker"] = "blocker"


class CenterAgentSpec(_Spec):
    kind: Literal["center"] = "center"


class ForkAwareAgentSpec(_Spec):
    kind: Literal["forkaware"] = "forkaware"
    seed: int | None = None


class PotentialAwareAgentSpec(_Spec):
    kind: Literal["potentialaware"] = "potentialaware"
    seed: int | None = None


class MCTSAgentSpec(_Spec):
    kind: Literal["mcts"] = "mcts"
    n_simulations: int = Field(default=200, gt=0, description="Rollouts per move.")
    exploration_c: float = Field(
        default=SQRT_2,
        gt=0,
        description="UCB1 exploration constant. Default is sqrt(2).",
    )
    seed: int | None = None
    rollout: Literal["random", "forkaware"] = Field(
        default="random",
        description=(
            "Rollout policy. `random` is the textbook UCT baseline. `forkaware` "
            "biases rollouts toward open-end-aware play — much higher-quality "
            "evaluation per simulation on deep games, at modest extra cost."
        ),
    )


class HumanAgentSpec(_Spec):
    """A human player. Decisions are made client-side; the server never asks
    a human to act. Existing as a serializable identity so saved game records
    document who actually played.
    """

    kind: Literal["human"] = "human"


class LLMAgentSpec(_Spec):
    """OpenAI-compatible remote LLM inference agent."""

    kind: Literal["llm"] = "llm"
    base_url: str = Field(min_length=1, description="OpenAI-compatible API base URL.")
    api_key_env: str = Field(
        min_length=1,
        description="Env var name containing the bearer token, if required.",
    )
    model: str = Field(min_length=1, description="Remote model identifier.")
    temperature: float = Field(default=0.0, ge=0.0)
    max_tokens: int = Field(default=32, gt=0)
    timeout_s: float = Field(default=30.0, gt=0)
    prompt_style: Literal["json_v1"] = "json_v1"
    fallback: Literal["rightmost_legal"] = "rightmost_legal"
    seed: int | None = None


class AlphaZeroAgentSpec(_Spec):
    """Trained PUCT agent. Identified by the checkpoint it loads; everything
    else (n_simulations, c_puct, temperature) is at-inference behavior."""

    kind: Literal["alphazero"] = "alphazero"
    checkpoint_path: str = Field(description="Path to a .pt checkpoint.")
    n_simulations: int = Field(default=200, gt=0)
    c_puct: float = Field(default=1.5, gt=0)
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        description="0 = argmax visit counts; >0 = sample from visit distribution.",
    )
    seed: int | None = None


AgentSpec = Annotated[
    RandomAgentSpec
    | GreedyAgentSpec
    | RightmostAgentSpec
    | AlphaBetaAgentSpec
    | ExtensionAgentSpec
    | BlockerAgentSpec
    | CenterAgentSpec
    | ForkAwareAgentSpec
    | PotentialAwareAgentSpec
    | MCTSAgentSpec
    | HumanAgentSpec
    | LLMAgentSpec
    | AlphaZeroAgentSpec,
    Field(discriminator="kind"),
]


KINDS: tuple[str, ...] = (
    "human",
    "random",
    "greedy",
    "rightmost",
    "alphabeta",
    "extension",
    "blocker",
    "center",
    "forkaware",
    "potentialaware",
    "mcts",
    "llm",
    "alphazero",
)
SPEC_TYPES: dict[str, type[BaseModel]] = {
    "human": HumanAgentSpec,
    "random": RandomAgentSpec,
    "greedy": GreedyAgentSpec,
    "rightmost": RightmostAgentSpec,
    "alphabeta": AlphaBetaAgentSpec,
    "extension": ExtensionAgentSpec,
    "blocker": BlockerAgentSpec,
    "center": CenterAgentSpec,
    "forkaware": ForkAwareAgentSpec,
    "potentialaware": PotentialAwareAgentSpec,
    "mcts": MCTSAgentSpec,
    "llm": LLMAgentSpec,
    "alphazero": AlphaZeroAgentSpec,
}

LABELS: dict[str, str] = {
    "human": "Human",
    "random": "Random",
    "greedy": "Greedy (1-ply)",
    "rightmost": "Rightmost",
    "alphabeta": "Alpha-Beta",
    "extension": "Extension",
    "blocker": "Blocker",
    "center": "Center",
    "forkaware": "ForkAware",
    "potentialaware": "PotentialAware",
    "mcts": "MCTS (UCT)",
    "llm": "LLM",
    "alphazero": "AlphaZero",
}


class _Random:
    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng

    def act(self, state: GameState) -> int:
        return int(self._rng.choice(legal_actions(state)))


class _Rightmost:
    def act(self, state: GameState) -> int:
        return int(legal_actions(state)[-1])


_Score = TypeVar("_Score", bound=tuple)


def _argmax_one_ply(
    state: GameState,
    rng: np.random.Generator,
    score: Callable[[GameState], _Score],
) -> int:
    """Pick the move maximizing `score(next_state)`, breaking ties uniformly."""
    best: _Score | None = None
    candidates: list[int] = []
    for cell in legal_actions(state):
        s = score(step(state, int(cell)))
        if best is None or s > best:
            best, candidates = s, [int(cell)]
        elif s == best:
            candidates.append(int(cell))
    return int(rng.choice(candidates))


class _Greedy:
    """One-ply lookahead scored by (my_run_len, my_end, -opp_run_len, -opp_end)."""

    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng

    def act(self, state: GameState) -> int:
        me = state.current_player
        opp = O if me == X else X

        def score(ns: GameState) -> tuple[int, int, int, int]:
            ml, me_end = longest_run(ns.board, me)
            ol, op_end = longest_run(ns.board, opp)
            return (ml, me_end, -ol, -op_end)

        return _argmax_one_ply(state, self._rng, score)


class _AlphaBeta:
    """Negamax + alpha-beta. With `depth=None`, plays the exact solver in
    `solve.py` (perfect on small `(N, schedule)`). With a finite `depth`,
    uses a depth-limited search with a simple heuristic at the horizon."""

    def __init__(self, depth: int | None) -> None:
        self._depth = depth

    def act(self, state: GameState) -> int:
        result = search_with_depth(state, self._depth)
        if result.best_cell < 0:
            raise ValueError("no legal moves: state is terminal")
        return int(result.best_cell)


def _longest_run_with_span(board: np.ndarray, mark: np.uint8) -> tuple[int, int, int]:
    """`longest_run` returns `(length, end)`. Here we also recover the start
    index of the rightmost-tied longest run, since extension/blocking logic
    needs both endpoints to find adjacent cells."""
    length, end = longest_run(board, mark)
    if length == 0:
        return 0, -1, -1
    return length, end - length + 1, end


def _empty_or_oob(board: np.ndarray, idx: int) -> bool:
    """A cell is "open" for the purpose of extending into if it's a real,
    empty cell on the board. Out-of-bounds counts as a wall (not open)."""
    return 0 <= idx < board.shape[0] and board[idx] == EMPTY


def _play_adjacent_to_run(state: GameState, mark: np.uint8) -> int:
    """Play adjacent to the longest run of `mark`, preferring the right side
    (aligns with the rightmost-end tie-break). Falls back to the rightmost
    legal cell if there's no run or it's fully capped."""
    length, start, end = _longest_run_with_span(state.board, mark)
    if length > 0:
        right = end + 1
        if _empty_or_oob(state.board, right):
            return int(right)
        left = start - 1
        if _empty_or_oob(state.board, left):
            return int(left)
    return int(legal_actions(state)[-1])


class _Extension:
    """Always extend the longest run of my own mark, preferring the rightmost
    open neighbor. Falls back to the rightmost empty cell."""

    def act(self, state: GameState) -> int:
        return _play_adjacent_to_run(state, state.current_player)


class _Blocker:
    """Block the opponent's longest run on its rightmost open side. Falls
    back to the rightmost empty cell if the opponent has no extendable run."""

    def act(self, state: GameState) -> int:
        opp = O if state.current_player == X else X
        return _play_adjacent_to_run(state, opp)


class _Center:
    """Play the center of the largest contiguous empty segment, breaking ties
    toward the right (mirrors the game's tie-break philosophy). Reveals how
    much pure geometry buys you absent any tie-break awareness."""

    def act(self, state: GameState) -> int:
        board = state.board
        n = board.shape[0]
        # Scan right-to-left so a tie on segment length resolves to the
        # rightmost segment (matches the game's tie-break direction).
        best_len = 0
        best_mid = 0
        i = n - 1
        while i >= 0:
            if board[i] != EMPTY:
                i -= 1
                continue
            j = i
            while j >= 0 and board[j] == EMPTY:
                j -= 1
            # segment is [j+1 ... i]
            seg_len = i - j
            seg_mid = (j + 1 + i + 1) // 2  # bias right on even-length segments
            if seg_len > best_len:
                best_len, best_mid = seg_len, seg_mid
            i = j
        return int(best_mid)


# ---------- MCTS (UCT) ------------------------------------------------------


def _run_value_open_aware(board: np.ndarray, player: np.uint8) -> tuple[float, int]:
    """Score the player's best run, weighted by how extensible it still is.

    A run's "open ends" are the count of empty cells immediately adjacent to
    it (0, 1, or 2). A 2-open-end run is a fork: an opponent with one
    placement can block only one side. We score each run as

        length + 0.5 · open_ends · length

    so a 2-run with two open ends (4) beats a static 3-run with zero open
    ends (3) — capturing that the former is about to become a 3- or 4-run
    while the latter is dead. Returns (best_score, end_index_of_best_run) so
    callers can apply the rightmost-end tie-break.
    """
    n = board.shape[0]
    best_score = 0.0
    best_end = -1
    i = 0
    while i < n:
        if board[i] != player:
            i += 1
            continue
        j = i
        while j < n and board[j] == player:
            j += 1
        length = j - i
        left_open = i > 0 and board[i - 1] == EMPTY
        right_open = j < n and board[j] == EMPTY
        open_ends = (1 if left_open else 0) + (1 if right_open else 0)
        score = float(length) + 0.5 * open_ends * length
        end = j - 1
        if (score, end) > (best_score, best_end):
            best_score, best_end = score, end
        i = j
    return best_score, best_end


class _ForkAware:
    """1-ply lookahead scored by open-end-aware run value.

    For each legal cell, simulate placing it and score the resulting position
    as `(my_eval, my_end, -opp_eval, -opp_end)` — same shape as `_Greedy`
    but with `_run_value_open_aware` instead of raw run length. Captures the
    difference between a dead 3-run and an extensible 2-run, which raw
    length-greedy collapses together.
    """

    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng

    def act(self, state: GameState) -> int:
        me = state.current_player
        opp = O if me == X else X

        def score(ns: GameState) -> tuple[float, int, float, int]:
            my_v, my_end = _run_value_open_aware(ns.board, me)
            op_v, op_end = _run_value_open_aware(ns.board, opp)
            return (my_v, my_end, -op_v, -op_end)

        return _argmax_one_ply(state, self._rng, score)


def _remaining_placements_by_side(state: GameState) -> tuple[int, int]:
    """How many placements X and O still have, summed across the current
    turn's remainder and all future turns. The only baseline that reasons
    about the *schedule*, not just the board."""
    schedule = state.spec.schedule
    cur = state.turn_idx
    x_rem = 0
    o_rem = 0
    if cur < len(schedule):
        if cur % 2 == 0:
            x_rem += state.placements_left
        else:
            o_rem += state.placements_left
    for t in range(cur + 1, len(schedule)):
        if t % 2 == 0:
            x_rem += schedule[t]
        else:
            o_rem += schedule[t]
    return x_rem, o_rem


def _max_potential_run(board: np.ndarray, player: np.uint8, my_remaining: int) -> int:
    """Length of the longest window the player could *still* turn into a run.

    A window of length L is achievable if (1) it contains no opponent marks
    and (2) the number of empty cells inside is ≤ `my_remaining`. We scan
    from longest possible down and return the first hit. This is what
    "PotentialAware" looks at instead of the current longest run.
    """
    n = board.shape[0]
    opp = O if player == X else X
    # Upper bound on achievable length: own marks already on the board + my
    # remaining placements, capped by N. No window longer than that can be
    # filled even in the best case.
    own_total = int((board == player).sum())
    upper = min(n, own_total + my_remaining)
    if upper <= 0:
        return 0
    # Sliding-window counts for each length, descending.
    for L in range(upper, 0, -1):
        # Initialize counts for the first window of length L.
        opp_count = 0
        empty_count = 0
        for k in range(L):
            c = board[k]
            if c == opp:
                opp_count += 1
            elif c == EMPTY:
                empty_count += 1
        if opp_count == 0 and empty_count <= my_remaining:
            return L
        for start in range(1, n - L + 1):
            # Slide: drop board[start-1], add board[start+L-1].
            out_c = board[start - 1]
            in_c = board[start + L - 1]
            if out_c == opp:
                opp_count -= 1
            elif out_c == EMPTY:
                empty_count -= 1
            if in_c == opp:
                opp_count += 1
            elif in_c == EMPTY:
                empty_count += 1
            if opp_count == 0 and empty_count <= my_remaining:
                return L
    return 0


class _PotentialAware:
    """1-ply lookahead scored by what runs the *board still permits*.

    For each candidate move, after playing it compute the longest run each
    side could in principle still achieve given that side's remaining
    placements across all future turns. Score = (my_potential − opp_potential)
    with rightmost tie-break.

    Unique among the baselines: it uses `state.spec.schedule`, not just the
    current board. Should be especially strong on configs with a big
    multi-placement turn coming up — preempt now.
    """

    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng

    def act(self, state: GameState) -> int:
        me = state.current_player
        opp = O if me == X else X
        # `_remaining_placements_by_side` depends only on (schedule, turn_idx,
        # placements_left), all of which are identical across candidate moves
        # from a given root state — so we hoist it out of the inner loop.
        legal = legal_actions(state)
        sample = step(state, int(legal[0]))
        x_rem, o_rem = _remaining_placements_by_side(sample)
        my_rem, op_rem = (x_rem, o_rem) if me == X else (o_rem, x_rem)

        def score(ns: GameState) -> tuple[int, int, int, int]:
            my_pot = _max_potential_run(ns.board, me, my_rem)
            op_pot = _max_potential_run(ns.board, opp, op_rem)
            # Tie-break on today's longest-run end so ties resolve toward
            # positions that look good now, not just in potential.
            _, my_end = longest_run(ns.board, me)
            _, op_end = longest_run(ns.board, opp)
            return (my_pot - op_pot, my_end, -op_pot, -op_end)

        return _argmax_one_ply(state, self._rng, score)


class _MCTSNode:
    __slots__ = ("state", "parent", "incoming_cell", "children", "untried", "visits", "value_sum")

    def __init__(
        self,
        state: GameState,
        parent: _MCTSNode | None,
        incoming_cell: int,
    ) -> None:
        self.state = state
        self.parent = parent
        self.incoming_cell = incoming_cell
        self.children: list[_MCTSNode] = []
        if state.is_terminal:
            self.untried: list[int] = []
        else:
            self.untried = sorted(legal_actions(state).tolist(), reverse=True)
        self.visits = 0
        self.value_sum = 0.0  # cumulative value from X's perspective


class _MCTS:
    """Classical UCT with random rollouts. Deterministic given `seed`.

    Acting:
      1. Build a fresh tree rooted at the current state.
      2. Run `n_simulations` rollouts: selection → expansion → rollout → backup.
      3. Pick the most-visited child as the move (the AlphaZero convention;
         more visited == better, with no need to inspect estimated values).

    The tree is discarded between moves. For research where you want to
    reuse it across moves, that's a Phase-3 concern.
    """

    def __init__(
        self,
        n_simulations: int,
        exploration_c: float,
        rng: np.random.Generator,
        rollout: Literal["random", "forkaware"] = "random",
    ) -> None:
        self._n = n_simulations
        self._c = exploration_c
        self._rng = rng
        # Heuristic rollouts share the MCTS rng so reproducibility is preserved
        # end-to-end.
        self._rollout_agent: _ForkAware | None = (
            _ForkAware(rng) if rollout == "forkaware" else None
        )

    def act(self, state: GameState) -> int:
        if state.is_terminal:
            raise ValueError("no legal moves: state is terminal")
        root = _MCTSNode(state, parent=None, incoming_cell=-1)
        for _ in range(self._n):
            leaf = self._select(root)
            value = self._rollout(leaf.state)
            self._backup(leaf, value)
        if not root.children:
            return int(legal_actions(state)[-1])
        best = max(root.children, key=lambda c: c.visits)
        return int(best.incoming_cell)

    def _select(self, node: _MCTSNode) -> _MCTSNode:
        while not node.state.is_terminal:
            if node.untried:
                return self._expand(node)
            node = self._best_child(node)
        return node

    def _expand(self, node: _MCTSNode) -> _MCTSNode:
        cell = node.untried.pop()
        child = _MCTSNode(step(node.state, cell), parent=node, incoming_cell=cell)
        node.children.append(child)
        return child

    def _best_child(self, node: _MCTSNode) -> _MCTSNode:
        # UCB1 from the side-to-move's perspective. Values are stored from X's
        # perspective, so flip for O.
        side_sign = 1.0 if node.state.current_player == X else -1.0
        log_n = math.log(node.visits) if node.visits > 0 else 0.0
        best: _MCTSNode | None = None
        best_score = -math.inf
        for child in node.children:
            mean = (child.value_sum / child.visits) if child.visits > 0 else 0.0
            exploit = side_sign * mean
            explore = self._c * math.sqrt(log_n / child.visits) if child.visits > 0 else math.inf
            score = exploit + explore
            if score > best_score:
                best, best_score = child, score
        assert best is not None
        return best

    def _rollout(self, state: GameState) -> float:
        s = state
        if self._rollout_agent is None:
            while not s.is_terminal:
                cells = legal_actions(s)
                s = step(s, int(self._rng.choice(cells)))
        else:
            agent = self._rollout_agent
            while not s.is_terminal:
                s = step(s, int(agent.act(s)))
        return float(outcome(s))  # +1 / -1 from X's perspective

    def _backup(self, leaf: _MCTSNode, value: float) -> None:
        node: _MCTSNode | None = leaf
        while node is not None:
            node.visits += 1
            node.value_sum += value
            node = node.parent


class _LLMAgent:
    def __init__(self, spec: LLMAgentSpec) -> None:
        self._spec = spec

    def act(self, state: GameState) -> int:
        if state.is_terminal:
            raise ValueError("no legal moves: state is terminal")
        legal = [int(cell) for cell in legal_actions(state).tolist()]
        prompt = render_prompt(state, prompt_style=self._spec.prompt_style)

        raw_response: str | None = None
        parsed_cell: int | None = None
        is_legal = False
        error: str | None = None
        started = time.perf_counter()
        try:
            raw_response = request_chat_completion(
                base_url=self._spec.base_url,
                api_key_env=self._spec.api_key_env,
                model=self._spec.model,
                prompt=prompt,
                temperature=self._spec.temperature,
                max_tokens=self._spec.max_tokens,
                timeout_s=self._spec.timeout_s,
                seed=self._spec.seed,
            )
            parsed_cell = parse_json_cell_reply(raw_response)
            is_legal = parsed_cell in legal
            if not is_legal:
                error = f"illegal cell: {parsed_cell}"
        except Exception as e:  # noqa: BLE001 - fallback is the contract here
            error = str(e)
        latency_ms = (time.perf_counter() - started) * 1000.0

        if is_legal and parsed_cell is not None:
            chosen = parsed_cell
            fallback_used = False
        else:
            chosen = legal[-1]
            fallback_used = True

        append_trace(
            {
                "prompt": prompt,
                "raw_response": raw_response,
                "parsed_cell": parsed_cell,
                "chosen_cell": chosen,
                "legality": is_legal,
                "fallback_used": fallback_used,
                "latency_ms": latency_ms,
                "error": error,
                "model_config": {
                    "kind": self._spec.kind,
                    "base_url": self._spec.base_url,
                    "api_key_env": self._spec.api_key_env,
                    "model": self._spec.model,
                    "temperature": self._spec.temperature,
                    "max_tokens": self._spec.max_tokens,
                    "timeout_s": self._spec.timeout_s,
                    "prompt_style": self._spec.prompt_style,
                    "fallback": self._spec.fallback,
                    "seed": self._spec.seed,
                },
                "state": {
                    "n": state.spec.n,
                    "schedule": list(state.spec.schedule),
                    "turn_idx": state.turn_idx,
                    "placements_left": state.placements_left,
                    "current_player": "X" if int(state.current_player) == int(X) else "O",
                    "board": board_to_string(state),
                    "legal_cells": legal,
                },
            }
        )
        return chosen


def build(spec: AgentSpec) -> Agent:
    """Instantiate a runnable agent from its spec. Human specs cannot be built
    because humans act in the client; callers must check `spec.kind != "human"`
    before calling, or be ready to handle this exception.
    """
    if isinstance(spec, RandomAgentSpec):
        return _Random(np.random.default_rng(spec.seed))
    if isinstance(spec, GreedyAgentSpec):
        return _Greedy(np.random.default_rng(spec.seed))
    if isinstance(spec, RightmostAgentSpec):
        return _Rightmost()
    if isinstance(spec, AlphaBetaAgentSpec):
        return _AlphaBeta(spec.depth)
    if isinstance(spec, ExtensionAgentSpec):
        return _Extension()
    if isinstance(spec, BlockerAgentSpec):
        return _Blocker()
    if isinstance(spec, CenterAgentSpec):
        return _Center()
    if isinstance(spec, ForkAwareAgentSpec):
        return _ForkAware(np.random.default_rng(spec.seed))
    if isinstance(spec, PotentialAwareAgentSpec):
        return _PotentialAware(np.random.default_rng(spec.seed))
    if isinstance(spec, MCTSAgentSpec):
        return _MCTS(
            spec.n_simulations,
            spec.exploration_c,
            np.random.default_rng(spec.seed),
            rollout=spec.rollout,
        )
    if isinstance(spec, HumanAgentSpec):
        raise ValueError("human agents act in the client; cannot build a server agent")
    if isinstance(spec, LLMAgentSpec):
        return _LLMAgent(spec)
    if isinstance(spec, AlphaZeroAgentSpec):
        # Lazy import so callers that only use classical baselines don't pay
        # the torch import cost.
        from .nn.agent import build_alphazero

        return build_alphazero(
            checkpoint_path=spec.checkpoint_path,
            n_simulations=spec.n_simulations,
            c_puct=spec.c_puct,
            temperature=spec.temperature,
            seed=spec.seed,
        )
    raise ValueError(f"unknown spec: {spec!r}")
