"""
Pairwise win-rate tournament: model checkpoints vs baselines.

Plays each pair on a shared set of sampled specs. Each spec is played twice
(swap_sides=True) so both colour advantages cancel. Outputs a CSV and prints
a summary table.

Usage:
  uv run python scripts/tournament.py \
      --ckpts data/runs/04_.../checkpoints/final.pt data/runs/06_.../checkpoints/final.pt \
      --labels 04_mixed 06_12to48 \
      --n-games 20 \
      --out results/tournament.csv

For small N (≤ 12), alphabeta is included; for larger N it is dropped since it
becomes too slow to be a useful baseline.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from main_street.agents import (
    AgentSpec,
    AlphaBetaAgentSpec,
    AlphaZeroAgentSpec,
    ForkAwareAgentSpec,
    GreedyAgentSpec,
    PotentialAwareAgentSpec,
    RandomAgentSpec,
    RightmostAgentSpec,
)
from main_street.core import GameSpec
from main_street.runner import play
from main_street.spec_sampling import SpecSamplerConfig


@dataclass
class MatchResult:
    agent_a: str
    agent_b: str
    n_games: int
    wins_a: int
    wins_b: int

    @property
    def win_rate_a(self) -> float:
        return self.wins_a / self.n_games if self.n_games > 0 else 0.0


def run_match(
    label_a: str,
    spec_a: AgentSpec,
    label_b: str,
    spec_b: AgentSpec,
    specs: list[GameSpec],
    n_games_per_spec: int,
    seed: int = 0,
) -> MatchResult:
    wins_a = wins_b = 0
    total = 0

    for game_idx, spec in enumerate(specs):
        for rep in range(n_games_per_spec):
            game_seed = seed * 100000 + game_idx * 100 + rep

            # A plays X, B plays O
            record = play(spec, spec_a, spec_b, seed=game_seed)
            if record.outcome == 1:
                wins_a += 1   # X wins → A wins
            else:
                wins_b += 1
            total += 1

            # Swap: B plays X, A plays O
            record = play(spec, spec_b, spec_a, seed=game_seed + 1)
            if record.outcome == 1:
                wins_b += 1   # X wins → B wins
            else:
                wins_a += 1
            total += 1

    return MatchResult(
        agent_a=label_a,
        agent_b=label_b,
        n_games=total,
        wins_a=wins_a,
        wins_b=wins_b,
    )


def sample_arena_specs(
    n_min: int, n_max: int, n_specs: int, seed: int = 0
) -> list[GameSpec]:
    sampler = SpecSamplerConfig(
        n_min=n_min,
        n_max=n_max,
        turns_min=2,
        turns_max=min(10, n_max),
        fill_min=0.45,
        fill_max=0.85,
        max_marks_per_turn=max(1, n_max // 3),
    )
    rng = np.random.default_rng(seed)
    seen: set[tuple[int, tuple[int, ...]]] = set()
    specs: list[GameSpec] = []
    while len(specs) < n_specs:
        s = sampler.sample(rng)
        key = (s.n, s.schedule)
        if key not in seen:
            seen.add(key)
            specs.append(s)
    return specs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="*", default=[])
    p.add_argument("--labels", nargs="*", default=[])
    p.add_argument("--n-sims", type=int, default=64)
    p.add_argument("--n-specs", type=int, default=10, help="game specs per arena")
    p.add_argument("--n-games", type=int, default=5, help="games per spec per matchup")
    p.add_argument("--n-min", type=int, default=8)
    p.add_argument("--n-max", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/tournament.csv")
    args = p.parse_args()

    ckpts = args.ckpts or []
    labels = args.labels or []
    if labels and len(labels) != len(ckpts):
        p.error("--labels must have the same length as --ckpts")
    if not labels:
        labels = [Path(c).parent.parent.name[:20] for c in ckpts]

    # Build agent registry: (label, spec)
    agent_specs: list[tuple[str, AgentSpec]] = [
        ("random", RandomAgentSpec(seed=0)),
        ("greedy", GreedyAgentSpec()),
        ("rightmost", RightmostAgentSpec()),
        ("forkaware", ForkAwareAgentSpec()),
        ("potentialaware", PotentialAwareAgentSpec()),
    ]
    # Include alphabeta only for small arenas
    if args.n_max <= 12:
        agent_specs.append(("alphabeta", AlphaBetaAgentSpec()))
    for ckpt, label in zip(ckpts, labels):
        agent_specs.append(
            (
                label,
                AlphaZeroAgentSpec(
                    checkpoint_path=ckpt,
                    n_simulations=args.n_sims,
                ),
            )
        )

    arena_specs = sample_arena_specs(args.n_min, args.n_max, args.n_specs, args.seed)
    print(f"Arena: {len(arena_specs)} specs, N∈[{args.n_min},{args.n_max}]")

    rows: list[dict] = []
    pairs = list(itertools.combinations(range(len(agent_specs)), 2))
    for i, j in pairs:
        la, sa = agent_specs[i]
        lb, sb = agent_specs[j]
        print(f"  {la} vs {lb}...", flush=True)
        result = run_match(la, sa, lb, sb, arena_specs, args.n_games, args.seed)
        rows.append(
            {
                "agent_a": result.agent_a,
                "agent_b": result.agent_b,
                "n_games": result.n_games,
                "wins_a": result.wins_a,
                "wins_b": result.wins_b,
                "win_rate_a": round(result.win_rate_a, 3),
            }
        )
        print(
            f"    {la} win rate: {result.win_rate_a:.3f} "
            f"({result.wins_a}/{result.n_games})"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["agent_a", "agent_b", "n_games", "wins_a", "wins_b", "win_rate_a"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
