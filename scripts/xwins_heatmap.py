"""
N-sweep X-wins oracle-agreement heatmap.

For each N in n_min..n_max, samples `n_samples` random (N, schedule) pairs
where the exact solver confirms X wins from the initial position, then scores
each model on those initial positions. Outputs a CSV ready for matplotlib.

Usage:
  uv run python scripts/xwins_heatmap.py \
      --ckpts data/runs/*/checkpoints/final.pt \
      --labels 04_mixed 06_12to48 ... \
      --out results/xwins_heatmap.csv

  # or score baselines only
  uv run python scripts/xwins_heatmap.py --baselines-only

Output columns:
  agent, N, oracle_agreement, n_positions
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

# ── repo root on path ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from main_street.agents import (
    AlphaBetaAgentSpec,
    AlphaZeroAgentSpec,
    ForkAwareAgentSpec,
    GreedyAgentSpec,
    RandomAgentSpec,
    RightmostAgentSpec,
    build,
)
from main_street.core import GameSpec, GameState
from main_street.eval.metrics import score_agent
from main_street.eval.positions import PositionSet, SourceSpec, build_position_set
from main_street.solve import Solver
from main_street.spec_sampling import SpecSamplerConfig


def sample_xwins_positions(
    n: int,
    n_samples: int,
    rng: np.random.Generator,
    max_attempts: int = 5000,
) -> PositionSet:
    """Sample up to `n_samples` initial positions where X is guaranteed to win."""
    sampler = SpecSamplerConfig(
        n_min=n,
        n_max=n,
        turns_min=2,
        turns_max=min(8, n),
        fill_min=0.40,
        fill_max=0.90,
        max_marks_per_turn=max(1, n // 2),
        random_weight=0.5,
        arc_weight=0.2,
        few_big_weight=0.2,
        many_small_weight=0.1,
    )
    solver = Solver()
    sources: list[SourceSpec] = []
    seen: set[tuple[int, tuple[int, ...]]] = set()
    attempts = 0

    while len(sources) < n_samples and attempts < max_attempts:
        attempts += 1
        spec = sampler.sample(rng)
        key = (spec.n, spec.schedule)
        if key in seen:
            continue
        seen.add(key)

        state = GameState.initial(spec)
        result = solver.solve(state)
        if result.value == 1:  # X wins under optimal play
            sources.append(
                SourceSpec(
                    spec=spec,
                    mode="initial_only",
                    label=f"N{n}",
                )
            )

    if not sources:
        # Fallback: for very small N, try two-turn schedules directly.
        for g1 in range(1, n):
            for g2 in range(g1 + 1, n + 1):
                if g1 + g2 > n:
                    continue
                spec = GameSpec(n=n, schedule=(g1, g2))
                key = (spec.n, spec.schedule)
                if key in seen:
                    continue
                seen.add(key)
                state = GameState.initial(spec)
                result = solver.solve(state)
                if result.value == 1:
                    sources.append(
                        SourceSpec(spec=spec, mode="initial_only", label=f"N{n}")
                    )
                    if len(sources) >= n_samples:
                        break
            if len(sources) >= n_samples:
                break

    return build_position_set(f"xwins_N{n}", sources)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-min", type=int, default=5)
    p.add_argument("--n-max", type=int, default=16)
    p.add_argument("--n-samples", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--ckpts",
        nargs="*",
        default=[],
        help="paths to .pt checkpoint files",
    )
    p.add_argument(
        "--labels",
        nargs="*",
        default=[],
        help="short label per checkpoint (same order as --ckpts)",
    )
    p.add_argument(
        "--n-sims",
        type=int,
        default=64,
        help="MCTS simulations for AlphaZero checkpoints",
    )
    p.add_argument("--baselines-only", action="store_true")
    p.add_argument("--out", default="results/xwins_heatmap.csv")
    args = p.parse_args()

    ckpts = args.ckpts or []
    labels = args.labels or []
    if labels and len(labels) != len(ckpts):
        p.error("--labels must have the same length as --ckpts")
    if not labels:
        labels = [Path(c).parent.parent.name[:20] for c in ckpts]

    # Build agents list: baselines first, then checkpoints.
    agents: list[tuple[str, object]] = [
        ("random", build(RandomAgentSpec(seed=0))),
        ("greedy", build(GreedyAgentSpec())),
        ("rightmost", build(RightmostAgentSpec())),
        ("forkaware", build(ForkAwareAgentSpec())),
        ("alphabeta", build(AlphaBetaAgentSpec())),
    ]
    if not args.baselines_only:
        for ckpt, label in zip(ckpts, labels):
            spec = AlphaZeroAgentSpec(
                checkpoint_path=ckpt,
                n_simulations=args.n_sims,
            )
            agents.append((label, build(spec)))

    rng = np.random.default_rng(args.seed)
    rows: list[dict] = []

    for n in range(args.n_min, args.n_max + 1):
        print(f"N={n}: sampling {args.n_samples} X-wins positions...", flush=True)
        ps = sample_xwins_positions(n, args.n_samples, rng)
        print(f"  got {len(ps)} positions")
        if len(ps) == 0:
            print(f"  WARNING: no X-wins positions found for N={n}, skipping")
            continue

        for label, agent in agents:
            score = score_agent(agent, ps, agent_label=label)
            rows.append(
                {
                    "agent": label,
                    "N": n,
                    "oracle_agreement": round(score.oracle_agreement, 4),
                    "n_positions": len(ps),
                }
            )
            print(f"  {label:20s}  agreement={score.oracle_agreement:.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["agent", "N", "oracle_agreement", "n_positions"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
