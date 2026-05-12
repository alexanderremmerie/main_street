"""Metrics over (agent, PositionSet).

`oracle_agreement` = fraction of positions where the agent's chosen cell is
in the optimal-cell mask. This is *top-1 agreement* with the exact solver,
which is what we actually care about: an agent that plays an optimal move at
every position plays perfectly.

`per_spec_agreement` breaks the same number down by source spec, so we can
see *where* an agent fails. The breakdown is what tells us whether failures
are concentrated on a config family (e.g. all (1,b,c) sandwiches) or spread.

`per_label_agreement` is for the diagnostics set: one row per named position.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from ..agents import Agent
from ..core import GameSpec
from .positions import PositionSet


@dataclass(frozen=True, slots=True)
class AgentScore:
    agent_label: str
    set_name: str
    n_positions: int
    oracle_agreement: float
    per_spec: dict[str, tuple[int, float]]  # spec_str -> (count, agreement)
    per_label: dict[str, bool]  # label -> agreed (only present if labels exist)


def _spec_str(spec: GameSpec) -> str:
    return f"n={spec.n} sched=({','.join(str(k) for k in spec.schedule)})"


def score_agent(agent: Agent, ps: PositionSet, agent_label: str = "agent") -> AgentScore:
    n = len(ps)
    correct = np.zeros(n, dtype=bool)

    for i in range(n):
        state = ps.state(i)
        cell = int(agent.act(state))
        ni = int(ps.n[i])
        correct[i] = bool(ps.optimal_mask[i, cell]) if 0 <= cell < ni else False

    per_spec: dict[str, tuple[int, float]] = {}
    by_spec: dict[int, list[bool]] = defaultdict(list)
    for i in range(n):
        by_spec[int(ps.spec_idx[i])].append(bool(correct[i]))
    for si, vals in by_spec.items():
        spec_str = _spec_str(ps.specs[si])
        per_spec[spec_str] = (len(vals), float(np.mean(vals)))

    per_label: dict[str, bool] = {}
    if any(ps.labels):
        for i, lbl in enumerate(ps.labels):
            if lbl:
                per_label[lbl] = bool(correct[i])

    return AgentScore(
        agent_label=agent_label,
        set_name=ps.name,
        n_positions=n,
        oracle_agreement=float(np.mean(correct)) if n > 0 else 0.0,
        per_spec=per_spec,
        per_label=per_label,
    )


def per_spec_agreement(score: AgentScore) -> list[tuple[str, int, float]]:
    """Convenience: (spec_str, count, agreement) sorted worst-first. Useful
    when ranking failure modes."""
    rows = [(k, c, a) for k, (c, a) in score.per_spec.items()]
    rows.sort(key=lambda r: r[2])
    return rows
