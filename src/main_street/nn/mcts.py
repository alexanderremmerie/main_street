"""PUCT search over a trained `(Encoder, Model)` pair.

`puct_search` is a pure function: takes a root state and the (encoder, model)
pair, returns a `Node` whose children's `visit_count` is the AlphaZero policy
target. `select_action` picks a move from a finished search; the caller
controls temperature.

Player handling. Within a turn, the current player does not flip between
states — `step()` keeps the same player active until `placements_left == 0`.
We carry value in X's frame everywhere; selection flips sign per-node based
on `state.current_player`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Final

import numpy as np
import torch
import torch.nn.functional as F

from ..core import GameState, X, outcome, step
from .encode import Encoder, to_device
from .models import Model

_PUCT_DEFAULT_C: Final[float] = 1.5


@dataclass(slots=True)
class Node:
    state: GameState
    priors: dict[int, float] = field(default_factory=dict)
    children: dict[int, Node] = field(default_factory=dict)
    visit_count: int = 0
    value_sum_x: float = 0.0  # cumulative value from X's perspective


def _select(node: Node, c_puct: float) -> int:
    parent_is_x = node.state.current_player == X
    sqrt_parent = math.sqrt(max(1, node.visit_count))
    best_a = -1
    best_score = -math.inf
    for a, p in node.priors.items():
        child = node.children.get(a)
        if child is None or child.visit_count == 0:
            q = 0.0
            n = 0
        else:
            avg_x = child.value_sum_x / child.visit_count
            q = avg_x if parent_is_x else -avg_x
            n = child.visit_count
        u = c_puct * p * sqrt_parent / (1 + n)
        score = q + u
        if score > best_score:
            best_score, best_a = score, a
    return best_a


def _expand(
    node: Node, model: Model, encoder: Encoder, device: torch.device
) -> float:
    """Evaluate `node` with the network. Sets priors. Returns value from
    X's perspective."""
    with torch.inference_mode():
        inputs = to_device(encoder([node.state]), device)
        logits, value = model(inputs)
    legal = inputs["legal_mask"][0]
    probs = F.softmax(logits[0], dim=-1)
    priors: dict[int, float] = {}
    for c in range(legal.shape[0]):
        if bool(legal[c]):
            # Floor priors so PUCT can still explore actions a confident
            # softmax pushed near zero.
            priors[c] = max(float(probs[c]), 1e-6)
    total = sum(priors.values())
    if total > 0:
        for k in priors:
            priors[k] /= total
    node.priors = priors
    v = float(value[0])
    return v if node.state.current_player == X else -v


def _add_dirichlet(
    node: Node, alpha: float, eps: float, rng: np.random.Generator
) -> None:
    if not node.priors:
        return
    actions = list(node.priors.keys())
    noise = rng.dirichlet([alpha] * len(actions))
    for i, a in enumerate(actions):
        node.priors[a] = (1.0 - eps) * node.priors[a] + eps * float(noise[i])


def puct_search(
    root_state: GameState,
    model: Model,
    encoder: Encoder,
    n_simulations: int,
    c_puct: float = _PUCT_DEFAULT_C,
    dirichlet_alpha: float | None = None,
    dirichlet_eps: float = 0.0,
    rng: np.random.Generator | None = None,
) -> Node:
    root = Node(state=root_state)
    if root_state.is_terminal:
        return root
    device = next(model.parameters()).device
    v_x = _expand(root, model, encoder, device)
    root.visit_count = 1
    root.value_sum_x = v_x
    if dirichlet_eps > 0.0 and dirichlet_alpha is not None and rng is not None:
        _add_dirichlet(root, dirichlet_alpha, dirichlet_eps, rng)

    for _ in range(n_simulations):
        node = root
        path: list[Node] = [root]
        # Walk down until an unexpanded (no priors) or terminal node.
        while node.priors and not node.state.is_terminal:
            a = _select(node, c_puct)
            child = node.children.get(a)
            if child is None:
                child = Node(state=step(node.state, a))
                node.children[a] = child
            path.append(child)
            node = child
        if node.state.is_terminal:
            leaf_v_x = float(outcome(node.state))
        else:
            leaf_v_x = _expand(node, model, encoder, device)
        for n in path:
            n.visit_count += 1
            n.value_sum_x += leaf_v_x

    return root


def visit_distribution(root: Node, temperature: float = 1.0) -> dict[int, float]:
    if not root.children:
        return dict(root.priors)
    counts = {a: c.visit_count for a, c in root.children.items()}
    if temperature == 0.0:
        best_a = max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
        return {a: (1.0 if a == best_a else 0.0) for a in counts}
    powered = {a: float(n) ** (1.0 / temperature) for a, n in counts.items()}
    total = sum(powered.values())
    if total == 0:
        return {a: root.priors[a] for a in root.priors}
    return {a: v / total for a, v in powered.items()}


def select_action(
    root: Node,
    temperature: float = 0.0,
    rng: np.random.Generator | None = None,
) -> int:
    dist = visit_distribution(root, temperature)
    if temperature == 0.0:
        return max(dist.items(), key=lambda kv: (kv[1], kv[0]))[0]
    if rng is None:
        rng = np.random.default_rng()
    actions = np.array(list(dist.keys()), dtype=np.int64)
    probs = np.array(list(dist.values()), dtype=np.float64)
    probs /= probs.sum()
    return int(rng.choice(actions, p=probs))
