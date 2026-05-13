from __future__ import annotations

import json
from functools import partial
from typing import Any

from ..grpo.config import RewardConfig


def completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts: list[str] = []
        for item in completion:
            if isinstance(item, dict):
                content = item.get("content")
                if isinstance(content, str):
                    parts.append(content)
        if parts:
            return "\n".join(parts)
    if isinstance(completion, dict):
        content = completion.get("content")
        if isinstance(content, str):
            return content
    raise ValueError(f"unsupported completion payload: {type(completion).__name__}")


def parse_completion_cell(completion: Any) -> int:
    text = completion_to_text(completion)
    payload = json.loads(text)
    if not isinstance(payload, dict) or set(payload) != {"cell"}:
        raise ValueError("completion must be exactly {'cell': int}")
    cell = payload["cell"]
    if isinstance(cell, bool) or not isinstance(cell, int):
        raise ValueError("cell must be an integer")
    return int(cell)


def strict_json_reward(
    completions: list[Any],
    *,
    valid_reward: float = 0.1,
    invalid_reward: float = 0.0,
    **_: Any,
) -> list[float]:
    rewards: list[float] = []
    for completion in completions:
        try:
            parse_completion_cell(completion)
        except (ValueError, json.JSONDecodeError):
            rewards.append(invalid_reward)
        else:
            rewards.append(valid_reward)
    return rewards


def legal_move_reward(
    completions: list[Any],
    *,
    legal_cells: list[list[int]],
    legal_reward: float = 0.5,
    illegal_reward: float = -1.0,
    **_: Any,
) -> list[float]:
    rewards: list[float] = []
    for completion, legal in zip(completions, legal_cells, strict=True):
        try:
            cell = parse_completion_cell(completion)
        except (ValueError, json.JSONDecodeError):
            rewards.append(illegal_reward)
            continue
        rewards.append(legal_reward if cell in legal else illegal_reward)
    return rewards


def optimal_move_reward(
    completions: list[Any],
    *,
    optimal_cells: list[list[int]],
    optimal_reward: float = 1.0,
    suboptimal_reward: float = 0.0,
    invalid_move_reward: float = -1.0,
    **_: Any,
) -> list[float]:
    rewards: list[float] = []
    for completion, optimal in zip(completions, optimal_cells, strict=True):
        try:
            cell = parse_completion_cell(completion)
        except (ValueError, json.JSONDecodeError):
            rewards.append(invalid_move_reward)
            continue
        rewards.append(optimal_reward if cell in optimal else suboptimal_reward)
    return rewards


def build_reward_funcs(cfg: RewardConfig) -> tuple[list[Any], list[float]]:
    format_func = partial(
        strict_json_reward,
        valid_reward=cfg.format_valid_reward,
        invalid_reward=cfg.format_invalid_reward,
    )
    format_func.__name__ = "strict_json_reward"
    legal_func = partial(
        legal_move_reward,
        legal_reward=cfg.legal_reward,
        illegal_reward=cfg.illegal_reward,
    )
    legal_func.__name__ = "legal_move_reward"
    optimal_func = partial(
        optimal_move_reward,
        optimal_reward=cfg.optimal_reward,
        suboptimal_reward=cfg.suboptimal_reward,
        invalid_move_reward=cfg.invalid_move_reward,
    )
    optimal_func.__name__ = "optimal_move_reward"
    return [format_func, legal_func, optimal_func], list(cfg.weights)

