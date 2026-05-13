from __future__ import annotations

import json
from pathlib import Path
from random import Random
from typing import Any

from ..core import GameState, legal_actions
from ..eval.positions import PositionSet
from ..eval.sets import PRESETS
from ..llm import board_to_string, render_prompt


def _reference_cell(optimal_cells: list[int]) -> int:
    if not optimal_cells:
        raise ValueError("optimal_cells must be non-empty")
    return max(optimal_cells)


def row_from_state(
    state: GameState,
    *,
    optimal_cells: list[int],
    oracle_value: int,
    source_set: str,
    source_label: str = "",
    prompt_style: str = "json_v1",
) -> dict[str, Any]:
    legal = [int(cell) for cell in legal_actions(state).tolist()]
    reference_cell = _reference_cell(optimal_cells)
    return {
        "prompt": render_prompt(state, prompt_style=prompt_style),
        "prompt_style": prompt_style,
        "n": state.spec.n,
        "schedule": list(state.spec.schedule),
        "turn_idx": state.turn_idx,
        "placements_left": state.placements_left,
        "current_player": "X" if int(state.current_player) == 1 else "O",
        "board": board_to_string(state),
        "legal_cells": legal,
        "optimal_cells": list(optimal_cells),
        "oracle_value": int(oracle_value),
        "source_set": source_set,
        "source_label": source_label,
        "reference_completion": json.dumps({"cell": reference_cell}),
    }


def rows_from_position_set(
    ps: PositionSet, *, prompt_style: str = "json_v1"
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i in range(len(ps)):
        rows.append(
            row_from_state(
                ps.state(i),
                optimal_cells=[int(cell) for cell in ps.optimal_cells(i).tolist()],
                oracle_value=int(ps.value[i]),
                source_set=ps.name,
                source_label=ps.labels[i],
                prompt_style=prompt_style,
            )
        )
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True))
            f.write("\n")
    return out


def export_position_set(
    ps: PositionSet,
    *,
    out_path: str | Path,
    prompt_style: str = "json_v1",
    limit: int | None = None,
    seed: int = 0,
) -> Path:
    rows = rows_from_position_set(ps, prompt_style=prompt_style)
    if limit is not None and limit < len(rows):
        rng = Random(seed)
        rng.shuffle(rows)
        rows = rows[:limit]
    return write_jsonl(rows, out_path)


def build_preset_dataset(
    preset: str,
    *,
    out_path: str | Path,
    prompt_style: str = "json_v1",
    limit: int | None = None,
    seed: int = 0,
) -> Path:
    if preset not in PRESETS:
        raise ValueError(f"unknown preset: {preset}. choices: {sorted(PRESETS)}")
    from ..eval.positions import build_position_set

    ps = build_position_set(name=preset, sources=PRESETS[preset])
    return export_position_set(
        ps,
        out_path=out_path,
        prompt_style=prompt_style,
        limit=limit,
        seed=seed,
    )
