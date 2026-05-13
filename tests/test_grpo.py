from __future__ import annotations

import json

from main_street.core import GameSpec
from main_street.eval.positions import SourceSpec, build_position_set
from main_street.grpo.data import rows_from_position_set, write_jsonl
from main_street.grpo.rewards import (
    legal_move_reward,
    optimal_move_reward,
    strict_json_reward,
)


def test_rows_from_position_set_exports_prompt_and_metadata():
    ps = build_position_set(
        "tiny",
        [SourceSpec(spec=GameSpec(n=4, schedule=(1, 1)), mode="initial_only", label="root")],
    )
    rows = rows_from_position_set(ps)
    assert len(rows) == 1
    row = rows[0]
    assert row["source_set"] == "tiny"
    assert row["source_label"] == "root"
    assert row["board"] == "...."
    assert row["current_player"] == "X"
    assert row["legal_cells"] == [0, 1, 2, 3]
    assert row["optimal_cells"]
    assert row["reference_completion"] == json.dumps({"cell": max(row["optimal_cells"])})
    assert "legal_cells=[0,1,2,3]" in row["prompt"]


def test_write_jsonl_round_trips_rows(tmp_path):
    ps = build_position_set(
        "tiny",
        [SourceSpec(spec=GameSpec(n=4, schedule=(1, 1)), mode="initial_only")],
    )
    rows = rows_from_position_set(ps)
    path = write_jsonl(rows, tmp_path / "train.jsonl")
    loaded = [json.loads(line) for line in path.read_text().splitlines()]
    assert loaded == rows


def test_strict_json_reward_scores_valid_and_invalid():
    rewards = strict_json_reward(
        ['{"cell": 2}', '{"cell": "2"}', "not json"],
        valid_reward=0.25,
        invalid_reward=-0.5,
    )
    assert rewards == [0.25, -0.5, -0.5]


def test_legal_move_reward_scores_legality():
    rewards = legal_move_reward(
        ['{"cell": 2}', '{"cell": 9}', "oops"],
        legal_cells=[[1, 2], [0, 1], [3]],
        legal_reward=0.5,
        illegal_reward=-1.0,
    )
    assert rewards == [0.5, -1.0, -1.0]


def test_optimal_move_reward_scores_optimality():
    rewards = optimal_move_reward(
        ['{"cell": 2}', '{"cell": 1}', "oops"],
        optimal_cells=[[2, 3], [0], [1]],
        optimal_reward=1.0,
        suboptimal_reward=0.2,
        invalid_move_reward=-1.0,
    )
    assert rewards == [1.0, 0.2, -1.0]
