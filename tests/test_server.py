import time
from pathlib import Path

from fastapi.testclient import TestClient

from main_street.nn.checkpoint import CheckpointMeta, save_checkpoint
from main_street.nn.encode import EncoderConfig, build_encoder
from main_street.nn.models import build_model
from main_street.server import create_app


def _client(tmp_path: Path) -> TestClient:
    app = create_app(db_path=tmp_path / "t.db")
    return TestClient(app)


def _wait_comparison(c: TestClient, comparison_id: str, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    last: dict | None = None
    while time.monotonic() < deadline:
        last = c.get(f"/api/comparisons/{comparison_id}").json()
        if last["status"] not in ("running", "cancelling"):
            return last
        time.sleep(0.05)
    raise AssertionError(f"comparison did not finish: {last}")


def _write_test_checkpoint(path: Path) -> None:
    cfg = EncoderConfig(max_n=12, max_turns=6)
    encoder = build_encoder(cfg)
    params = {"channels": 16, "n_blocks": 2}
    model = build_model("simple_conv", encoder, params)
    save_checkpoint(
        path,
        model=model,
        model_name="simple_conv",
        model_params=params,
        encoder_config=cfg,
        meta=CheckpointMeta(run_id="test", iter=0),
    )


def test_agents_lists_all_kinds(tmp_path: Path):
    c = _client(tmp_path)
    r = c.get("/api/agents")
    assert r.status_code == 200
    kinds = {a["kind"] for a in r.json()}
    assert kinds == {
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
        "alphazero",
    }


def test_move_returns_legal_cell(tmp_path: Path):
    c = _client(tmp_path)
    r = c.post(
        "/api/move",
        json={
            "spec": {"n": 6, "schedule": [1, 1, 1]},
            "actions": [],
            "agent": {"kind": "rightmost"},
        },
    )
    assert r.status_code == 200, r.text
    assert r.json()["cell"] == 5  # rightmost first


def test_move_rejects_terminal_state(tmp_path: Path):
    c = _client(tmp_path)
    r = c.post(
        "/api/move",
        json={
            "spec": {"n": 3, "schedule": [1, 1, 1]},
            "actions": [0, 1, 2],
            "agent": {"kind": "rightmost"},
        },
    )
    assert r.status_code == 400


def test_move_rejects_human_agent(tmp_path: Path):
    c = _client(tmp_path)
    r = c.post(
        "/api/move",
        json={
            "spec": {"n": 3, "schedule": [1, 1, 1]},
            "actions": [],
            "agent": {"kind": "human"},
        },
    )
    assert r.status_code == 400


def test_move_rejects_illegal_action_history(tmp_path: Path):
    c = _client(tmp_path)
    r = c.post(
        "/api/move",
        json={
            "spec": {"n": 4, "schedule": [1, 1, 1, 1]},
            "actions": [0, 0],  # cell 0 played twice
            "agent": {"kind": "rightmost"},
        },
    )
    assert r.status_code == 400


def test_inspect_model_returns_policy_and_search_stats(tmp_path: Path):
    c = _client(tmp_path)
    ckpt = tmp_path / "model.pt"
    _write_test_checkpoint(ckpt)
    created = c.post(
        "/api/players",
        json={
            "label": "test az",
            "agent_spec": {
                "kind": "alphazero",
                "checkpoint_path": str(ckpt),
                "n_simulations": 8,
                "c_puct": 1.5,
                "temperature": 0.0,
            },
        },
    ).json()

    r = c.post(
        "/api/inspect-model",
        json={
            "spec": {"n": 6, "schedule": [2, 3]},
            "actions": [],
            "player_id": created["id"],
            "n_simulations": 8,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["player_id"] == created["id"]
    assert body["current_player"] == 1
    assert isinstance(body["raw_value"], float)
    assert 0 <= body["puct_action"] < 6
    assert len(body["moves"]) == 6
    assert abs(sum(m["raw_policy"] for m in body["moves"]) - 1.0) < 1e-5
    assert all(0 <= m["puct_visit_prob"] <= 1 for m in body["moves"])


def test_inspect_model_rejects_non_model_player(tmp_path: Path):
    c = _client(tmp_path)
    r = c.post(
        "/api/inspect-model",
        json={
            "spec": {"n": 6, "schedule": [2, 3]},
            "actions": [],
            "player_id": "p_greedy_0",
            "n_simulations": 8,
        },
    )
    assert r.status_code == 400


def test_save_persists_completed_game(tmp_path: Path):
    c = _client(tmp_path)
    payload = {
        "spec": {"n": 4, "schedule": [1, 1, 1, 1]},
        "x_agent": {"kind": "human"},
        "o_agent": {"kind": "rightmost"},
        "actions": [0, 3, 1, 2],
    }
    r = c.post("/api/games", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["outcome"] in (-1, 0, 1)
    assert body["actions"] == [0, 3, 1, 2]

    # And it shows up in the listing.
    r = c.get("/api/games")
    assert r.json()["total"] == 1


def test_save_rejects_incomplete_game(tmp_path: Path):
    c = _client(tmp_path)
    r = c.post(
        "/api/games",
        json={
            "spec": {"n": 4, "schedule": [1, 1, 1, 1]},
            "x_agent": {"kind": "human"},
            "o_agent": {"kind": "rightmost"},
            "actions": [0, 3],
        },
    )
    assert r.status_code == 400


def test_save_rejects_illegal_action(tmp_path: Path):
    c = _client(tmp_path)
    r = c.post(
        "/api/games",
        json={
            "spec": {"n": 4, "schedule": [1, 1, 1, 1]},
            "x_agent": {"kind": "human"},
            "o_agent": {"kind": "rightmost"},
            "actions": [0, 0, 1, 2],
        },
    )
    assert r.status_code == 400


def test_oracle_returns_value_and_per_cell(tmp_path: Path):
    c = _client(tmp_path)
    r = c.post(
        "/api/oracle",
        json={"spec": {"n": 3, "schedule": [1, 1, 1]}, "actions": []},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["value"] in (-1, 1)
    assert body["is_terminal"] is False
    # Keys come back as JSON strings; values are ±1.
    assert {int(k) for k in body["per_cell_values"]} == {0, 1, 2}
    assert all(v in (-1, 1) for v in body["per_cell_values"].values())
    assert body["best_cell"] in (0, 1, 2)


def test_oracle_terminal_state(tmp_path: Path):
    c = _client(tmp_path)
    r = c.post(
        "/api/oracle",
        json={"spec": {"n": 3, "schedule": [1, 1, 1]}, "actions": [0, 1, 2]},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["is_terminal"] is True
    assert body["best_cell"] == -1
    assert body["per_cell_values"] == {}


def test_oracle_refuses_oversized_spec(tmp_path: Path):
    c = _client(tmp_path)
    # 20-cell board, lots of placements -> enormous state space.
    r = c.post(
        "/api/oracle",
        json={"spec": {"n": 20, "schedule": [3, 3, 3, 3, 3]}, "actions": []},
    )
    assert r.status_code == 413, r.text


def test_oracle_rejects_illegal_actions(tmp_path: Path):
    c = _client(tmp_path)
    r = c.post(
        "/api/oracle",
        json={"spec": {"n": 3, "schedule": [1, 1]}, "actions": [0, 0]},
    )
    assert r.status_code == 400


def test_specs_aggregates_played_specs(tmp_path: Path):
    c = _client(tmp_path)
    # No games yet -> empty list.
    assert c.get("/api/specs").json() == []

    spec_small = {"n": 4, "schedule": [1, 1, 1, 1]}
    c.post(
        "/api/games",
        json={
            "spec": spec_small,
            "x_agent": {"kind": "human"},
            "o_agent": {"kind": "rightmost"},
            "actions": [0, 3, 1, 2],
        },
    )
    body = c.get("/api/specs").json()
    assert len(body) == 1
    row = body[0]
    assert row["spec"] == {"n": 4, "schedule": [1, 1, 1, 1]}
    assert row["n_games"] == 1
    assert row["x_wins"] + row["o_wins"] + row["ties"] == 1


def test_players_default_seed(tmp_path: Path):
    c = _client(tmp_path)
    r = c.get("/api/players")
    assert r.status_code == 200
    players = r.json()
    labels = {p["label"] for p in players}
    assert labels >= {"Random", "Greedy", "Rightmost", "AlphaBeta (exact)", "AlphaBeta (depth=4)"}
    assert all(p["is_default"] for p in players)


def test_players_seed_is_idempotent(tmp_path: Path):
    # Connecting twice to the same DB must not duplicate or fail.
    c = _client(tmp_path)
    first = c.get("/api/players").json()
    c2 = _client(tmp_path)
    second = c2.get("/api/players").json()
    assert {p["id"] for p in first} == {p["id"] for p in second}


def test_create_custom_player(tmp_path: Path):
    c = _client(tmp_path)
    r = c.post(
        "/api/players",
        json={"label": "Greedy seed=7", "agent_spec": {"kind": "greedy", "seed": 7}},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["is_default"] is False
    assert body["label"] == "Greedy seed=7"
    assert body["id"].startswith("p_")


def test_create_player_rejects_duplicate_label(tmp_path: Path):
    c = _client(tmp_path)
    r = c.post(
        "/api/players",
        json={"label": "Random", "agent_spec": {"kind": "random", "seed": 1}},
    )
    assert r.status_code == 409


def test_create_player_rejects_human(tmp_path: Path):
    c = _client(tmp_path)
    r = c.post(
        "/api/players",
        json={"label": "me", "agent_spec": {"kind": "human"}},
    )
    assert r.status_code == 400


def test_delete_default_player_forbidden(tmp_path: Path):
    c = _client(tmp_path)
    r = c.delete("/api/players/p_random_0")
    assert r.status_code == 400


def test_delete_custom_player(tmp_path: Path):
    c = _client(tmp_path)
    created = c.post(
        "/api/players",
        json={"label": "todelete", "agent_spec": {"kind": "rightmost"}},
    ).json()
    r = c.delete(f"/api/players/{created['id']}")
    assert r.status_code == 200
    assert c.get("/api/players").json().count(created) == 0


def test_analyze_returns_verdict_per_player(tmp_path: Path):
    c = _client(tmp_path)
    r = c.post(
        "/api/analyze",
        json={
            "spec": {"n": 4, "schedule": [1, 1, 1, 1]},
            "actions": [],
            "player_ids": ["p_rightmost", "p_alphabeta_exact"],
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    by_id = {v["player_id"]: v for v in body["verdicts"]}
    assert by_id["p_rightmost"]["cell"] == 3
    assert by_id["p_alphabeta_exact"]["cell"] in (0, 1, 2, 3)
    # 4-cell spec is small enough for the oracle.
    assert body["oracle_value"] in (-1, 1)
    assert by_id["p_alphabeta_exact"]["agrees_with_oracle"] is True


def test_analyze_terminal_position_rejected(tmp_path: Path):
    c = _client(tmp_path)
    r = c.post(
        "/api/analyze",
        json={
            "spec": {"n": 3, "schedule": [1, 1, 1]},
            "actions": [0, 1, 2],
            "player_ids": ["p_rightmost"],
        },
    )
    assert r.status_code == 400


def test_analyze_missing_player_marks_error(tmp_path: Path):
    c = _client(tmp_path)
    r = c.post(
        "/api/analyze",
        json={
            "spec": {"n": 3, "schedule": [1, 1]},
            "actions": [],
            "player_ids": ["p_does_not_exist"],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["verdicts"][0]["error"] == "player not found"
    assert body["verdicts"][0]["cell"] is None


def test_analyze_without_oracle_on_large_spec(tmp_path: Path):
    c = _client(tmp_path)
    r = c.post(
        "/api/analyze",
        json={
            "spec": {"n": 20, "schedule": [3, 3, 3, 3, 3]},
            "actions": [],
            "player_ids": ["p_rightmost"],
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["oracle_value"] is None
    assert body["verdicts"][0]["cell"] == 19
    assert body["verdicts"][0]["agrees_with_oracle"] is None


def test_comparison_basic_matrix(tmp_path: Path):
    c = _client(tmp_path)
    r = c.post(
        "/api/comparisons",
        json={
            "player_ids": ["p_rightmost", "p_greedy_0", "p_random_0"],
            "specs": [{"n": 5, "schedule": [1, 1, 1]}],
            "n_games_per_spec": 4,
            "swap_sides": True,
            "seed": 0,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "running"
    assert body["progress_done"] <= body["progress_total"] == 12
    body = _wait_comparison(c, body["id"])
    assert body["status"] == "done"
    assert body["progress_done"] == body["progress_total"] == 12
    pairs = body["summary"]["pairs"]
    assert len(pairs) == 3  # C(3, 2)
    pair_ids = {tuple(sorted([p["a_player_id"], p["b_player_id"]])) for p in pairs}
    assert pair_ids == {
        tuple(sorted(["p_rightmost", "p_greedy_0"])),
        tuple(sorted(["p_rightmost", "p_random_0"])),
        tuple(sorted(["p_greedy_0", "p_random_0"])),
    }
    for p in pairs:
        assert 0.0 <= p["a_winrate"] <= 1.0
        assert p["n_games"] == 4
        # Underlying eval was persisted.
        assert c.get(f"/api/evals/{p['eval_id']}").status_code == 200


def test_sample_specs_endpoint(tmp_path: Path):
    c = _client(tmp_path)
    r = c.post(
        "/api/specs/sample",
        json={
            "count": 12,
            "seed": 7,
            "sampler": {
                "kind": "mixture",
                "n_min": 6,
                "n_max": 8,
                "turns_min": 2,
                "turns_max": 4,
                "fill_min": 0.5,
                "fill_max": 0.9,
                "max_marks_per_turn": 4,
            },
        },
    )
    assert r.status_code == 200, r.text
    specs = r.json()
    assert len(specs) == 12
    assert len({(s["n"], tuple(s["schedule"])) for s in specs}) == 12
    for spec in specs:
        assert 6 <= spec["n"] <= 8
        assert 2 <= len(spec["schedule"]) <= 4
        assert sum(spec["schedule"]) <= spec["n"]


def test_comparison_sampled_specs(tmp_path: Path):
    c = _client(tmp_path)
    r = c.post(
        "/api/comparisons",
        json={
            "player_ids": ["p_rightmost", "p_random_0"],
            "specs": [],
            "spec_sampler": {
                "kind": "mixture",
                "n_min": 5,
                "n_max": 6,
                "turns_min": 2,
                "turns_max": 3,
                "fill_min": 0.5,
                "fill_max": 0.8,
                "max_marks_per_turn": 3,
            },
            "n_sampled_specs": 3,
            "n_games_per_spec": 2,
            "swap_sides": True,
            "seed": 11,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["progress_total"] == 6
    body = _wait_comparison(c, body["id"])
    assert body["progress_done"] == body["progress_total"] == 6
    assert len(body["config"]["specs"]) == 3
    assert body["config"]["spec_sampler"]["n_min"] == 5
    assert body["summary"]["n_total_games"] == 6
    pair = body["summary"]["pairs"][0]
    ev = c.get(f"/api/evals/{pair['eval_id']}").json()
    assert ev["config"]["specs"] == body["config"]["specs"]


def test_comparison_requires_two_players(tmp_path: Path):
    c = _client(tmp_path)
    r = c.post(
        "/api/comparisons",
        json={
            "player_ids": ["p_rightmost"],
            "specs": [{"n": 4, "schedule": [1, 1]}],
            "n_games_per_spec": 2,
        },
    )
    assert r.status_code == 422  # pydantic min_length validation


def test_comparison_rejects_unknown_player(tmp_path: Path):
    c = _client(tmp_path)
    r = c.post(
        "/api/comparisons",
        json={
            "player_ids": ["p_rightmost", "p_does_not_exist"],
            "specs": [{"n": 4, "schedule": [1, 1]}],
            "n_games_per_spec": 2,
        },
    )
    assert r.status_code == 400


def test_comparison_rejects_duplicate_players(tmp_path: Path):
    c = _client(tmp_path)
    r = c.post(
        "/api/comparisons",
        json={
            "player_ids": ["p_rightmost", "p_rightmost"],
            "specs": [{"n": 4, "schedule": [1, 1]}],
            "n_games_per_spec": 2,
        },
    )
    assert r.status_code == 400


def test_cancel_comparison(tmp_path: Path):
    c = _client(tmp_path)
    created = c.post(
        "/api/comparisons",
        json={
            "player_ids": ["p_rightmost", "p_random_0"],
            "specs": [{"n": 6, "schedule": [1, 1, 1, 1, 1, 1]}],
            "n_games_per_spec": 200,
            "swap_sides": True,
            "seed": 0,
        },
    ).json()
    r = c.post(f"/api/comparisons/{created['id']}/cancel")
    assert r.status_code == 200, r.text
    assert r.json()["status"] in ("cancelling", "cancelled")
    final = _wait_comparison(c, created["id"])
    assert final["status"] in ("cancelled", "done")


def test_comparison_listing(tmp_path: Path):
    c = _client(tmp_path)
    assert c.get("/api/comparisons").json() == []
    created = c.post(
        "/api/comparisons",
        json={
            "player_ids": ["p_rightmost", "p_random_0"],
            "specs": [{"n": 4, "schedule": [1, 1]}],
            "n_games_per_spec": 2,
        },
    ).json()
    rows = c.get("/api/comparisons").json()
    assert len(rows) == 1
    assert rows[0]["id"] == created["id"]
    detail = _wait_comparison(c, created["id"])
    assert detail["summary"]["n_total_games"] == 2  # 1 pair × 1 spec × 2 games


def test_eval_rejects_human_agent(tmp_path: Path):
    c = _client(tmp_path)
    r = c.post(
        "/api/evals",
        json={
            "agent_a": {"kind": "human"},
            "agent_b": {"kind": "rightmost"},
            "specs": [{"n": 4, "schedule": [1, 1, 1, 1]}],
            "n_games_per_spec": 2,
            "swap_sides": True,
            "seed": 0,
        },
    )
    assert r.status_code == 400
