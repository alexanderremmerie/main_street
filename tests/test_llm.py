from __future__ import annotations

import json

from main_street.agents import LLMAgentSpec, build
from main_street.core import GameSpec, GameState, step
from main_street.llm import parse_json_cell_reply, render_prompt
from tests._llm_test_utils import llm_server, openai_chat_response


def test_render_prompt_serializes_state_deterministically():
    spec = GameSpec(n=6, schedule=(1, 2, 1))
    state = GameState.initial(spec)
    state = step(state, 4)  # X
    state = step(state, 1)  # O
    prompt = render_prompt(state)
    assert prompt == "\n".join(
        (
            "Choose one legal move in this 1D placement game.",
            'Return JSON only: {"cell": <int>}',
            "n=6",
            "schedule=[1,2,1]",
            "turn_idx=1",
            "placements_left=1",
            "current_player=O",
            "board=.O..X.",
            "legal_cells=[0,2,3,5]",
        )
    )


def test_parse_json_cell_reply_accepts_valid_payload():
    assert parse_json_cell_reply('{"cell": 7}') == 7


def test_parse_json_cell_reply_rejects_malformed_json():
    try:
        parse_json_cell_reply("{cell: 7}")
    except ValueError as e:
        assert "not valid JSON" in str(e)
    else:
        raise AssertionError("expected malformed JSON to fail")


def test_parse_json_cell_reply_rejects_non_integer_cell():
    try:
        parse_json_cell_reply(json.dumps({"cell": "3"}))
    except ValueError as e:
        assert "integer" in str(e)
    else:
        raise AssertionError("expected non-integer cell to fail")


def test_llm_agent_falls_back_on_illegal_cell(monkeypatch):
    monkeypatch.setenv("TEST_LLM_KEY", "dummy")

    def responder(
        _path: str, _payload: dict[str, object]
    ) -> tuple[int, dict[str, object], float | None]:
        return 200, openai_chat_response('{"cell": 99}'), None

    with llm_server(responder) as (base_url, requests):
        agent = build(
            LLMAgentSpec(
                base_url=base_url,
                api_key_env="TEST_LLM_KEY",
                model="test-model",
                temperature=0.0,
                max_tokens=16,
                timeout_s=1.0,
            )
        )
        cell = agent.act(GameState.initial(GameSpec(n=5, schedule=(1, 1, 1))))
    assert cell == 4
    assert requests[0]["path"] == "/v1/chat/completions"


def test_llm_agent_falls_back_on_timeout(monkeypatch):
    monkeypatch.setenv("TEST_LLM_KEY", "dummy")

    def responder(
        _path: str, _payload: dict[str, object]
    ) -> tuple[int, dict[str, object], float | None]:
        return 200, openai_chat_response('{"cell": 1}'), 0.2

    with llm_server(responder) as (base_url, _requests):
        agent = build(
            LLMAgentSpec(
                base_url=base_url,
                api_key_env="TEST_LLM_KEY",
                model="test-model",
                temperature=0.0,
                max_tokens=16,
                timeout_s=0.05,
            )
        )
        cell = agent.act(GameState.initial(GameSpec(n=5, schedule=(1, 1, 1))))
    assert cell == 4
