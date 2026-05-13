"""LLM prompt, parsing, tracing, and smoke-test CLI helpers."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .core import EMPTY, GameSpec, GameState, O, X, final_state, legal_actions

DEFAULT_TRACE_ROOT = Path("data") / "llm_runs"
TRACE_ROOT_ENV = "MAIN_STREET_LLM_TRACE_ROOT"
_TRACE_CONTEXT: ContextVar[LLMTraceContext | None] = ContextVar(
    "main_street_llm_trace_context", default=None
)


class LLMRequestError(RuntimeError):
    """Transport- or response-level failure from the OpenAI-compatible endpoint."""


@dataclass(frozen=True, slots=True)
class LLMTraceContext:
    run_id: str
    source: str
    move_index: int
    game_id: str | None = None
    eval_id: str | None = None
    actor_id: str | None = None
    actor_label: str | None = None


def resolve_trace_root() -> Path:
    override = os.getenv(TRACE_ROOT_ENV)
    return Path(override) if override else DEFAULT_TRACE_ROOT


def current_trace_context() -> LLMTraceContext | None:
    return _TRACE_CONTEXT.get()


@contextmanager
def use_trace_context(ctx: LLMTraceContext) -> Any:
    token: Token[LLMTraceContext | None] = _TRACE_CONTEXT.set(ctx)
    try:
        yield
    finally:
        _TRACE_CONTEXT.reset(token)


def board_to_string(state: GameState) -> str:
    chars = []
    for cell in state.board.tolist():
        if cell == int(EMPTY):
            chars.append(".")
        elif cell == int(X):
            chars.append("X")
        elif cell == int(O):
            chars.append("O")
        else:
            raise ValueError(f"unexpected board value: {cell}")
    return "".join(chars)


def render_prompt(state: GameState, *, prompt_style: str = "json_v1") -> str:
    if prompt_style != "json_v1":
        raise ValueError(f"unsupported prompt style: {prompt_style}")
    current = "X" if int(state.current_player) == int(X) else "O"
    schedule = ",".join(str(k) for k in state.spec.schedule)
    legal = ",".join(str(int(cell)) for cell in legal_actions(state).tolist())
    return "\n".join(
        (
            "Choose one legal move in this 1D placement game.",
            'Return JSON only: {"cell": <int>}',
            f"n={state.spec.n}",
            f"schedule=[{schedule}]",
            f"turn_idx={state.turn_idx}",
            f"placements_left={state.placements_left}",
            f"current_player={current}",
            f"board={board_to_string(state)}",
            f"legal_cells=[{legal}]",
        )
    )


def parse_json_cell_reply(text: str) -> int:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"response is not valid JSON: {e.msg}") from e
    if not isinstance(payload, dict):
        raise ValueError("response must be a JSON object")
    if set(payload) != {"cell"}:
        raise ValueError("response must have exactly one key: cell")
    cell = payload["cell"]
    if isinstance(cell, bool) or not isinstance(cell, int):
        raise ValueError("cell must be an integer")
    return int(cell)


def _extract_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("missing choices in response payload")
    first = choices[0]
    if not isinstance(first, dict):
        raise ValueError("malformed first choice in response payload")
    if "message" in first:
        message = first["message"]
        if not isinstance(message, dict):
            raise ValueError("message must be an object")
        content = message.get("content")
    else:
        content = first.get("text")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                chunks.append(part["text"])
        if chunks:
            return "".join(chunks)
    raise ValueError("response payload does not contain assistant text")


def request_chat_completion(
    *,
    base_url: str,
    api_key_env: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout_s: float,
    seed: int | None,
) -> str:
    base = base_url.rstrip("/") + "/"
    url = urllib.parse.urljoin(base, "chat/completions")
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": 'Return JSON only: {"cell": <int>}'},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if seed is not None:
        payload["seed"] = seed

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    api_key = os.getenv(api_key_env)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw_body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise LLMRequestError(f"HTTP {e.code}: {body}") from e
    except TimeoutError as e:
        raise LLMRequestError("request timed out") from e
    except urllib.error.URLError as e:
        reason = e.reason if getattr(e, "reason", None) else str(e)
        raise LLMRequestError(f"request failed: {reason}") from e
    except OSError as e:
        raise LLMRequestError(f"request failed: {e}") from e

    try:
        response_payload = json.loads(raw_body)
    except json.JSONDecodeError as e:
        raise LLMRequestError(f"response was not JSON: {e.msg}") from e
    try:
        return _extract_content(response_payload)
    except ValueError as e:
        raise LLMRequestError(str(e)) from e


def append_trace(entry: dict[str, Any]) -> Path | None:
    ctx = current_trace_context()
    if ctx is None:
        return None
    path = resolve_trace_root() / ctx.source / f"{ctx.run_id}.jsonl"
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        **asdict(ctx),
        **entry,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True))
        f.write("\n")
    return path


def _parse_actions(text: str) -> tuple[int, ...]:
    if not text.strip():
        return ()
    try:
        return tuple(int(part) for part in text.split(",") if part.strip())
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"invalid actions: {e}") from e


def _load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _cmd_smoke(args: argparse.Namespace) -> int:
    try:
        spec = GameSpec.model_validate(_load_json(args.spec))
        from .agents import LLMAgentSpec, build

        agent_spec = LLMAgentSpec.model_validate(_load_json(args.agent_spec))
    except (OSError, ValidationError, json.JSONDecodeError) as e:
        print(f"invalid input: {e}", file=sys.stderr)
        return 2

    try:
        state = final_state(spec, _parse_actions(args.actions))
    except ValueError as e:
        print(f"invalid action sequence: {e}", file=sys.stderr)
        return 2
    if state.is_terminal:
        print("position is terminal", file=sys.stderr)
        return 2

    prompt = render_prompt(state, prompt_style=agent_spec.prompt_style)
    if args.print_prompt:
        print(prompt)

    run_id = args.run_id or uuid.uuid4().hex
    agent = build(agent_spec)
    with use_trace_context(
        LLMTraceContext(
            run_id=run_id,
            source="cli",
            move_index=len(_parse_actions(args.actions)),
            actor_id="cli",
            actor_label="cli",
        )
    ):
        cell = int(agent.act(state))
    print(json.dumps({"cell": cell, "trace_run_id": run_id}))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m main_street.llm")
    parser.add_argument("--spec", required=True, help="path to a JSON GameSpec")
    parser.add_argument("--agent-spec", required=True, help="path to a JSON LLMAgentSpec")
    parser.add_argument(
        "--actions",
        default="",
        help="comma-separated action history, e.g. 4,0,3",
    )
    parser.add_argument("--run-id", help="trace run id; defaults to a random uuid")
    parser.add_argument(
        "--print-prompt",
        action="store_true",
        help="print the rendered prompt before requesting a move",
    )
    args = parser.parse_args(argv)
    return int(_cmd_smoke(args))


if __name__ == "__main__":
    sys.exit(main())
