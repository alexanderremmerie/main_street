from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from typing import Any

from common import endpoint_base_url, load_config


def _request_json(
    url: str,
    payload: dict[str, Any] | None,
    api_key: str | None,
    timeout_s: float,
) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method="GET" if data is None else "POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vllm_server/scripts/smoke.py")
    parser.add_argument("--config", required=True, help="path to a stack config JSON file")
    parser.add_argument(
        "--prompt",
        default='Return JSON only: {"cell": 0}',
        help="user prompt to send to /chat/completions",
    )
    args = parser.parse_args(argv)

    cfg, _config_path = load_config(args.config)
    base_url = endpoint_base_url(cfg)
    model_name = cfg.server.served_model_name or cfg.server.model
    api_key = os.getenv(cfg.server.api_key_env)

    try:
        models = _request_json(f"{base_url}/models", payload=None, api_key=api_key, timeout_s=5.0)
        chat = _request_json(
            f"{base_url}/chat/completions",
            payload={
                "model": model_name,
                "messages": [
                    {"role": "system", "content": 'Return JSON only: {"cell": <int>}'},
                    {"role": "user", "content": args.prompt},
                ],
                "temperature": cfg.agent.temperature,
                "max_tokens": cfg.agent.max_tokens,
            },
            api_key=api_key,
            timeout_s=cfg.agent.timeout_s,
        )
    except urllib.error.URLError as e:
        print(f"smoke check failed: {e}")
        return 1

    print("models:")
    print(json.dumps(models, indent=2))
    print("\nchat completion:")
    print(json.dumps(chat, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
