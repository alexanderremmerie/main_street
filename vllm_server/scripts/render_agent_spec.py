from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from common import endpoint_base_url, load_config


def build_agent_spec(config_path: str) -> dict[str, object]:
    cfg, _resolved = load_config(config_path)
    model_name = cfg.server.served_model_name or cfg.server.model
    return {
        "kind": "llm",
        "base_url": endpoint_base_url(cfg),
        "api_key_env": cfg.server.api_key_env,
        "model": model_name,
        "temperature": cfg.agent.temperature,
        "max_tokens": cfg.agent.max_tokens,
        "timeout_s": cfg.agent.timeout_s,
        "prompt_style": cfg.agent.prompt_style,
        "fallback": cfg.agent.fallback,
        "seed": cfg.agent.seed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vllm_server/scripts/render_agent_spec.py")
    parser.add_argument("--config", required=True, help="path to a stack config JSON file")
    parser.add_argument("--out", help="optional file to write instead of stdout")
    args = parser.parse_args(argv)

    payload = build_agent_spec(args.config)
    text = json.dumps(payload, indent=2) + "\n"
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text)
        print(f"wrote {out_path}", file=sys.stderr)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
