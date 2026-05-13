from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from common import VLLMStackConfig, load_config


def build_command(cfg: VLLMStackConfig) -> list[str]:
    server = cfg.server
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        server.model,
        "--host",
        server.host,
        "--port",
        str(server.port),
        "--tensor-parallel-size",
        str(server.tensor_parallel_size),
    ]
    if server.served_model_name:
        cmd.extend(["--served-model-name", server.served_model_name])
    if server.dtype:
        cmd.extend(["--dtype", server.dtype])
    if server.gpu_memory_utilization is not None:
        cmd.extend(["--gpu-memory-utilization", str(server.gpu_memory_utilization)])
    if server.max_model_len is not None:
        cmd.extend(["--max-model-len", str(server.max_model_len)])
    if server.swap_space is not None:
        cmd.extend(["--swap-space", str(server.swap_space)])
    if server.trust_remote_code:
        cmd.append("--trust-remote-code")
    if server.enable_prefix_caching:
        cmd.append("--enable-prefix-caching")
    api_key = os.getenv(server.api_key_env)
    if api_key:
        cmd.extend(["--api-key", api_key])
    cmd.extend(server.extra_args)
    return cmd


def write_resolved_metadata(config_path: Path, cfg: VLLMStackConfig) -> Path:
    artifact_root = Path("data") / "vllm_server" / config_path.stem
    artifact_root.mkdir(parents=True, exist_ok=True)
    out_path = artifact_root / "resolved_stack_config.json"
    out_path.write_text(json.dumps(cfg.model_dump(mode="json"), indent=2))
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vllm_server/scripts/launch.py")
    parser.add_argument("--config", required=True, help="path to a stack config JSON file")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved command and exit without starting vLLM",
    )
    args = parser.parse_args(argv)

    cfg, config_path = load_config(args.config)
    cmd = build_command(cfg)
    metadata_path = write_resolved_metadata(config_path, cfg)
    print(f"wrote {metadata_path}")
    print("launch command:")
    print(" ".join(shlex.quote(part) for part in cmd))
    if args.dry_run:
        return 0
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
