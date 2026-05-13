from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    served_model_name: str | None = None
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    api_key_env: str = "VLLM_API_KEY"
    dtype: str | None = None
    gpu_memory_utilization: float | None = Field(default=None, gt=0.0, le=1.0)
    max_model_len: int | None = Field(default=None, gt=0)
    tensor_parallel_size: int = Field(default=1, gt=0)
    swap_space: int | None = Field(default=None, ge=0)
    trust_remote_code: bool = False
    enable_prefix_caching: bool = True
    extra_args: list[str] = Field(default_factory=list)


class AgentRuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    temperature: float = Field(default=0.0, ge=0.0)
    max_tokens: int = Field(default=16, gt=0)
    timeout_s: float = Field(default=30.0, gt=0.0)
    prompt_style: Literal["json_v1"] = "json_v1"
    fallback: Literal["rightmost_legal"] = "rightmost_legal"
    seed: int | None = None


class GRPOConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    trainer: str = "trl"
    trace_root: str = "data/llm_runs"
    export_root: str = "data/grpo_exports"
    prompt_dataset_name: str = "main_street_actions"
    reward_mode: str = "reserved"
    notes: str = ""


class VLLMStackConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    server: ServerConfig
    agent: AgentRuntimeConfig
    grpo: GRPOConfig = Field(default_factory=GRPOConfig)


def load_config(path: str | Path) -> tuple[VLLMStackConfig, Path]:
    config_path = Path(path).resolve()
    raw = json.loads(config_path.read_text())
    return VLLMStackConfig.model_validate(raw), config_path


def endpoint_base_url(cfg: VLLMStackConfig) -> str:
    return f"http://127.0.0.1:{cfg.server.port}/v1"

