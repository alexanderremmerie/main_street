# vLLM Server Scaffold

This folder is a small, config-driven serving scaffold for running an OpenAI-compatible
`vLLM` endpoint alongside the `main_street` repo.

It is intentionally split into three config sections:

- `server`: how to launch `vLLM`
- `agent`: how `main_street` should talk to that endpoint
- `grpo`: reserved metadata for future trace export / GRPO jobs

The goal is to keep the first baseline easy to run now without painting us into a
corner when we later add:

- trace-to-dataset export
- SFT warm starts
- GRPO training jobs
- multiple served model variants

## Files

- `configs/qwen3_8b_local.example.json`: default local serve/train config for a 4090 or Modal
- `configs/qwen3_0p6b_local.example.json`: optional smaller smoke-only config for very cheap iteration
- `configs/grpo.dev.example.json`: example showing how to reserve training/export metadata
- `scripts/launch.py`: launches `vLLM` from a config file
- `scripts/smoke.py`: checks `/v1/models` and `/v1/chat/completions`
- `scripts/render_agent_spec.py`: emits a `main_street` `LLMAgentSpec` JSON blob

## Quick Start

1. Copy an example config and edit the `model` path/name:

```bash
cp vllm_server/configs/qwen3_8b_local.example.json \
  vllm_server/configs/local.json
```

2. Optionally set an API key for the server:

```bash
export VLLM_API_KEY=dev-token
```

3. Launch the server:

```bash
uv run python vllm_server/scripts/launch.py \
  --config vllm_server/configs/local.json
```

4. Smoke-test it:

```bash
uv run python vllm_server/scripts/smoke.py \
  --config vllm_server/configs/local.json
```

5. Emit an agent spec for `main_street`:

```bash
uv run python vllm_server/scripts/render_agent_spec.py \
  --config vllm_server/configs/local.json
```

## Using With `main_street`

The rendered JSON can be used directly with the repo's `llm` agent kind, for example:

```bash
uv run python -m main_street.llm \
  --spec spec.json \
  --agent-spec llm_agent.json
```

Or via the app/server by creating a player whose `agent_spec.kind` is `llm`.

## GRPO Expansion Path

This folder does not implement GRPO yet. It reserves the wiring we will likely need:

- `grpo.trace_root`: where online traces are collected from the game stack
- `grpo.export_root`: where converted datasets can land
- `grpo.prompt_dataset_name`: stable dataset naming for train/eval splits
- `grpo.reward_mode`: placeholder for later reward definitions

That lets us grow from:

1. `vLLM` serving
2. baseline evals
3. trace export
4. SFT warm start
5. GRPO loops

without replacing the config contract again.

## Model Choice

The default target in this repo is `Qwen/Qwen3-8B`.

Why:

- Main Street is text-only right now, so a plain text model is the correct baseline
- `Qwen3-8B` is a realistic first model for `4090 -> LoRA/GRPO -> vLLM re-serve`
- the same setup is easy to move to Modal later

Use `Qwen/Qwen3-VL-8B-Instruct` only if the task becomes multimodal, for example if
the policy should read screenshots or rendered boards rather than structured text prompts.

The `qwen3_0p6b_local.example.json` file is retained only as a cheap smoke-test option if
you want a much lighter train/eval loop while debugging reward functions or data export.
