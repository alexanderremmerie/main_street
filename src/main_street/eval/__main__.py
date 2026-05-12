"""CLI for the eval harness.

  uv run python -m main_street.eval build <preset>           # starter|holdout|diagnostics
  uv run python -m main_street.eval score <set> <agent_kind> # default-spec'd baseline
  uv run python -m main_street.eval score <set> --spec <p>   # JSON AgentSpec
  uv run python -m main_street.eval score <set> --ckpt <p>   # AlphaZero from checkpoint
  uv run python -m main_street.eval list

Outputs land at `data/eval/<set>/` (built sets) and `data/eval/<set>/scores/<agent>.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from ..agents import KINDS, SPEC_TYPES, build
from .metrics import per_spec_agreement, score_agent
from .positions import DEFAULT_ROOT, PositionSet, assert_valid, build_position_set
from .sets import PRESETS


def cmd_build(args: argparse.Namespace) -> int:
    preset = args.preset
    if preset not in PRESETS:
        print(f"unknown preset: {preset}. choices: {list(PRESETS)}", file=sys.stderr)
        return 2
    sources = PRESETS[preset]
    print(f"building '{preset}' from {len(sources)} specs...")
    ps = build_position_set(name=preset, sources=sources)
    assert_valid(ps)
    out = ps.save(root=DEFAULT_ROOT)
    print(f"wrote {out}  ({len(ps)} positions, max_n={ps.max_n})")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    set_name: str = args.set
    ps = PositionSet.load(set_name, root=DEFAULT_ROOT)

    # Resolve the agent spec. Precedence: --ckpt > --spec > positional kind.
    if args.ckpt:
        from ..agents import AlphaZeroAgentSpec

        agent_spec = AlphaZeroAgentSpec(
            checkpoint_path=args.ckpt,
            n_simulations=args.n_sims,
        )
        agent_label = args.label or f"alphazero:{Path(args.ckpt).stem}"
    elif args.spec:
        spec_json = json.loads(Path(args.spec).read_text())
        kind = spec_json["kind"]
        spec_cls = SPEC_TYPES[kind]
        agent_spec = spec_cls.model_validate(spec_json)
        agent_label = args.label or f"{kind}:{Path(args.spec).stem}"
    else:
        kind = args.kind
        if kind not in SPEC_TYPES:
            print(f"unknown agent kind: {kind}. choices: {list(KINDS)}", file=sys.stderr)
            return 2
        if kind == "human":
            print("human agent cannot be scored offline", file=sys.stderr)
            return 2
        agent_spec = SPEC_TYPES[kind]()
        agent_label = args.label or kind

    agent = build(agent_spec)
    score = score_agent(agent, ps, agent_label=agent_label)
    print(f"{agent_label} on {set_name}: agreement={score.oracle_agreement:.4f} "
          f"({score.n_positions} positions)")
    worst = per_spec_agreement(score)[:5]
    if worst:
        print("  worst specs:")
        for spec, count, agr in worst:
            print(f"    {agr:.3f}  ({count:>5}) {spec}")
    if score.per_label:
        print("  diagnostics:")
        for lbl, ok in score.per_label.items():
            print(f"    {'OK ' if ok else 'XX '} {lbl}")

    out_dir = DEFAULT_ROOT / set_name / "scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{agent_label.replace('/', '_')}.json"
    payload = asdict(score)
    payload["agent_spec"] = agent_spec.model_dump()
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out_path}")
    return 0


def cmd_list(_args: argparse.Namespace) -> int:
    root = DEFAULT_ROOT
    if not root.exists():
        print("(no eval sets built yet)")
        return 0
    for d in sorted(root.iterdir()):
        manifest = d / "manifest.json"
        if not manifest.exists():
            continue
        m = json.loads(manifest.read_text())
        print(f"{m['name']:<16} count={m['count']:>6}  max_n={m['max_n']:>2}  "
              f"specs={len(m['specs'])}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="main_street.eval")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build a preset PositionSet")
    b.add_argument("preset", choices=list(PRESETS))
    b.set_defaults(func=cmd_build)

    s = sub.add_parser("score", help="score an agent on a PositionSet")
    s.add_argument("set")
    s.add_argument("kind", nargs="?", default="random", help="agent kind (default: random)")
    s.add_argument("--spec", help="path to a JSON agent spec (overrides kind)")
    s.add_argument(
        "--ckpt", help="path to a .pt checkpoint; builds AlphaZeroAgentSpec (overrides --spec)"
    )
    s.add_argument(
        "--n-sims",
        type=int,
        default=64,
        dest="n_sims",
        help="MCTS simulations (only used with --ckpt; default 64).",
    )
    s.add_argument("--label", help="label to record the score under")
    s.set_defaults(func=cmd_score)

    li = sub.add_parser("list", help="list built sets")
    li.set_defaults(func=cmd_list)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
