import type { AgentSpec } from "../types";
import { NumberInput, Select, TextInput } from "./Form";

const MCTS_ROLLOUT_OPTIONS: { value: "random" | "forkaware"; label: string }[] = [
  { value: "random", label: "random" },
  { value: "forkaware", label: "forkaware" },
];

const KINDS: { value: AgentSpec["kind"]; label: string }[] = [
  { value: "human", label: "Human" },
  { value: "random", label: "Random" },
  { value: "rightmost", label: "Rightmost" },
  { value: "extension", label: "Extension" },
  { value: "blocker", label: "Blocker" },
  { value: "center", label: "Center" },
  { value: "greedy", label: "Greedy (1-ply)" },
  { value: "forkaware", label: "ForkAware" },
  { value: "potentialaware", label: "PotentialAware" },
  { value: "mcts", label: "MCTS (UCT)" },
  { value: "alphabeta", label: "Alpha-Beta" },
  { value: "alphazero", label: "AlphaZero" },
];

const KINDS_BOT_ONLY = KINDS.filter((k) => k.value !== "human");

export function AgentField({
  value,
  onChange,
  allowHuman = true,
}: {
  value: AgentSpec;
  onChange: (s: AgentSpec) => void;
  allowHuman?: boolean;
}) {
  const setKind = (k: AgentSpec["kind"]) => {
    switch (k) {
      case "human":
        onChange({ kind: "human" });
        break;
      case "random":
      case "greedy":
      case "forkaware":
      case "potentialaware":
        onChange({ kind: k, seed: 0 });
        break;
      case "rightmost":
      case "extension":
      case "blocker":
      case "center":
        onChange({ kind: k });
        break;
      case "mcts":
        onChange({ kind: "mcts", n_simulations: 200, seed: 0, rollout: "random" });
        break;
      case "alphabeta":
        onChange({ kind: "alphabeta", depth: 4 });
        break;
      case "alphazero":
        onChange({
          kind: "alphazero",
          checkpoint_path: "",
          n_simulations: 64,
          c_puct: 1.5,
          temperature: 0,
        });
        break;
    }
  };

  return (
    <div className="flex items-center gap-2 flex-wrap">
      <Select
        value={value.kind}
        options={allowHuman ? KINDS : KINDS_BOT_ONLY}
        onChange={setKind}
      />
      {(value.kind === "random" ||
        value.kind === "greedy" ||
        value.kind === "forkaware" ||
        value.kind === "potentialaware") && (
        <Param label="seed">
          <NumberInput
            value={value.seed ?? 0}
            onChange={(n) => onChange({ ...value, seed: n ?? 0 })}
          />
        </Param>
      )}
      {value.kind === "alphabeta" && (
        <Param label="depth" hint="empty = full search">
          <NumberInput
            value={value.depth ?? null}
            placeholder="full"
            onChange={(n) => onChange({ kind: "alphabeta", depth: n })}
          />
        </Param>
      )}
      {value.kind === "mcts" && (
        <>
          <Param label="sims">
            <NumberInput
              value={value.n_simulations ?? 200}
              onChange={(n) =>
                onChange({ ...value, n_simulations: n ?? 1 })
              }
            />
          </Param>
          <Param label="rollout">
            <Select
              value={value.rollout ?? "random"}
              options={MCTS_ROLLOUT_OPTIONS}
              onChange={(r) => onChange({ ...value, rollout: r })}
              width="w-32"
            />
          </Param>
          <Param label="seed">
            <NumberInput
              value={value.seed ?? 0}
              onChange={(n) => onChange({ ...value, seed: n ?? 0 })}
            />
          </Param>
        </>
      )}
      {value.kind === "alphazero" && (
        <>
          <Param label="ckpt">
            <TextInput
              value={value.checkpoint_path}
              placeholder="data/runs/.../final.pt"
              width="w-80"
              mono
              onChange={(s) => onChange({ ...value, checkpoint_path: s })}
            />
          </Param>
          <Param label="sims">
            <NumberInput
              value={value.n_simulations ?? 64}
              onChange={(n) => onChange({ ...value, n_simulations: n ?? 1 })}
            />
          </Param>
          <Param label="temp">
            <NumberInput
              value={value.temperature ?? 0}
              onChange={(n) =>
                onChange({ ...value, temperature: n ?? 0 })
              }
            />
          </Param>
        </>
      )}
    </div>
  );
}

function Param({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex items-center gap-1.5 text-xs text-neutral-500">
      <span>{label}</span>
      {children}
      {hint && <span className="text-neutral-400">{hint}</span>}
    </div>
  );
}
