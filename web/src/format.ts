import type { AgentSpec, ComparisonRecord, GameSpec } from "./types";

export function formatSpec(spec: GameSpec): string {
  return `N=${spec.n} [${spec.schedule.join(",")}]`;
}

export type Progress = { done: number; total: number; pct: number };

export function comparisonProgress(rec: ComparisonRecord): Progress {
  const done = rec.summary?.n_total_games ?? rec.progress_done;
  const total = rec.progress_total || rec.summary?.n_total_games || 0;
  const pct = total > 0 ? Math.min(100, Math.round((done / total) * 100)) : 0;
  return { done, total, pct };
}

// `seed: null` and `seed: 0` are different runtime behaviors (nondeterministic
// vs fixed). Render them distinctly — printing "s=0" for a null-seed agent
// silently misrepresents what the engine will do.
function formatSeed(seed: number | null | undefined): string {
  return seed == null ? "s=random" : `s=${seed}`;
}

export function formatAgent(spec: AgentSpec): string {
  switch (spec.kind) {
    case "human":
      return "human";
    case "random":
      return `random ${formatSeed(spec.seed)}`;
    case "greedy":
      return `greedy ${formatSeed(spec.seed)}`;
    case "rightmost":
      return "rightmost";
    case "alphabeta":
      return spec.depth == null ? "alphabeta full" : `alphabeta d=${spec.depth}`;
    case "extension":
      return "extension";
    case "blocker":
      return "blocker";
    case "center":
      return "center";
    case "forkaware":
      return `forkaware ${formatSeed(spec.seed)}`;
    case "potentialaware":
      return `potentialaware ${formatSeed(spec.seed)}`;
    case "mcts": {
      const r = spec.rollout ?? "random";
      return `mcts sims=${spec.n_simulations ?? 200} ${r} ${formatSeed(spec.seed)}`;
    }
    case "alphazero": {
      // Show only the trailing path component so the label stays readable.
      const tail = spec.checkpoint_path.split("/").slice(-2).join("/");
      return `alphazero sims=${spec.n_simulations ?? 200} ckpt=${tail}`;
    }
  }
}
