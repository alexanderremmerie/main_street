import type {
  AgentSpec,
  AnalyzeResponse,
  CheckpointInfo,
  ComparisonConfig,
  ComparisonRecord,
  EvalConfig,
  EvalRecord,
  GameRecord,
  GameSpec,
  InspectModelResponse,
  OracleResponse,
  PlayerRecord,
  SpecSamplerConfig,
  SpecSummary,
} from "./types";

class ApiError extends Error {
  status: number;
  detail: string;
  constructor(status: number, detail: string) {
    super(detail);
    this.status = status;
    this.detail = detail;
  }
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  let r: Response;
  try {
    r = await fetch(path, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    });
  } catch {
    throw new ApiError(0, "Cannot reach the API. Is the backend running on :8000?");
  }
  if (!r.ok) {
    let detail = `${r.status} ${r.statusText}`;
    // FastAPI returns a JSON `detail`; in dev a raw traceback can come back as
    // text. Try both so the user sees the most informative message.
    const raw = await r.text().catch(() => "");
    if (raw) {
      try {
        const body = JSON.parse(raw);
        if (typeof body.detail === "string") {
          detail = body.detail;
        } else if (Array.isArray(body.detail) && body.detail[0]?.msg) {
          detail = body.detail
            .map((d: { loc: string[]; msg: string }) => `${d.loc.slice(1).join(".")}: ${d.msg}`)
            .join("; ");
        } else {
          detail = raw;
        }
      } catch {
        detail = raw;
      }
    }
    throw new ApiError(r.status, detail);
  }
  return r.json() as Promise<T>;
}

export { ApiError };

/** Extract a user-facing message from any thrown value. */
export function errorMessage(e: unknown): string {
  return e instanceof ApiError ? e.detail : String(e);
}

export const api = {
  listGames: (
    params: {
      limit?: number;
      offset?: number;
      eval_id?: string;
      x_kind?: string;
      o_kind?: string;
    } = {},
  ) => {
    const q = new URLSearchParams();
    for (const [k, v] of Object.entries(params)) if (v !== undefined) q.set(k, String(v));
    return req<{ games: GameRecord[]; total: number }>(`/api/games?${q}`);
  },

  /** Ask the exact solver for the value of the current position. May 413 if
   *  the state space is too large to enumerate. */
  oracle: (body: { spec: GameSpec; actions: number[] }) =>
    req<OracleResponse>("/api/oracle", { method: "POST", body: JSON.stringify(body) }),

  getGame: (id: string) => req<GameRecord>(`/api/games/${id}`),

  /** Persist a finished game (any mix of human + bot players). */
  saveGame: (body: {
    spec: GameSpec;
    x_agent: AgentSpec;
    o_agent: AgentSpec;
    actions: number[];
  }) => req<GameRecord>("/api/games", { method: "POST", body: JSON.stringify(body) }),

  /** Ask a bot agent to choose its next cell from the current position. */
  computeMove: (body: { spec: GameSpec; actions: number[]; agent: AgentSpec }) =>
    req<{ cell: number }>("/api/move", { method: "POST", body: JSON.stringify(body) }),

  listSpecs: () => req<SpecSummary[]>("/api/specs"),
  sampleSpecs: (body: { sampler: SpecSamplerConfig; count: number; seed: number }) =>
    req<GameSpec[]>("/api/specs/sample", { method: "POST", body: JSON.stringify(body) }),

  listPlayers: () => req<PlayerRecord[]>("/api/players"),
  listCheckpoints: () => req<CheckpointInfo[]>("/api/checkpoints"),
  createPlayer: (body: { label: string; agent_spec: AgentSpec }) =>
    req<PlayerRecord>("/api/players", { method: "POST", body: JSON.stringify(body) }),
  deletePlayer: (id: string) =>
    req<{ deleted: string }>(`/api/players/${id}`, { method: "DELETE" }),

  /** Ask every player in `player_ids` what it would play at the given position.
   *  The oracle is included automatically when the spec is small enough. */
  analyze: (body: { spec: GameSpec; actions: number[]; player_ids: string[] }) =>
    req<AnalyzeResponse>("/api/analyze", { method: "POST", body: JSON.stringify(body) }),
  inspectModel: (body: {
    spec: GameSpec;
    actions: number[];
    player_id: string;
    n_simulations: number;
  }) =>
    req<InspectModelResponse>("/api/inspect-model", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  listEvals: () => req<EvalRecord[]>("/api/evals"),
  getEval: (id: string) => req<EvalRecord>(`/api/evals/${id}`),
  runEval: (cfg: EvalConfig) =>
    req<EvalRecord>("/api/evals", { method: "POST", body: JSON.stringify(cfg) }),

  listComparisons: () => req<ComparisonRecord[]>("/api/comparisons"),
  getComparison: (id: string) => req<ComparisonRecord>(`/api/comparisons/${id}`),
  runComparison: (cfg: ComparisonConfig) =>
    req<ComparisonRecord>("/api/comparisons", { method: "POST", body: JSON.stringify(cfg) }),
  cancelComparison: (id: string) =>
    req<ComparisonRecord>(`/api/comparisons/${id}/cancel`, { method: "POST" }),
};
