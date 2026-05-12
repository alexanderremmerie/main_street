export type GameSpec = { n: number; schedule: number[] };

export type SpecSamplerConfig = {
  kind: "mixture";
  n_min: number;
  n_max: number;
  turns_min: number;
  turns_max: number;
  fill_min: number;
  fill_max: number;
  max_marks_per_turn: number;
  random_weight?: number;
  arc_weight?: number;
  few_big_weight?: number;
  many_small_weight?: number;
};

export type AgentSpec =
  | { kind: "human" }
  | { kind: "random"; seed?: number | null }
  | { kind: "greedy"; seed?: number | null }
  | { kind: "rightmost" }
  | { kind: "alphabeta"; depth?: number | null }
  | { kind: "extension" }
  | { kind: "blocker" }
  | { kind: "center" }
  | { kind: "forkaware"; seed?: number | null }
  | { kind: "potentialaware"; seed?: number | null }
  | {
      kind: "mcts";
      n_simulations?: number;
      exploration_c?: number;
      seed?: number | null;
      rollout?: "random" | "forkaware";
    }
  | {
      kind: "alphazero";
      checkpoint_path: string;
      n_simulations?: number;
      c_puct?: number;
      temperature?: number;
      seed?: number | null;
    };

export type AgentKind = AgentSpec["kind"];

export type GameRecord = {
  id: string;
  spec: GameSpec;
  x_agent: AgentSpec;
  o_agent: AgentSpec;
  actions: number[];
  outcome: -1 | 0 | 1;
  created_at: string;
  eval_id: string | null;
  seed: number | null;
};

export type EvalSummary = {
  n_games: number;
  a_wins: number;
  b_wins: number;
  ties: number;
  a_winrate: number;
};

export type EvalConfig = {
  agent_a: AgentSpec;
  agent_b: AgentSpec;
  specs: GameSpec[];
  n_games_per_spec: number;
  swap_sides: boolean;
  seed: number;
};

export type SpecSummary = {
  spec: GameSpec;
  n_games: number;
  x_wins: number;
  o_wins: number;
  ties: number;
  last_game_at: string | null;
};

export type PlayerRecord = {
  id: string;
  label: string;
  agent_spec: AgentSpec;
  is_default: boolean;
  created_at: string;
};

export type CheckpointInfo = {
  path: string;
  run_id: string;
  run_name: string;
  label: string;
  iter: number | null;
  is_final: boolean;
};

export type PlayerVerdict = {
  player_id: string;
  label: string;
  cell: number | null;
  agrees_with_oracle: boolean | null;
  error: string | null;
};

export type AnalyzeResponse = {
  verdicts: PlayerVerdict[];
  oracle_value: -1 | 0 | 1 | null;
  oracle_best_cell: number | null;
  oracle_per_cell_values: Record<number, -1 | 0 | 1> | null;
};

export type InspectMove = {
  cell: number;
  raw_policy: number;
  puct_visits: number;
  puct_visit_prob: number;
  puct_value: number | null;
};

export type InspectModelResponse = {
  player_id: string;
  label: string;
  checkpoint_path: string;
  current_player: 1 | 2;
  raw_value: number;
  puct_action: number;
  n_simulations: number;
  moves: InspectMove[];
};

export type OracleResponse = {
  value: -1 | 0 | 1;
  best_cell: number;
  /** Map cell index -> value from X's perspective (+1 X wins, -1 O wins). */
  per_cell_values: Record<number, -1 | 0 | 1>;
  is_terminal: boolean;
};

export type PairResult = {
  a_player_id: string;
  b_player_id: string;
  eval_id: string;
  a_winrate: number;
  n_games: number;
};

export type ComparisonSummary = {
  pairs: PairResult[];
  n_total_games: number;
};

export type ComparisonConfig = {
  player_ids: string[];
  specs: GameSpec[];
  spec_sampler?: SpecSamplerConfig | null;
  n_sampled_specs?: number | null;
  n_games_per_spec: number;
  swap_sides: boolean;
  seed: number;
};

export type ComparisonRecord = {
  id: string;
  config: ComparisonConfig;
  status: "running" | "done" | "failed" | "cancelling" | "cancelled";
  summary: ComparisonSummary | null;
  progress_done: number;
  progress_total: number;
  created_at: string;
};

export type EvalRecord = {
  id: string;
  config: EvalConfig;
  status: "running" | "done" | "failed" | "cancelling" | "cancelled";
  summary: EvalSummary | null;
  created_at: string;
};
