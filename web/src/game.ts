import type { GameSpec } from "./types";

export type Cell = 0 | 1 | 2;

export type State = {
  board: Cell[];
  turnIdx: number;
  placementsLeft: number;
  isTerminal: boolean;
  currentPlayer: 1 | 2 | null;
};

export function initialState(spec: GameSpec): State {
  if (spec.schedule.length === 0) {
    throw new Error("initialState: spec.schedule must be non-empty");
  }
  if (spec.n <= 0) {
    throw new Error(`initialState: spec.n must be positive (got ${spec.n})`);
  }
  return {
    board: Array(spec.n).fill(0) as Cell[],
    turnIdx: 0,
    placementsLeft: spec.schedule[0],
    isTerminal: false,
    currentPlayer: 1,
  };
}

export function step(spec: GameSpec, state: State, cell: number): State {
  if (state.isTerminal || state.currentPlayer === null) return state;
  // JS array writes silently extend out-of-bounds; guard explicitly so a stale
  // index can't grow the board past spec.n.
  if (!Number.isInteger(cell) || cell < 0 || cell >= spec.n) {
    throw new RangeError(`cell ${cell} out of range for n=${spec.n}`);
  }
  if (state.board[cell] !== 0) {
    throw new Error(`cell ${cell} is not empty`);
  }
  const board = state.board.slice() as Cell[];
  board[cell] = state.currentPlayer;
  let placementsLeft = state.placementsLeft - 1;
  let turnIdx = state.turnIdx;
  if (placementsLeft === 0) {
    turnIdx += 1;
    placementsLeft = turnIdx < spec.schedule.length ? spec.schedule[turnIdx] : 0;
  }
  const isTerminal = turnIdx >= spec.schedule.length;
  const currentPlayer = isTerminal ? null : turnIdx % 2 === 0 ? 1 : 2;
  return { board, turnIdx, placementsLeft, isTerminal, currentPlayer };
}

export function statesAlong(spec: GameSpec, actions: number[]): State[] {
  const out: State[] = [initialState(spec)];
  for (const a of actions) out.push(step(spec, out[out.length - 1], a));
  return out;
}

export function longestRun(board: Cell[], mark: 1 | 2): { len: number; end: number } {
  let bestLen = 0;
  let bestEnd = -1;
  let cur = 0;
  for (let i = 0; i < board.length; i++) {
    if (board[i] === mark) {
      cur += 1;
      if (cur >= bestLen) { bestLen = cur; bestEnd = i; }
    } else cur = 0;
  }
  return { len: bestLen, end: bestEnd };
}

/** Winner from the final board: +1 if X wins, -1 if O, 0 only on an empty
 *  board (impossible at terminal in a valid game). Mirrors core.outcome. */
export function winner(board: Cell[]): -1 | 0 | 1 {
  const x = longestRun(board, 1);
  const o = longestRun(board, 2);
  if (x.len === 0 && o.len === 0) return 0;
  if (x.len !== o.len) return x.len > o.len ? 1 : -1;
  return x.end > o.end ? 1 : -1;
}

export function winningCells(state: State): Set<number> {
  const out = new Set<number>();
  if (!state.isTerminal) return out;
  const x = longestRun(state.board, 1);
  const o = longestRun(state.board, 2);
  if (x.len === 0 && o.len === 0) return out;
  const w = winner(state.board) === 1 ? x : o;
  for (let i = w.end - w.len + 1; i <= w.end; i++) out.add(i);
  return out;
}
