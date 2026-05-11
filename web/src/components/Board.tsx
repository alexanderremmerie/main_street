import { winningCells, type Cell, type State } from "../game";

/** Per-cell oracle annotation drawn underneath the cell label. `value` is from
 *  X's perspective (+1 X wins, -1 O wins) and `isBest` marks the optimal move
 *  for the side to move. */
export type CellAnnotation = { value: -1 | 0 | 1; isBest: boolean };

type Props = {
  state: State;
  highlightWinner?: boolean;
  lastAction?: number | null;
  /** When set, empty cells become clickable; the callback receives the cell index. */
  onCellClick?: (cell: number) => void;
  /** Optional accent color class (e.g. "X" hover blue or "O" hover amber). */
  hoverPlayer?: "X" | "O" | null;
  /** Optional oracle annotations keyed by cell index. */
  annotations?: Record<number, CellAnnotation>;
};

const cellGrid = (n: number): React.CSSProperties => ({
  display: "grid",
  gap: "1px",
  gridTemplateColumns: `repeat(${n}, minmax(20px, 36px))`,
});

const HOVER_X = "hover:bg-blue-100";
const HOVER_O = "hover:bg-amber-100";

export function Board({
  state,
  highlightWinner = false,
  lastAction = null,
  onCellClick,
  hoverPlayer = null,
  annotations,
}: Props) {
  const winners = highlightWinner ? winningCells(state) : new Set<number>();
  const n = state.board.length;
  const grid = cellGrid(n);
  const hoverCls = hoverPlayer === "X" ? HOVER_X : hoverPlayer === "O" ? HOVER_O : "";

  return (
    <div className="overflow-x-auto select-none">
      <div className="w-fit">
        <div style={grid} className="bg-neutral-200 border border-neutral-200">
          {state.board.map((c, i) => (
            <BoardCell
              key={i}
              value={c}
              isWinner={winners.has(i)}
              isLast={lastAction === i}
              clickable={Boolean(onCellClick) && c === 0}
              hoverCls={hoverCls}
              annotation={annotations?.[i]}
              onClick={onCellClick && c === 0 ? () => onCellClick(i) : undefined}
            />
          ))}
        </div>
        <div style={grid} className="mt-1">
          {state.board.map((_, i) => (
            <div
              key={i}
              className="text-[10px] text-neutral-400 font-mono tabular-nums text-center"
            >
              {i}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function BoardCell({
  value,
  isWinner,
  isLast,
  clickable,
  hoverCls,
  annotation,
  onClick,
}: {
  value: Cell;
  isWinner: boolean;
  isLast: boolean;
  clickable: boolean;
  hoverCls: string;
  annotation?: CellAnnotation;
  onClick?: () => void;
}) {
  let bg = "bg-white";
  let fg = "";
  let label = "";
  if (value === 1) {
    bg = "bg-[#1d4ed8]";
    fg = "text-white";
    label = "X";
  } else if (value === 2) {
    bg = "bg-[#b45309]";
    fg = "text-white";
    label = "O";
  } else if (annotation) {
    // Tint empty cells by the value the side-to-move achieves by playing here.
    // We don't color occupied cells because the result is fixed for them.
    if (annotation.isBest) {
      bg = "bg-emerald-100";
    } else if (annotation.value === 1) {
      bg = "bg-blue-50";
    } else if (annotation.value === -1) {
      bg = "bg-amber-50";
    }
  }

  const inner = isWinner
    ? "shadow-[inset_0_0_0_2px_#0a0a0a]"
    : isLast
      ? "shadow-[inset_0_0_0_1px_#525252]"
      : annotation?.isBest && value === 0
        ? "shadow-[inset_0_0_0_2px_#059669]"
        : "";

  const interactive = clickable ? `cursor-pointer ${hoverCls}` : "";

  return (
    <div
      onClick={onClick}
      className={`aspect-square flex items-center justify-center font-mono font-semibold text-xs ${bg} ${fg} ${inner} ${interactive}`}
    >
      {label}
    </div>
  );
}
