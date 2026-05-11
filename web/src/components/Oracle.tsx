import type { CellAnnotation } from "./Board";
import { Mark } from "./Mark";
import { Notice } from "./Notice";
import type { OracleResponse } from "../types";
import type { OracleResult } from "../oracle";

/** Convert oracle output into the per-cell annotations the Board renders. */
export function oracleAnnotations(o: OracleResponse): Record<number, CellAnnotation> {
  const out: Record<number, CellAnnotation> = {};
  for (const [k, v] of Object.entries(o.per_cell_values)) {
    out[Number(k)] = { value: v, isBest: Number(k) === o.best_cell };
  }
  return out;
}

/** Compact strip that shows the solver's verdict for the current position. */
export function OraclePanel({ result }: { result: OracleResult }) {
  if (result.kind === "idle") return null;
  if (result.kind === "loading") {
    return <PanelShell>solving…</PanelShell>;
  }
  if (result.kind === "too-large") {
    return (
      <PanelShell muted>
        Position too large for the exact solver.
      </PanelShell>
    );
  }
  if (result.kind === "error") {
    return <Notice kind="error">{result.message}</Notice>;
  }

  const { value, best_cell, is_terminal } = result.data;
  const winner: "X" | "O" = value === 1 ? "X" : "O";
  return (
    <PanelShell>
      <span className="flex items-center gap-2">
        <span className="text-neutral-500">Oracle:</span>
        <Mark player={winner} />
        <span className="text-neutral-900">wins with best play</span>
        {!is_terminal && (
          <>
            <span className="text-neutral-300">·</span>
            <span className="text-neutral-500">best cell</span>
            <span className="font-mono tabular-nums text-neutral-900">{best_cell}</span>
          </>
        )}
      </span>
    </PanelShell>
  );
}

function PanelShell({
  children,
  muted = false,
}: {
  children: React.ReactNode;
  muted?: boolean;
}) {
  return (
    <div
      className={`border border-neutral-200 bg-white px-3 py-2 text-sm ${
        muted ? "text-neutral-500" : "text-neutral-700"
      }`}
    >
      {children}
    </div>
  );
}
