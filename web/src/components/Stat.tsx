import type { ReactNode } from "react";

/**
 * StatTile: one labelled number. A row of these gives a detail page its
 * "at-a-glance" header. Composes via `StatRow` for the standard 4-up layout.
 */
export function StatTile({
  label,
  value,
  hint,
  emphasis = "default",
}: {
  label: string;
  value: ReactNode;
  hint?: ReactNode;
  /** `accent` is used to highlight a single winning number in a group. */
  emphasis?: "default" | "accent";
}) {
  const valueCls =
    emphasis === "accent"
      ? "text-neutral-900 font-semibold"
      : "text-neutral-900";
  return (
    <div className="bg-white px-4 py-3">
      <div className="text-[10px] uppercase tracking-wide text-neutral-500">{label}</div>
      <div className={`font-mono text-2xl tabular-nums mt-1 ${valueCls}`}>{value}</div>
      {hint && <div className="text-xs text-neutral-500 mt-0.5">{hint}</div>}
    </div>
  );
}

/** Four-up tile row sharing one outer border and 1px gaps between cells. */
export function StatRow({ children }: { children: ReactNode }) {
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-px bg-neutral-200 border border-neutral-200">
      {children}
    </div>
  );
}

/**
 * Horizontal bar showing a value in [0, 1] split between two ends. Used in
 * the Specs grid (X-winrate) and anywhere else a 0..1 proportion deserves a
 * visual cue rather than just a decimal.
 */
export function ProportionBar({
  value,
  leftLabel,
  rightLabel,
}: {
  value: number;
  leftLabel?: string;
  rightLabel?: string;
}) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  return (
    <div className="space-y-1">
      <div className="h-1.5 bg-neutral-100 overflow-hidden flex">
        <div
          className="bg-[#1d4ed8]"
          style={{ width: `${pct}%` }}
          aria-hidden
        />
        <div
          className="bg-[#b45309]"
          style={{ width: `${100 - pct}%` }}
          aria-hidden
        />
      </div>
      {(leftLabel || rightLabel) && (
        <div className="flex justify-between text-[10px] text-neutral-500 tabular-nums">
          <span>{leftLabel}</span>
          <span>{rightLabel}</span>
        </div>
      )}
    </div>
  );
}
