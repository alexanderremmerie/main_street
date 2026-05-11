/**
 * Parse a comma-separated schedule string ("1,2,2,1") into an array of
 * positive integers, returning null on any malformed input. Used by every
 * page that lets a user type a schedule by hand.
 */
export function parseSchedule(s: string): number[] | null {
  const parts = s.split(",").map((p) => p.trim()).filter(Boolean);
  if (parts.length === 0) return null;
  const out: number[] = [];
  for (const p of parts) {
    const v = Number(p);
    if (!Number.isFinite(v) || v <= 0 || !Number.isInteger(v)) return null;
    out.push(v);
  }
  return out;
}
