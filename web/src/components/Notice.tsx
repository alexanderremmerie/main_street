type Props = { kind?: "error" | "info"; children: React.ReactNode };

export function Notice({ kind = "info", children }: Props) {
  const cls = kind === "error" ? "border-red-300 text-red-700 bg-red-50" : "border-neutral-200 text-neutral-700 bg-neutral-50";
  return <div className={`border px-3 py-2 text-sm ${cls}`}>{children}</div>;
}
