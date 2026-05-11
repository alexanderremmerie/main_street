export function TimeAgo({ iso }: { iso: string }) {
  const t = new Date(iso).getTime();
  const s = Math.floor((Date.now() - t) / 1000);
  let label: string;
  if (s < 60) label = `${s}s ago`;
  else if (s < 3600) label = `${Math.floor(s / 60)}m ago`;
  else if (s < 86400) label = `${Math.floor(s / 3600)}h ago`;
  else label = `${Math.floor(s / 86400)}d ago`;
  return (
    <time dateTime={iso} title={new Date(iso).toLocaleString()} className="tabular-nums">
      {label}
    </time>
  );
}
