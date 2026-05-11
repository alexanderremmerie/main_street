/**
 * Page-level layout primitives. Every research page renders the same shape
 * (title, optional description, optional right-aligned actions), and every
 * section inside a page renders the same h2 + optional action. Keeping these
 * in one place is the difference between "polished" and "I styled each page
 * slightly differently because I forgot what I did last time."
 */

import type { ReactNode } from "react";

export function PageHeader({
  title,
  description,
  actions,
}: {
  title: string;
  description?: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <div className="flex items-start justify-between gap-6 mb-8">
      <div className="space-y-1 min-w-0">
        <h1 className="text-lg font-semibold text-neutral-900 tracking-tight">{title}</h1>
        {description && (
          <p className="text-sm text-neutral-500 max-w-prose">{description}</p>
        )}
      </div>
      {actions && <div className="flex items-center gap-2 shrink-0">{actions}</div>}
    </div>
  );
}

export function SectionHeader({
  title,
  description,
  actions,
}: {
  title: string;
  description?: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <div className="flex items-baseline justify-between gap-4 mb-3">
      <div>
        <h2 className="text-sm font-semibold text-neutral-900 tracking-tight">{title}</h2>
        {description && (
          <p className="text-xs text-neutral-500 mt-0.5 max-w-prose">{description}</p>
        )}
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </div>
  );
}

/** A muted block used in place of a table when there's nothing to show. */
export function EmptyState({ children }: { children: ReactNode }) {
  return (
    <div className="border border-dashed border-neutral-200 px-4 py-8 text-sm text-neutral-500 text-center">
      {children}
    </div>
  );
}

export function LoadingState({ children = "Loading…" }: { children?: ReactNode }) {
  return (
    <div className="text-sm text-neutral-500 px-1 py-2" role="status" aria-live="polite">
      {children}
    </div>
  );
}

/**
 * One row in a label-aligned description list — used on detail pages to lay
 * out a record's metadata above the rich content. Wrap several in a
 * `<div className="border border-neutral-200">…</div>` for the framed look.
 */
export function DescriptionRow({
  label,
  children,
}: {
  label: string;
  children: ReactNode;
}) {
  return (
    <div className="flex items-start px-4 py-2.5 border-b border-neutral-200 last:border-b-0 gap-3">
      <span className="w-32 text-sm text-neutral-500 shrink-0 mt-0.5">{label}</span>
      <div className="flex-1">{children}</div>
    </div>
  );
}
