import type React from "react";

export function FormBox({ children }: { children: React.ReactNode }) {
  return (
    <div className="border border-neutral-200 bg-white">{children}</div>
  );
}

export function FormRow({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex items-center px-4 py-2.5 border-b border-neutral-200 last:border-b-0 gap-4">
      <label className="w-28 shrink-0 text-sm text-neutral-500">{label}</label>
      <div className="flex-1 flex items-center flex-wrap gap-2">{children}</div>
      {hint && <span className="text-xs text-neutral-400 font-mono tabular-nums">{hint}</span>}
    </div>
  );
}

export function FormFooter({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between px-4 py-3 border-t border-neutral-200">
      {children}
    </div>
  );
}

const inputCls =
  "border border-neutral-300 bg-white px-2 h-8 focus:outline-none focus:border-neutral-900 transition-colors";

export function NumberInput({
  value,
  onChange,
  placeholder,
  width = "w-20",
}: {
  value: number | null | undefined;
  onChange: (n: number | null) => void;
  placeholder?: string;
  width?: string;
}) {
  return (
    <input
      type="number"
      className={`${inputCls} ${width} font-mono tabular-nums`}
      placeholder={placeholder}
      value={value ?? ""}
      onChange={(e) => onChange(e.target.value === "" ? null : Number(e.target.value))}
    />
  );
}

export function TextInput({
  value,
  onChange,
  width = "w-48",
  mono = false,
  placeholder,
}: {
  value: string;
  onChange: (s: string) => void;
  width?: string;
  mono?: boolean;
  placeholder?: string;
}) {
  return (
    <input
      type="text"
      className={`${inputCls} ${width} ${mono ? "font-mono" : ""}`}
      value={value}
      placeholder={placeholder}
      onChange={(e) => onChange(e.target.value)}
    />
  );
}

export function Select<T extends string>({
  value,
  options,
  onChange,
  width = "w-44",
}: {
  value: T;
  options: { value: T; label: string }[];
  onChange: (v: T) => void;
  width?: string;
}) {
  return (
    <select
      className={`${inputCls} ${width}`}
      value={value}
      onChange={(e) => onChange(e.target.value as T)}
    >
      {options.map((o) => (
        <option key={o.value} value={o.value}>
          {o.label}
        </option>
      ))}
    </select>
  );
}

export function PrimaryButton({
  children,
  onClick,
  disabled,
  type = "button",
}: {
  children: React.ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  type?: "button" | "submit";
}) {
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      // Primary actions in DeepMind blue. Black is reserved for *state*
      // (selected chips, links) so the eye can tell "this is the action"
      // (blue) from "this is currently on" (black).
      className="bg-blue-800 text-white font-medium px-4 h-8 text-sm hover:bg-blue-900 active:bg-blue-950 disabled:opacity-40 disabled:hover:bg-blue-800 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-800 focus-visible:ring-offset-2 transition-colors"
    >
      {children}
    </button>
  );
}

/**
 * Outline button used for everything that isn't the page's primary action:
 * undo, clear, delete, pagination, "all/none" chips, remove-row buttons, etc.
 * One component means consistent height/padding/disabled-state across pages,
 * which is half the reason the UI was reading as "vague" — pages had subtly
 * different versions of the same button.
 */
export function SecondaryButton({
  children,
  onClick,
  disabled,
  title,
  size = "md",
}: {
  children: React.ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  title?: string;
  /** `sm` for inline chips (h-7), `md` for standalone actions (h-8). */
  size?: "sm" | "md";
}) {
  // Three-tier visual hierarchy on the page:
  //   Primary action  →  indigo fill (PrimaryButton)
  //   Secondary       →  gray fill   (this)
  //   Input field     →  white       (inputCls)
  // Each tier looks distinct from the others at a glance. The previous
  // "outlined on white" style for secondaries collapsed two of those tiers.
  const sizing = size === "sm" ? "h-7 px-2.5 text-xs" : "h-8 px-3 text-xs";
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      title={title}
      className={`${sizing} font-medium text-neutral-900 bg-neutral-100 border border-neutral-400 hover:bg-neutral-200 hover:border-neutral-700 active:bg-neutral-300 disabled:opacity-40 disabled:hover:bg-neutral-100 disabled:hover:border-neutral-400 focus:outline-none focus-visible:ring-2 focus-visible:ring-neutral-900 focus-visible:ring-offset-2 whitespace-nowrap transition-colors`}
    >
      {children}
    </button>
  );
}

export function Checkbox({
  checked,
  onChange,
  label,
}: {
  checked: boolean;
  onChange: (b: boolean) => void;
  label: string;
}) {
  return (
    <label className="inline-flex items-center gap-2 text-sm text-neutral-700 cursor-pointer select-none">
      <input
        type="checkbox"
        className="accent-neutral-900"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
      />
      {label}
    </label>
  );
}
