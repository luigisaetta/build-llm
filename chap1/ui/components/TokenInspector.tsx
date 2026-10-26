import type { Token } from "@/lib/types";

type TokenInspectorProps = {
  token: Token | null;
  index: number | null;
};

function toHex(value: number): string {
  return value.toString(16).toUpperCase().padStart(2, "0");
}

/** The details of the token under the mouse pointer. */
export function TokenInspector({ token, index }: TokenInspectorProps) {
  if (token === null || index === null) {
    return (
      <div className="flex h-full min-h-28 items-center justify-center rounded-xl border border-dashed border-white/10 p-4 text-center text-sm text-slate-500">
        Move the mouse over a token to inspect it
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3 rounded-xl border border-white/10 bg-white/5 p-4 text-sm">
      <Field label="Position">#{index.toLocaleString()}</Field>
      <Field label="Token ID">
        <span className="text-lg text-violet-300">{token.id}</span>
      </Field>
      <Field label="Text">
        {/* JSON.stringify shows spaces and newlines as escapes, like Python's repr(). */}
        <code className="break-all text-cyan-200">{JSON.stringify(token.text)}</code>
      </Field>
      <Field label={`Bytes (${token.byte_values.length})`}>
        <code className="break-all text-amber-200">
          {token.byte_values.map(toHex).join(" ")}
        </code>
      </Field>
      {token.is_partial_character && (
        <p className="rounded-lg bg-amber-400/10 px-3 py-2 text-amber-200">
          This token holds only part of a character: byte-level BPE can split
          one multi-byte UTF-8 character across tokens.
        </p>
      )}
    </div>
  );
}

function Field({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <div className="text-xs uppercase tracking-wider text-slate-500">
        {label}
      </div>
      <div className="mt-0.5 font-mono text-slate-100">{children}</div>
    </div>
  );
}
