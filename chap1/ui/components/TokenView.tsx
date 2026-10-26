import type { Token } from "@/lib/types";

// Tokens beyond this limit are counted but not drawn, to keep the page fast.
export const MAX_TOKENS_SHOWN = 5000;

// Consecutive tokens get different colors, so their boundaries are visible.
// The class names are written in full so that Tailwind includes them.
const TOKEN_COLORS = [
  "bg-violet-500/25 text-violet-100 ring-violet-400/40",
  "bg-cyan-500/25 text-cyan-100 ring-cyan-400/40",
  "bg-amber-500/25 text-amber-100 ring-amber-400/40",
  "bg-emerald-500/25 text-emerald-100 ring-emerald-400/40",
  "bg-rose-500/25 text-rose-100 ring-rose-400/40",
  "bg-sky-500/25 text-sky-100 ring-sky-400/40",
];

type TokenViewProps = {
  tokens: Token[];
  showIds: boolean;
  showWhitespace: boolean;
  activeIndex: number | null;
  onActivate: (index: number | null) => void;
};

/**
 * Make spaces, tabs, and newlines visible. Newlines are kept after the
 * arrow, so that the text still wraps where the original does.
 */
function displayText(text: string, showWhitespace: boolean): string {
  if (!showWhitespace) {
    return text;
  }
  return text.replace(/ /g, "·").replace(/\t/g, "→").replace(/\n/g, "↵\n");
}

/** The text drawn as a sequence of colored tokens. */
export function TokenView({
  tokens,
  showIds,
  showWhitespace,
  activeIndex,
  onActivate,
}: TokenViewProps) {
  const shown = tokens.slice(0, MAX_TOKENS_SHOWN);

  return (
    <div>
      <div
        className="max-h-[32rem] overflow-y-auto whitespace-pre-wrap break-words rounded-xl border border-white/10 bg-black/30 p-4 font-mono text-[15px] leading-8"
        onMouseLeave={() => onActivate(null)}
      >
        {shown.map((token, index) => {
          const color = TOKEN_COLORS[index % TOKEN_COLORS.length];
          const isActive = index === activeIndex;
          return (
            <span
              key={index}
              onMouseEnter={() => onActivate(index)}
              className={
                "cursor-default rounded-[4px] px-[1px] py-0.5 transition " +
                color +
                (isActive ? " ring-2 brightness-150" : "") +
                (token.is_partial_character ? " italic opacity-80" : "")
              }
            >
              {showIds ? (
                <span className="text-xs tabular-nums opacity-90">
                  {token.id}
                </span>
              ) : (
                displayText(token.text, showWhitespace)
              )}
            </span>
          );
        })}
      </div>
      {tokens.length > MAX_TOKENS_SHOWN && (
        <p className="mt-2 text-sm text-amber-300">
          Showing the first {MAX_TOKENS_SHOWN.toLocaleString()} of{" "}
          {tokens.length.toLocaleString()} tokens.
        </p>
      )}
    </div>
  );
}
