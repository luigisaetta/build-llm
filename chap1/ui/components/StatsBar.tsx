import type { TokenizeResponse } from "@/lib/types";

/** Cards with the main numbers of a tokenization. */
export function StatsBar({ result }: { result: TokenizeResponse }) {
  const stats = [
    { label: "Tokens", value: result.token_count.toLocaleString(), highlight: true },
    { label: "Characters", value: result.character_count.toLocaleString() },
    { label: "Words", value: result.word_count.toLocaleString() },
    { label: "Characters / token", value: result.characters_per_token.toFixed(2) },
    { label: "Vocabulary size", value: result.vocabulary_size.toLocaleString() },
  ];

  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
      {stats.map((stat) => (
        <div
          key={stat.label}
          className={
            "rounded-2xl border p-4 " +
            (stat.highlight
              ? "border-violet-400/40 bg-gradient-to-br from-violet-500/20 to-cyan-500/10"
              : "border-white/10 bg-white/5")
          }
        >
          <div className="text-xs uppercase tracking-wider text-slate-400">
            {stat.label}
          </div>
          <div
            className={
              "mt-1 font-mono font-semibold tabular-nums " +
              (stat.highlight
                ? "bg-gradient-to-r from-violet-300 to-cyan-300 bg-clip-text text-3xl text-transparent"
                : "text-2xl text-slate-100")
            }
          >
            {stat.value}
          </div>
        </div>
      ))}
    </div>
  );
}
