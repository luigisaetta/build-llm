"use client";

import { useState } from "react";

import { SAMPLES } from "@/lib/samples";

type SourceMode = "paste" | "url";

type SourcePanelProps = {
  text: string;
  onTextChange: (text: string) => void;
  onFetchUrl: (url: string) => Promise<void>;
  isFetching: boolean;
  sourceNote: string | null;
};

/** The input of the demo: pasted text, a sample, or the text of a URL. */
export function SourcePanel({
  text,
  onTextChange,
  onFetchUrl,
  isFetching,
  sourceNote,
}: SourcePanelProps) {
  const [mode, setMode] = useState<SourceMode>("paste");
  const [url, setUrl] = useState("");

  async function handleFetch(event: React.FormEvent) {
    event.preventDefault();
    if (url.trim()) {
      await onFetchUrl(url.trim());
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="flex w-fit gap-1 rounded-xl border border-white/10 bg-black/20 p-1">
        <TabButton active={mode === "paste"} onClick={() => setMode("paste")}>
          ✍️ Paste text
        </TabButton>
        <TabButton active={mode === "url"} onClick={() => setMode("url")}>
          🌐 From URL
        </TabButton>
      </div>

      {mode === "url" && (
        <form onSubmit={handleFetch} className="flex gap-2">
          <input
            type="url"
            value={url}
            onChange={(event) => setUrl(event.target.value)}
            placeholder="https://en.wikipedia.org/wiki/Large_language_model"
            className="flex-1 rounded-xl border border-white/10 bg-black/30 px-4 py-2.5 text-slate-100 placeholder:text-slate-500 focus:border-violet-400 focus:outline-none"
          />
          <button
            type="submit"
            disabled={isFetching || !url.trim()}
            className="rounded-xl bg-gradient-to-r from-violet-500 to-cyan-500 px-5 py-2.5 font-medium text-white shadow-lg shadow-violet-500/20 transition hover:brightness-110 disabled:opacity-40"
          >
            {isFetching ? "Reading…" : "Fetch text"}
          </button>
        </form>
      )}

      {sourceNote && (
        <p className="rounded-lg border border-cyan-400/20 bg-cyan-400/5 px-3 py-2 text-sm text-cyan-200">
          {sourceNote}
        </p>
      )}

      <textarea
        value={text}
        onChange={(event) => onTextChange(event.target.value)}
        placeholder="Paste or type some text here…"
        rows={8}
        className="w-full resize-y rounded-xl border border-white/10 bg-black/30 p-4 font-mono text-sm leading-relaxed text-slate-100 placeholder:text-slate-500 focus:border-violet-400 focus:outline-none"
      />

      <div className="flex flex-wrap items-center gap-2 text-sm">
        <span className="text-slate-400">Try a sample:</span>
        {SAMPLES.map((sample) => (
          <button
            key={sample.label}
            type="button"
            onClick={() => onTextChange(sample.text)}
            className="rounded-lg border border-white/10 bg-white/5 px-2.5 py-1 text-slate-300 transition hover:border-white/30 hover:text-white"
          >
            {sample.label}
          </button>
        ))}
        {text && (
          <button
            type="button"
            onClick={() => onTextChange("")}
            className="ml-auto text-slate-400 transition hover:text-rose-300"
          >
            Clear
          </button>
        )}
      </div>
    </div>
  );
}

function TabButton({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={
        "rounded-lg px-4 py-1.5 text-sm transition " +
        (active
          ? "bg-white/10 text-white shadow"
          : "text-slate-400 hover:text-white")
      }
    >
      {children}
    </button>
  );
}
