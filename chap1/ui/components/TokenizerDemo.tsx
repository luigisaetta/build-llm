"use client";

import { useEffect, useRef, useState } from "react";

import { EncodingPicker } from "@/components/EncodingPicker";
import { SourcePanel } from "@/components/SourcePanel";
import { StatsBar } from "@/components/StatsBar";
import { TokenInspector } from "@/components/TokenInspector";
import { TokenView } from "@/components/TokenView";
import * as api from "@/lib/api-client";
import { SAMPLES } from "@/lib/samples";
import type { EncodingInfo, TokenizeResponse } from "@/lib/types";

// Wait for a short pause in typing before tokenizing again.
const TOKENIZE_DELAY_MS = 250;

/** The interactive tokenizer: input, encoding choice, and results. */
export function TokenizerDemo() {
  const [encodings, setEncodings] = useState<EncodingInfo[]>([]);
  const [encoding, setEncoding] = useState("gpt2");
  const [text, setText] = useState(SAMPLES[0].text);
  const [result, setResult] = useState<TokenizeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isTokenizing, setIsTokenizing] = useState(false);
  const [isFetching, setIsFetching] = useState(false);
  const [sourceNote, setSourceNote] = useState<string | null>(null);
  const [showIds, setShowIds] = useState(false);
  const [showWhitespace, setShowWhitespace] = useState(false);
  const [activeIndex, setActiveIndex] = useState<number | null>(null);

  // Each request gets a number: answers to older requests are ignored, so a
  // slow answer cannot overwrite the result of a newer text.
  const latestRequest = useRef(0);

  useEffect(() => {
    api
      .getEncodings()
      .then((response) => {
        setEncodings(response.encodings);
        setEncoding(response.default);
      })
      .catch((reason: Error) => setError(reason.message));
  }, []);

  useEffect(() => {
    if (!text) {
      return;
    }
    const requestNumber = ++latestRequest.current;
    const timer = setTimeout(async () => {
      setIsTokenizing(true);
      try {
        const response = await api.tokenize(text, encoding);
        if (requestNumber === latestRequest.current) {
          setResult(response);
          setError(null);
          setActiveIndex(null);
        }
      } catch (reason) {
        if (requestNumber === latestRequest.current) {
          setError((reason as Error).message);
        }
      } finally {
        if (requestNumber === latestRequest.current) {
          setIsTokenizing(false);
        }
      }
    }, TOKENIZE_DELAY_MS);
    return () => clearTimeout(timer);
  }, [text, encoding]);

  function handleTextChange(newText: string) {
    setText(newText);
    setSourceNote(null);
  }

  async function handleFetchUrl(url: string) {
    setIsFetching(true);
    try {
      const fetched = await api.fetchUrl(url);
      setText(fetched.text);
      const title = fetched.title ? `“${fetched.title}”` : fetched.url;
      const note = fetched.truncated ? " (truncated to the size limit)" : "";
      setSourceNote(`Text read from ${title}${note}.`);
      setError(null);
    } catch (reason) {
      setError((reason as Error).message);
    } finally {
      setIsFetching(false);
    }
  }

  // An empty text has nothing to show, whatever the last result was.
  const visibleResult = text ? result : null;
  const activeToken =
    visibleResult && activeIndex !== null
      ? visibleResult.tokens[activeIndex]
      : null;

  return (
    <div className="flex flex-col gap-6">
      <Card step="1" title="Choose a tokenizer">
        <EncodingPicker
          encodings={encodings}
          selected={encoding}
          onSelect={setEncoding}
        />
      </Card>

      <Card step="2" title="Provide some text">
        <SourcePanel
          text={text}
          onTextChange={handleTextChange}
          onFetchUrl={handleFetchUrl}
          isFetching={isFetching}
          sourceNote={sourceNote}
        />
      </Card>

      {error && (
        <div className="rounded-xl border border-rose-400/30 bg-rose-500/10 px-4 py-3 text-rose-200">
          {error}
        </div>
      )}

      {visibleResult && (
        <Card
          step="3"
          title="See the tokens"
          aside={
            <div className="flex items-center gap-4 text-sm text-slate-300">
              {isTokenizing && (
                <span className="animate-pulse text-cyan-300">Tokenizing…</span>
              )}
              <Toggle label="Show IDs" checked={showIds} onChange={setShowIds} />
              <Toggle
                label="Show whitespace"
                checked={showWhitespace}
                onChange={setShowWhitespace}
              />
            </div>
          }
        >
          <div className="flex flex-col gap-5">
            <StatsBar result={visibleResult} />
            <div className="grid gap-4 lg:grid-cols-[1fr_18rem]">
              <TokenView
                tokens={visibleResult.tokens}
                showIds={showIds}
                showWhitespace={showWhitespace}
                activeIndex={activeIndex}
                onActivate={setActiveIndex}
              />
              <TokenInspector token={activeToken} index={activeIndex} />
            </div>
          </div>
        </Card>
      )}
    </div>
  );
}

function Card({
  step,
  title,
  aside,
  children,
}: {
  step: string;
  title: string;
  aside?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <section className="rounded-3xl border border-white/10 bg-white/[0.04] p-6 shadow-2xl shadow-black/30 backdrop-blur-xl">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <h2 className="flex items-center gap-3 text-lg font-semibold text-white">
          <span className="flex h-7 w-7 items-center justify-center rounded-full bg-gradient-to-br from-violet-500 to-cyan-500 text-sm">
            {step}
          </span>
          {title}
        </h2>
        {aside}
      </div>
      {children}
    </section>
  );
}

function Toggle({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
}) {
  return (
    <label className="flex cursor-pointer items-center gap-2 select-none">
      <input
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(event.target.checked)}
        className="h-4 w-4 accent-violet-500"
      />
      {label}
    </label>
  );
}
