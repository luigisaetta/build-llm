import { TokenizerDemo } from "@/components/TokenizerDemo";

export default function Home() {
  return (
    <main className="mx-auto flex w-full max-w-6xl flex-col gap-10 px-6 py-14">
      <header className="flex flex-col gap-4">
        <span className="w-fit rounded-full border border-violet-400/30 bg-violet-500/10 px-3 py-1 text-xs font-medium uppercase tracking-widest text-violet-200">
          build-llm · Chapter 1 · Text data and tokenization
        </span>
        <h1 className="text-5xl font-bold tracking-tight text-white sm:text-6xl">
          How does an LLM{" "}
          <span className="bg-gradient-to-r from-violet-400 via-fuchsia-300 to-cyan-300 bg-clip-text text-transparent">
            see your text?
          </span>
        </h1>
        <p className="max-w-3xl text-lg text-slate-300">
          A language model never reads letters or words. Before anything else,
          the text is split into <strong className="text-white">tokens</strong>{" "}
          and each token becomes an integer ID. Pick a tokenizer, paste some
          text or read a web page, and watch it happen.
        </p>
      </header>

      <TokenizerDemo />

      <footer className="pb-4 text-center text-sm text-slate-500">
        Tokenization by{" "}
        <a
          className="text-slate-300 underline-offset-4 hover:underline"
          href="https://github.com/openai/tiktoken"
        >
          tiktoken
        </a>
        , served by a Python FastAPI backend. Inspired by &ldquo;Build a Large
        Language Model (From Scratch)&rdquo; by Sebastian Raschka.
      </footer>
    </main>
  );
}
