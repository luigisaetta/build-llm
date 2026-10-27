# Chapter 1: text data and tokenization

A language model never reads letters or words. Before anything else, the text
is split into **tokens**, and each token is mapped to an integer **token ID**.
The model only ever sees these IDs.

This chapter has two parts:

1. The **Tokenizer explorer**, an interactive demo (web UI and command line)
   that shows how GPT tokenizers split a text.
2. The **data loader notebook** ([`dataloader.ipynb`](dataloader.ipynb)),
   which turns a tokenized text into the input and target batches used to
   train a model to predict the next token.

## Part 1: the Tokenizer explorer

The **Tokenizer explorer** shows how the Byte Pair Encoding (BPE) tokenizers used by GPT models split a
text, using the [tiktoken](https://github.com/openai/tiktoken) library. With
shows how the Byte Pair Encoding (BPE) tokenizers used by GPT models split a
text, using the [tiktoken](https://github.com/openai/tiktoken) library. With
it you can:

* choose the tokenizer: `gpt2` (the one used in the book), `cl100k_base`
  (GPT-4), `o200k_base` (GPT-4o), and others;
* paste a text, pick a sample, or read the text of a web page from its URL;
* see the text as a sequence of colored tokens, with their IDs and raw bytes;
* read the counts: tokens, characters, words, and characters per token.

Things worth trying:

* Compare the number of tokens of the English and Italian samples: languages
  other than English usually need more tokens for the same content.
* Switch from `gpt2` to `o200k_base` on the same text: larger vocabularies
  need fewer tokens.
* Load the Emoji sample: one emoji is split into several tokens, each holding
  a few bytes of its UTF-8 encoding. These tokens are shown in italics.
* Turn on "Show whitespace": most tokens start with a space, which belongs
  to the token itself.

### Architecture

```text
Browser (React)        Next.js server (port 3000)          FastAPI (port 8000)
     |                          |                                   |
     | POST /api/tokenize       |                                   |
     |------------------------->| POST http://127.0.0.1:8000/tokenize
     |                          |---------------------------------->| tiktoken
     |                          |<------------- JSON ---------------|
     |<--------- JSON ----------|                                   |
```

* **`build_llm/tokenization.py`**, **`build_llm/text_sources.py`**: the
  reusable Python logic, shared by the command line and the API.
* **`chap1/tokenize_text.py`**: the command-line tool.
* **`chap1/api.py`**: the HTTP API (FastAPI), used by the web UI.
* **`chap1/ui/`**: the web UI (Next.js). The browser calls only the routes of
  the Next.js app (`chap1/ui/app/api/`), which forward the requests to the
  Python backend. So no CORS configuration is needed, and the backend address
  is configured in one place.

### Prerequisites

* The `build-llm` Conda environment, with the packages of
  `requirements.txt` and `requirements-dev.txt` installed. Check it with
  `python -m chap0.check_environment`.
* [Node.js](https://nodejs.org/) 20.9 or later, with npm (for the web UI).
* Network access the first time you use each tokenizer: tiktoken downloads
  its vocabulary file once and keeps it in a local cache.

### Run the demo

The demo needs two processes, each in its own terminal.

**Terminal 1: the Python backend.** From the repository root:

```bash
conda activate build-llm
```

```bash
python -m chap1.api
```

The API listens on http://127.0.0.1:8000. Its interactive documentation is at
http://127.0.0.1:8000/docs. Use `--port` to change the port, and `--reload`
to restart the server automatically when the code changes.

**Terminal 2: the web UI.** The first time, install the JavaScript
dependencies:

```bash
cd chap1/ui
```

```bash
npm install
```

Then start the development server:

```bash
npm run dev
```

Open http://localhost:3000 in your browser.

If the backend runs on a different address, set `BACKEND_URL` before starting
the UI, for example:

```bash
BACKEND_URL=http://127.0.0.1:9000 npm run dev
```

To stop the demo, press `Ctrl+C` in both terminals.

### Use the command line

The same tokenization is available from the command line, without the UI.
Run these commands from the repository root:

```bash
python -m chap1.tokenize_text --text "Hello, world!"
```

```bash
python -m chap1.tokenize_text --encoding o200k_base --url https://en.wikipedia.org/wiki/Large_language_model
```

```bash
python -m chap1.tokenize_text --file notes.txt --format json
```

```bash
echo "Text from standard input" | python -m chap1.tokenize_text
```

```bash
python -m chap1.tokenize_text --list-encodings
```

Example output:

```text
Encoding:             gpt2
Vocabulary size:      50,257
Characters:           13
Words:                2
Tokens:               4
Characters per token: 3.25

     #       ID  Text
     0    15496  'Hello'
     1       11  ','
     2      995  ' world'
     3        0  '!'
```

Use `--show N` to list more or fewer tokens (`--show 0` lists all of them).
The exit code is 0 on success, 1 if the text cannot be read, and 2 for
invalid arguments.

## Part 2: the data loader notebook

To train a GPT model, the tokenized text must become pairs of sequences: an
**input** and a **target**, which is the input shifted forward by one token.
For each position, the model learns to predict the target token from the
input tokens up to that position.

The notebook [`dataloader.ipynb`](dataloader.ipynb), largely inspired by the
[notebook of the book](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch02/01_main-chapter-code/dataloader.ipynb),
shows how:

* `GPTDatasetV1` tokenizes a text with tiktoken (`gpt2`) and cuts it into
  input and target sequences with a **sliding window**;
* `create_dataloader_v1` wraps the dataset in a PyTorch `DataLoader`, which
  groups the sequences into shuffled batches;
* a batch of token IDs becomes the model input: **token embeddings** plus
  **positional embeddings**, with shape `(batch_size, max_length, emb_dim)`,
  for example `(8, 4, 256)`.

It also explains the difference between three parameters:

* `max_length`: the number of tokens of each input sequence;
* `stride`: how many tokens the window moves between one sequence and the
  next (a stride smaller than `max_length` makes the sequences overlap);
* `context_length`: the number of positions of the positional embedding
  table, which must be at least `max_length`.

The text used is [`the-verdict.txt`](the-verdict.txt): the short story
"The Verdict" by Edith Wharton (1908), in the public domain, the same text
used in the book.

### Run the notebook

From the repository root, with the `build-llm` environment active:

```bash
python -m jupyterlab
```

Open `chap1/dataloader.ipynb` and run the cells in order. The notebook reads
`the-verdict.txt` from its own folder, and needs network access only the first
time, to download the `gpt2` vocabulary.

## Tests and checks

From the repository root:

```bash
pytest
```

The tests use a tiny hand-built BPE encoding and fake HTTP responses, so they
need no network. One test checks the real `gpt2` encoding; it may download
the vocabulary, so it runs only on request:

```bash
pytest -m network
```

For the web UI, from `chap1/ui`:

```bash
npm run lint
```

```bash
npm run build
```

## Limitations of the Tokenizer explorer

* Reading a URL works for HTML pages and plain-text files, not for PDF or
  other formats. When a page has a `<main>` or `<article>` element, only its
  content is kept; some sites still put menus or lists of links inside it
  (on Wikipedia, for example, the list of languages).
* Pages that build their content with JavaScript may return little text.
* Texts longer than 200,000 characters are rejected by the tokenize endpoint,
  and text read from a URL is truncated to that size. The UI draws at most
  the first 5,000 tokens; the counts always refer to the whole text.
* Special tokens such as `<|endoftext|>` are tokenized as plain text.
* The backend reads any URL it is asked for. It is meant to run locally, on
  `127.0.0.1`: do not expose it on a public network.
