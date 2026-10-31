# Chapter 4: load the pretrained GPT-2 weights

This chapter loads the public weights of **GPT-2 Small** into the `GPTModel`
of [Chapter 3](../chap3/README.md), and uses it to continue a text. With
random weights the model writes nonsense; with the GPT-2 weights, the same
architecture writes fluent English.

The book loads the original OpenAI weights, which are TensorFlow checkpoints,
and reads them with TensorFlow. This repository does not use TensorFlow: it
downloads the same weights from Hugging Face, in the
[safetensors](https://huggingface.co/docs/safetensors) format, and reads them
with the small `safetensors` library. The notebook explains the differences
and the conversion step by step.

## Notebooks

1. **[Load the GPT-2 Small weights and generate text](load_gpt2_weights.ipynb)**:
   download `model.safetensors` from
   [`openai-community/gpt2`](https://huggingface.co/openai-community/gpt2),
   inspect the tensors of the checkpoint, and map them to the parameters of
   `GPTModel`: rename them, split the fused query, key, and value projection,
   transpose the `Conv1D` weights, and copy the token embeddings into the
   output head (GPT-2 uses weight tying). Then load the weights, tokenize a
   prompt, and generate a continuation with greedy decoding.

The reusable code is in the `build_llm` package:

* [`build_llm/gpt2.py`](../build_llm/gpt2.py): `GPTModel` and
  `generate_text_simple`;
* [`build_llm/gpt2_weights.py`](../build_llm/gpt2_weights.py): the GPT-2
  Small configuration, the download with checksum verification, and the
  conversion of the weights.

## A first training run

[`first_training.py`](first_training.py) trains a GPT-2 Small-sized model
from random weights on **31 complete public-domain books, totaling
10,056,985 GPT-2 tokens**. The original twelve books are joined by nineteen
novels, including "War and Peace", "Les Misérables", "The Brothers Karamazov",
and "The Count of Monte Cristo". The texts in `chap4/books/` are converted
from the [Standard Ebooks](https://standardebooks.org/) XHTML sources.
The [book catalogue](books/README.md) lists every book and its token count;
[NOTICE](../NOTICE) and [the manifest](books_manifest.json) record provenance,
licenses, modifications, and pinned source revisions.

At startup, training discovers all `.txt` files directly inside `chap4/books/`
and sorts them alphabetically. Adding or removing a local book automatically
changes the training corpus; no filename list or manifest is read by the
training script. Other files and nested directories are ignored. An empty or
missing `books/` directory produces an error before the model is allocated.

Prepare the nineteen added books before the first training run:

```bash
python -m chap4.prepare_books
```

Preparation downloads pinned official source archives, keeps only bodymatter
in EPUB reading order, preserves chapter headings and paragraphs, and removes
notes, note references, editorial front/back matter, and Gutenberg boilerplate.
The converter normalizes invisible formatting characters and typographic
spaces. The converted files are checked against SHA-256 checksums; existing
modified files are preserved with an error. The original twelve texts remain
unchanged. Added text files are ignored by Git and restored with the command
above. Preparation requires network access only when files are missing (or
when tiktoken's encoding is not cached).

To extend the manifest later with another official Standard Ebooks repository:

```bash
python -m chap4.prepare_books --target-tokens 11000000 --add charles-dickens_a-tale-of-two-cities
```

This records the revision, conversion checksum, and token count for the new
edition. Also update NOTICE and the book catalogue when adding third-party
material. Whole books are retained, so a target may be exceeded slightly.

**The split is made per book**: the first 90% of each book goes to training
and the last 10% to validation, with `split_documents` of
`build_llm/dataloaders.py`. Splitting the joined text once would leave only
the end of the last book in the validation set, so the validation loss
would measure one book only. Inside each set, the books are separated by the
GPT-2 end-of-text token, `<|endoftext|>`.

**Weight tying** is enabled, an addition to the book (`TIE_EMBEDDINGS = True`
in the script, the `tie_embeddings` argument of `GPTModel`). The output head,
which turns each final vector into one score per vocabulary token, and the
token embedding table are both `(vocab_size, emb_dim)` matrices: with weight
tying they are one shared matrix, as in the original GPT-2. Compared with the
book:

* the model has 38,597,376 fewer parameters: 123,822,336 instead of
  162,419,712;
* every update of the output head also improves the token embeddings, which
  helps with little training data;
* the shared matrix is initialized as in GPT-2, with standard deviation 0.02.
  The default initialization of `nn.Embedding` (standard deviation 1) would
  make the first loss about 475 instead of about 11, ln(50,257).

Set `TIE_EMBEDDINGS = False` to train the model of the book.

The script uses a context of 256 tokens (`GPT_CONFIG_SMALL_256`). It prints
the title **A First GPT Training Run**, a compact configuration summary
(device, model with or without weight tying, parameter count, epochs,
batch size, context, learning rate, and split),
an aligned book table with character and token totals, and token and batch
counts for the training and validation sets. It then prints
an aligned table at the first optimizer update, every 10 steps, and at the
end of each epoch. The table shows epoch progress, the one-based global
optimizer step, training and validation losses, their gap (`validation -
training`), and a marker whenever validation reaches a new minimum.
Interactive terminals use colors; set `NO_COLOR` to disable them. Redirected
output stays plain text. Each epoch ends with two mini-charts of the latest
40 evaluations on a shared loss scale (higher characters mean higher loss),
the best validation loss so far, and elapsed time. ASCII trends are used
when the output encoding cannot display Unicode.

At the end, it takes a random passage of the validation set, uses its
first 50 tokens as a prompt, and prints the 50 tokens generated by the model
next to the real continuation from the book. Run it from the repository
root:

```bash
python -m chap4.first_training
```

Training downloads nothing and saves nothing. With the current configuration
(context 256, batch size 4), the per-book 90/10 split produces
**9,058,141 training tokens (8,845 batches per epoch)** and
**998,936 validation tokens (976 batches)**, including document
separators. Split token counts can differ slightly from the sum of complete
books because BPE is applied separately after cutting each text by character.
An epoch has about 9.6 times as many training batches as the twelve-book run;
no wall-clock duration has been measured yet. Expected result: the training
loss falls, while validation loss helps reveal overfitting. Ten million tokens
is still a small corpus for a model of about 124M parameters.

With "The Time Machine" alone (10 epochs, batches of 2 sequences), the run
took about 5 minutes on a MacBook with Apple Silicon (MPS): the training loss
fell from about 9.8 to below 0.2, and the validation loss reached its
minimum, about 5.7, around epoch 4, before rising to about 7.2. The run with
the 31-book corpus has not been recorded yet.

The data loader is in [`build_llm/dataloaders.py`](../build_llm/dataloaders.py).

## Prerequisites

* The `build-llm` Conda environment, with the dependencies and the editable
  installation of the [Quickstart](../QUICKSTART.md). The `safetensors`
  package is in `requirements.txt`.
* Network access for the first run, to download about 548 MB from Hugging
  Face, and to let tiktoken download the GPT-2 encoding if it is not cached.
* About 2 GB of free memory: the checkpoint and the model are both in memory
  while the weights are loaded.

## Run the notebook

From the repository root:

```bash
conda activate build-llm
```

```bash
python -m jupyterlab
```

Open `load_gpt2_weights.ipynb` with the kernel of the `build-llm` environment,
and run it from top to bottom. The model runs on the Apple Silicon GPU (MPS)
when available, otherwise on CUDA or the CPU.

Expected result: the prompt "Every effort moves you" continues with
"forward. The first step is to understand the importance of your work.",
and a second prompt shows the repetitions typical of greedy decoding.

## Downloaded files and cleanup

The weights are saved in `models/gpt2-small/model.safetensors`, in the
repository root. The `models/` folder is ignored by Git. The download uses a
fixed revision of the Hugging Face repository, and the file is accepted only
if its SHA-256 checksum matches the published one. Later runs reuse the file.

To free the disk space, delete the folder:

```bash
rm -r models/gpt2-small
```

## Troubleshooting

* **`ModuleNotFoundError: safetensors`**: install the dependencies again with
  `pip install -r requirements.txt`, then restart the kernel.
* **`Checksum mismatch`**: the download was corrupted, and the partial file
  was removed. Run the cell again.
* **Download errors**: check the network connection, and that
  `huggingface.co` is reachable.

## Scope and next steps

The notebook uses greedy decoding, which always picks the most likely token:
the output is deterministic but tends to repeat itself. The next steps are
temperature and top-k sampling, and then the training loop.

## Development checks

From the repository root, with the `build-llm` environment active:

```bash
black .
```

```bash
pylint build_llm chap* tests
```

```bash
pytest
```

The tests of `build_llm/gpt2_weights.py` run without network access: they use
small random checkpoints and a fake download. A network test checks that
Hugging Face still publishes the expected file, without downloading it:

```bash
pytest -m network
```

The test suite does not execute the notebook. Verify it separately with a
fresh kernel and **Run All Cells**.
