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
counts for the training set, the validation set, and the validation sample
(see below). It then prints
an aligned table at the first optimizer update, every 10 steps, and at the
end of each epoch. The table shows epoch progress, the one-based global
optimizer step, training and validation losses, their gap (`validation -
training`), and a marker whenever validation reaches a new minimum.
Interactive terminals use colors; set `NO_COLOR` to disable them. Redirected
output stays plain text. Each epoch ends with two mini-charts of the latest
40 evaluations on a shared loss scale (higher characters mean higher loss),
the best validation loss so far, and elapsed time. ASCII trends are used
when the output encoding cannot display Unicode.

**How the losses of the table are measured.** The training loss uses 5
random batches of the training set, different at each evaluation. The
validation loss uses a **fixed validation sample**: 100 sequences of 256
tokens (25 batches of 4), chosen once at regular intervals inside every book,
in proportion to the length of its validation part and at least one per book.
With the current corpus all 31 books are represented, from 1 sequence for
the shortest to 7 for the longest. The same sample at every evaluation makes
the validation losses comparable from one step to the next, and spreading it
over all the books makes it represent the whole validation set. The size is
`VAL_SAMPLE_SEQUENCES` in the script; the code is in
[`build_llm/validation_sample.py`](../build_llm/validation_sample.py).

At the end, it takes a random passage of the validation set, uses its
first 50 tokens as a prompt, and prints the 50 tokens generated by the model
next to the real continuation from the book. Run it from the repository
root:

```bash
python -m chap4.first_training
```

Training downloads nothing. It saves a checkpoint at the end of every epoch
(see below). With the current configuration
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

### Checkpoints and resume

At the end of every epoch the script saves a checkpoint in
`models/checkpoints/first_training/`, as `epoch-001.pt`, `epoch-002.pt`, and
so on. The `models/` folder is ignored by Git. A checkpoint holds everything
needed to continue as if the training had never stopped:

* the model weights;
* the AdamW state, two running averages per parameter;
* the random number generator states, which decide the batch order of the
  next epoch and the dropout masks;
* the step and token counters and the loss history;
* the settings that must not change between runs.

To resume an interrupted training from the most recent checkpoint, or from a
given one:

```bash
python -m chap4.first_training --resume
```

```bash
python -m chap4.first_training --resume models/checkpoints/first_training/epoch-002.pt
```

The training continues with the epoch after the checkpoint, up to
`NUM_EPOCHS`. A resumed run is exactly the same as an uninterrupted one: the
tests check that stopping after epoch 1 and resuming gives the same weights
after epoch 2, on the CPU. On MPS and CUDA, some operations are not
bit-for-bit deterministic, so the results can differ slightly. To train a
finished run for more epochs, raise `NUM_EPOCHS` and resume. If all the
epochs are already done, the script skips the training and only generates
the final sample from the restored model.

Things to know:

* **Size.** With weight tying, a checkpoint is about 1.5 GB: 0.5 GB of
  weights and 1 GB of AdamW state. The script keeps the 2 most recent ones
  (`KEEP_CHECKPOINTS`) and deletes the older files of the same folder after
  each save; set `KEEP_CHECKPOINTS = None` to keep them all.
* **Same settings.** A resume is refused, with the list of differences, if
  the model configuration, weight tying, batch size, split, learning rate,
  weight decay, gradient clipping, or the books (names and sizes) changed.
  Changing them would make it a different training. A setting added later,
  such as gradient clipping, matches older checkpoints only when it is
  disabled (`None`).
* **No overwriting.** A new run, without `--resume`, refuses to start if the
  checkpoint folder already holds checkpoints. To start a separate run,
  choose another folder with `--checkpoint-dir`, for example
  `--checkpoint-dir models/checkpoints/run-2`.
* **Safe loading.** Checkpoints are written to a temporary file and then
  renamed, so an interruption while saving never leaves a corrupted file.
  They are read with `torch.load(weights_only=True)`, which never executes
  code from the file.

The code is in [`build_llm/checkpoints.py`](../build_llm/checkpoints.py).

### Gradient clipping and learning rate schedules

**Gradient clipping** is enabled (`GRAD_CLIP_NORM = 1.0`). Before each
optimizer step, if the norm of all the gradients together is larger than 1.0,
the gradients are scaled down to that norm: their direction does not change,
only the size of the step. With batches of 4 sequences the gradients are
noisy, and a rare batch can produce a very large gradient that undoes many
steps of progress. Set `GRAD_CLIP_NORM = None` to train as in the book; that
is also needed to resume a checkpoint saved before this option existed.

**Learning rate schedules** are available in
[`build_llm/lr_schedule.py`](../build_llm/lr_schedule.py), but the script does
not use them yet: it still trains with a constant learning rate of 4e-4.
`learning_rate(step, ...)` supports two schedules, both with a linear warmup:

| Schedule | After the warmup | Total steps |
| --- | --- | --- |
| `"cosine"` | decreases along half a cosine wave, down to the minimum at the last step | fixed from the start: it sets the shape of the whole curve |
| `"wsd"` (warmup, stable, decay) | stays at the peak, then decreases linearly to the minimum in the last `decay_fraction` of the steps | can still change until the decay starts, so a run can be extended |

The learning rate is a pure function of the optimizer step, with no internal
state: a resumed training gets exactly the same learning rates as an
uninterrupted one from the step number saved in the checkpoint.

## Changelog

* **2026-10-31: gradient clipping and learning rate schedules.** Training
  clips the gradient norm to 1.0 before every optimizer step
  (`GRAD_CLIP_NORM`), and the setting is checked when a run resumes.
  `build_llm/lr_schedule.py` adds the cosine and WSD schedules, not yet used
  by the script. The run saved before this change, with a constant learning
  rate and no clipping, can be resumed only with `GRAD_CLIP_NORM = None`.
* **2026-10-31: checkpoints and resume.** The script saves a checkpoint at
  the end of every epoch, keeps the 2 most recent ones, and resumes from one
  with `--resume`; see [Checkpoints and resume](#checkpoints-and-resume).
  `train_model_simple` now returns a `TrainingState` instead of three lists,
  and `main` accepts the command-line arguments.
* **2026-10-31: fixed validation sample spread over all the books.** Until
  this change, every periodic evaluation computed the validation loss on the
  first 5 batches of the validation loader. That loader is not shuffled, and
  the books are sorted by file name, so the loss always measured the same
  5,120 tokens, 0.5% of the validation set: the end of "A Christmas Carol"
  and the beginning of the validation part of "Bleak House". Now the
  validation loss uses a fixed sample of 100 sequences (25 batches), chosen
  once at regular intervals inside each of the 31 books; see
  `build_llm/validation_sample.py`. The validation losses of earlier runs are
  not comparable with the new ones. Each evaluation now reads 30 batches
  instead of 10 (5 training and 25 validation), so with `EVAL_FREQ = 10` an
  epoch takes longer; a larger `EVAL_FREQ` reduces the cost.

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

The training checkpoints are in `models/checkpoints/first_training/`, about
1.5 GB each. Deleting the folder removes them, and with them the possibility
to resume that run:

```bash
rm -r models/checkpoints/first_training
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
