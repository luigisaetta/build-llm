# Chapter 4: pretrained weights and a first training run

In [Chapter 3](../chap3/README.md) we built `GPTModel`, a GPT-2-like model,
but its weights were random. This chapter gives it real weights, in two ways:

1. **Load the weights of GPT-2 Small**, published by OpenAI. With random
   weights the model writes nonsense; with the GPT-2 weights, the same
   architecture writes fluent English. This shows that our implementation is
   correct, down to the last matrix.
2. **Train the model ourselves, from random weights**, on a laptop: about 10
   million tokens of public-domain books, with everything a real training run
   needs — a reproducible corpus, a validation split, a learning rate
   schedule, gradient clipping, checkpoints to stop and resume, and a log of
   the losses.

The first part takes a few minutes. The second is a real training run: about
an hour per epoch on a MacBook with Apple Silicon, six epochs in all. Ten
million tokens are far fewer than GPT-2 saw, so do not expect GPT-2 quality:
the goal is to see, and measure, how a language model learns.

The chapter follows Chapter 5 of Sebastian Raschka's *Build a Large Language
Model (From Scratch)*, with some differences explained below: the weights are
read without TensorFlow, and the training adds weight tying, a learning rate
schedule, gradient clipping, and checkpoints. The chapter numbering of this
repository differs from the book.

## What is in this chapter

| File | What it does | Time |
| --- | --- | --- |
| [`load_gpt2_weights.ipynb`](load_gpt2_weights.ipynb) | Loads the GPT-2 Small weights into `GPTModel`, step by step, and generates text | minutes |
| [`learning_rate_schedules.ipynb`](learning_rate_schedules.ipynb) | Explains and draws the cosine and WSD learning rate schedules | seconds |
| [`prepare_books.py`](prepare_books.py) | Downloads and verifies the training books listed in the manifest | minutes, once |
| [`first_training.py`](first_training.py) | Trains the model from random weights, with checkpoints and a metrics file | hours |
| [`generate_from_checkpoint.py`](generate_from_checkpoint.py) | Completes a prompt with a saved checkpoint, to test it | seconds |
| [`books/`](books/README.md) and [`books_manifest.json`](books_manifest.json) | The 31 training books and their list, with sources and checksums | — |

The reusable code is in the `build_llm` package:

| Module | Content |
| --- | --- |
| [`gpt2.py`](../build_llm/gpt2.py) | `GPTModel`, with optional weight tying, and `generate_text_simple` |
| [`gpt2_weights.py`](../build_llm/gpt2_weights.py) | Download and conversion of the GPT-2 Small weights |
| [`book_corpus.py`](../build_llm/book_corpus.py) | The manifest of the corpus, and the check of the books folder |
| [`ebooks.py`](../build_llm/ebooks.py) | Conversion of Standard Ebooks editions to plain text |
| [`dataloaders.py`](../build_llm/dataloaders.py) | Sliding-window dataset, data loader, and the per-book split |
| [`validation_sample.py`](../build_llm/validation_sample.py) | The fixed validation sample spread over all the books |
| [`lr_schedule.py`](../build_llm/lr_schedule.py) | The cosine and WSD learning rate schedules |
| [`checkpoints.py`](../build_llm/checkpoints.py) | Saving, checking, and restoring training checkpoints |
| [`training_log.py`](../build_llm/training_log.py) | The CSV file of the losses |
| [`training_progress.py`](../build_llm/training_progress.py) | The progress table printed during training |

## Prerequisites

* The `build-llm` Conda environment, with the dependencies and the editable
  installation of the [Quickstart](../QUICKSTART.md). `safetensors` is in
  `requirements.txt`; `matplotlib`, for the charts of the schedules notebook,
  is in `requirements-dev.txt`.
* **Network access** the first time: about 548 MB of GPT-2 weights from
  Hugging Face, the downloaded training books from GitHub, and the GPT-2
  encoding of tiktoken if it is not cached.
* **Memory:** about 2 GB to load the GPT-2 weights; several GB for the
  training, which keeps the model, its gradients, and the AdamW state in
  memory.
* **Disk:** about 0.5 GB for the GPT-2 weights, about 40 MB for the books,
  and about 1.5 GB for each training checkpoint (2 are kept).

## Part 1: load the GPT-2 Small weights

[`load_gpt2_weights.ipynb`](load_gpt2_weights.ipynb) downloads
`model.safetensors` from
[`openai-community/gpt2`](https://huggingface.co/openai-community/gpt2),
inspects the tensors of the checkpoint, and maps them to the parameters of
`GPTModel`: it renames them, splits the fused query, key, and value
projection, transposes the `Conv1D` weights, and copies the token embeddings
into the output head. Then it loads the weights, tokenizes a prompt, and
generates a continuation with greedy decoding.

**A different path from the book.** OpenAI released the original weights as
TensorFlow checkpoints, and the book reads them with TensorFlow. This
repository does not use TensorFlow: it reads the same weights, published by
Hugging Face in the [safetensors](https://huggingface.co/docs/safetensors)
format, with the small `safetensors` library. The numbers are the same, so
the model is the same; the notebook explains the differences in names and
layout.

From the repository root:

```bash
conda activate build-llm
```

```bash
python -m jupyterlab
```

Open the notebook with the kernel of the `build-llm` environment and run it
from top to bottom. Expected result: the prompt "Every effort moves you"
continues with "forward. The first step is to understand the importance of
your work.", and a second prompt shows the repetitions typical of greedy
decoding.

The weights are saved in `models/gpt2-small/model.safetensors`, in the
repository root; `models/` is ignored by Git. The download uses a fixed
revision of the Hugging Face repository, and the file is accepted only if its
SHA-256 checksum matches the published one.

## Part 2: train a GPT model from scratch

### Quick start

From the repository root, with the `build-llm` environment active:

1. Prepare the books, once. The command downloads the 19 books that are not
   committed to the repository, and verifies all 31:

   ```bash
   python -m chap4.prepare_books
   ```

2. Train. `caffeinate -i` keeps the Mac awake while the command runs: during
   a sleep the training is suspended. Keep the lid open.

   ```bash
   caffeinate -i python -m chap4.first_training
   ```

3. To stop, press Ctrl-C right after a "Checkpoint saved" line, at the end of
   an epoch. To continue later from the last checkpoint:

   ```bash
   caffeinate -i python -m chap4.first_training --resume
   ```

4. Test what the model has learned, at any time, even during the training:

   ```bash
   python -m chap4.generate_from_checkpoint "It was a dark and stormy night" --device cpu
   ```

The next sections explain each part.

### The corpus

The model is trained on **31 complete public-domain books, 10,056,985 GPT-2
tokens**: novels and novellas in English or in English translation, from
"The Time Machine" to "War and Peace", converted to plain text from the
carefully proofread editions of [Standard Ebooks](https://standardebooks.org/).
The [books README](books/README.md) lists them, with their token counts and
the criteria of the choice; [NOTICE](../NOTICE) records their provenance and
licenses.

**The manifest defines the corpus.**
[`books_manifest.json`](books_manifest.json) lists all 31 books, each with its
Standard Ebooks repository, the exact revision used, and the SHA-256 checksum
of the converted text. Twelve books are committed to this repository; the
other nineteen are downloaded and converted again by `prepare_books`, which
verifies every checksum. `first_training.py` reads the list of books from the
manifest and, before training, checks the folder: a missing book, a modified
book, or an extra `.txt` file stops the run with the list of problems. The
training data can never change silently, and anyone can rebuild exactly the
same corpus.

**The split is made per book.** The first 90% of each book goes to training,
the last 10% to validation. Splitting the joined text once would put only the
end of the last book in the validation set. Inside each set the books are
joined with the GPT-2 end-of-text token, `<|endoftext|>`, which marks where a
text ends and the next begins. The result:

| Set | Tokens | Batches per epoch |
| --- | --- | --- |
| Training | 9,058,141 | 8,845 |
| Validation | 998,936 | 976 |

The text is cut into non-overlapping sequences of 256 tokens, the context of
the model, in batches of 4 sequences: about 1,000 tokens per optimizer step.

### The model

The model has the dimensions of GPT-2 Small — 12 transformer blocks, 12
attention heads, embeddings of 768 values, the 50,257-token vocabulary of
GPT-2 — with a context of 256 tokens instead of 1,024, to make each step
faster (`GPT_CONFIG_SMALL_256`).

**Weight tying** is enabled (`TIE_EMBEDDINGS = True`), an addition to the
book. The token embedding table and the output head are both
`(vocab_size, emb_dim)` matrices; with weight tying they are one shared
matrix, as in the original GPT-2:

* the model has **123,822,336 parameters** instead of 162,419,712;
* every update of the output head also improves the token embeddings, which
  helps when the training data is small;
* the shared matrix is initialized as in GPT-2, with standard deviation 0.02;
  the default initialization of `nn.Embedding` would make the first loss
  about 475 instead of about 11, that is ln(50,257).

Set `TIE_EMBEDDINGS = False` to train the model of the book.

### How the model is trained

| Setting | Value | Constant |
| --- | --- | --- |
| Epochs | 6, that is 53,070 optimizer steps | `NUM_EPOCHS` |
| Optimizer | AdamW, weight decay 0.1 | `WEIGHT_DECAY` |
| Learning rate | cosine: warmup to 4e-4 over 2% of the steps, then down to 4e-5 | `LR_SCHEDULE`, `LEARNING_RATE`, `MIN_LEARNING_RATE`, `WARMUP_FRACTION` |
| Gradient clipping | norm 1.0 | `GRAD_CLIP_NORM` |
| Batch | 4 sequences of 256 tokens | `BATCH_SIZE` |
| Evaluation | every 50 steps, and at the end of each epoch | `EVAL_FREQ` |

* **Learning rate schedule.** The learning rate grows linearly in the first
  1,061 steps, the warmup, then decreases along half a cosine wave to 10% of
  its peak at the last step. Large steps learn fast at the start; small steps
  at the end let the model settle. The cosine spreads the decay over the whole
  run, so `NUM_EPOCHS` must be the number of epochs that the run will really
  complete. The learning rate is computed from the step number before every
  step, so a resumed run gets exactly the same values. The
  [schedules notebook](learning_rate_schedules.ipynb) draws the curve and
  compares it with WSD, the other schedule available
  (`LR_SCHEDULE = "wsd"`).
* **Gradient clipping.** Before each step, if the norm of all the gradients
  together is larger than 1.0, they are scaled down to that norm: same
  direction, smaller step. With batches this small the gradients are noisy,
  and a rare batch could produce a huge gradient that undoes many steps of
  progress.
* **Evaluation.** The training loss uses 5 random training batches, different
  each time. The validation loss uses a **fixed validation sample**: 100
  sequences (25 batches), chosen once at regular intervals inside every book,
  in proportion to its length and at least one per book. The same sample at
  every evaluation makes the losses comparable from step to step, and
  spreading it over all the 31 books makes it represent the whole validation
  set (`VAL_SAMPLE_SEQUENCES`).

The [questions and answers](../docs/questions-and-answers.md#chapter-4-pretrained-weights-and-training)
explain gradient clipping and the two schedules in more detail.

### Reading the output

The script prints:

1. a summary of the run: device, model, parameter count, how the run starts
   (new, or resumed from a checkpoint), epochs, context, learning rate
   schedule, gradient clipping, split, and checkpoint folder;
2. the table of the books, with characters and tokens;
3. the tokens and batches of the training set, the validation set, and the
   validation sample;
4. for each epoch, a progress table, with a row at every evaluation:

   | Column | Meaning |
   | --- | --- |
   | Progress | how much of the epoch is done |
   | Step | one-based optimizer step, over the whole run |
   | Learn. rate | learning rate of the last step |
   | Train, Val | training and validation losses |
   | Gap | validation minus training loss: it grows when the model starts to memorize |
   | Validation | "new best" when the validation loss reaches a new minimum |

5. at the end of each epoch, two mini-charts of the last 40 evaluations, the
   best validation loss so far, the duration of the epoch, and the path of the
   saved checkpoint;
6. at the end of the training, a random passage of the validation set: its
   first 50 tokens as a prompt, the 50 tokens generated by the model, and the
   real continuation from the book.

Terminals show colors; set `NO_COLOR` to disable them. Redirected output is
plain text.

### Checkpoints and resume

At the end of every epoch the script saves a checkpoint in
`models/checkpoints/first_training/`, as `epoch-001.pt`, `epoch-002.pt`, and
so on. A checkpoint holds everything needed to continue as if the training
had never stopped: the model weights, the AdamW state, the random number
generator states (they decide the batch order of the next epoch and the
dropout masks), the step and token counters, the loss history, and the
settings of the run.

A typical scenario: start a 6-epoch run, press Ctrl-C after the checkpoint of
epoch 3, and the next day resume:

```bash
python -m chap4.first_training --resume
```

```bash
python -m chap4.first_training --resume models/checkpoints/first_training/epoch-002.pt
```

The first command resumes from the most recent checkpoint, the second from a
given one. The training continues with the next epoch, up to `NUM_EPOCHS`,
and the result is the same as an uninterrupted run: the tests check that the
weights and the metrics file are identical, on the CPU. On MPS and CUDA some
operations are not bit-for-bit deterministic, so the results can differ
slightly. Ctrl-C prints the command to resume; the work done since the last
checkpoint is lost.

Rules that protect a run:

* **Same settings.** A resume is refused, with the list of differences, if
  the model, weight tying, batch size, split, number of epochs, learning rate
  schedule, weight decay, gradient clipping, or the books changed: it would be
  a different training. In particular, `NUM_EPOCHS` cannot change, because
  the cosine schedule depends on the total number of steps.
* **No overwriting.** A new run, without `--resume`, refuses to start if the
  checkpoint folder already holds checkpoints. To start a separate run, choose
  another folder, for example `--checkpoint-dir models/checkpoints/run-2`.
* **Size.** A checkpoint is about 1.5 GB: 0.5 GB of weights and 1 GB of AdamW
  state. The 2 most recent ones are kept (`KEEP_CHECKPOINTS`); older files of
  the same folder are deleted after each save. `None` keeps them all.
* **Safe files.** Checkpoints are written to a temporary file and then
  renamed, so the most recent one is always complete, even while the next one
  is being written. They are read with `torch.load(weights_only=True)`, which
  never executes code from the file.

### The metrics file: losses to plot

Every evaluation is also written to `metrics.csv`, in the checkpoint folder,
one row per evaluation, with the columns `step`, `epoch`, `tokens_seen`,
`learning_rate`, `train_loss`, and `val_loss`. Each row is written to disk at
once, so the file can be read and plotted while the training runs. A new run
starts an empty file; a resumed run keeps the rows up to its checkpoint and
drops the later ones, so the file always describes one continuous run. Read
it with `read_metrics`:

```python
from build_llm.training_log import read_metrics

rows = read_metrics("models/checkpoints/first_training/metrics.csv")
steps = [row["step"] for row in rows]
validation = [row["val_loss"] for row in rows]
```

### Test a checkpoint: complete a prompt

[`generate_from_checkpoint.py`](generate_from_checkpoint.py) loads a
checkpoint, by default the most recent one, and completes an English prompt
with greedy decoding:

```bash
python -m chap4.generate_from_checkpoint "It was a dark and stormy night"
```

```bash
python -m chap4.generate_from_checkpoint "Mr. Darcy looked at Elizabeth" --max-new-tokens 100 --checkpoint models/checkpoints/first_training/epoch-001.pt
```

Options: `--max-new-tokens` (default 50), `--checkpoint` for a given file,
`--checkpoint-dir` for the folder of another run, and `--device cpu` to leave
the GPU to a running training. The script rebuilds the model from the
configuration stored in the checkpoint and reads only its weights. It prints
the checkpoint, its epoch and steps, and its last and best validation losses,
then the prompt followed by the generated text. Generation stops at the
maximum number of tokens, or at the end-of-text token that separates the
books.

### Results so far

The run with the cosine schedule and gradient clipping has not been recorded
yet.

A first baseline run, with a constant learning rate of 4e-4 and no clipping,
took about an hour per epoch on a MacBook with Apple Silicon (MPS). Its
validation loss went from about 4.36 after epoch 1 to about 3.98 after epoch
2; its checkpoints were then deleted. The greedy completions were grammatical
but repetitive, for example:

```text
Mr. Darcy looked at Elizabeth and said, “I am not a friend.”

“I am not a friend,” said Elizabeth, “but I am not a friend.”
```

With much less data, "The Time Machine" alone (about 41,600 tokens), the
training loss fell below 0.2 in 10 epochs while the validation loss reached
its minimum, about 5.7, around epoch 4, and then rose: a clear example of a
model that memorizes its data instead of generalizing.

## Cleanup

| What | Where | Size | Command |
| --- | --- | --- | --- |
| GPT-2 weights | `models/gpt2-small/` | about 0.5 GB | `rm -r models/gpt2-small` |
| Training checkpoints and metrics | `models/checkpoints/first_training/` | about 1.5 GB per checkpoint | `rm -r models/checkpoints/first_training` |
| Downloaded books | the 19 `.txt` files of `chap4/books/` ignored by Git | about 33 MB | `git clean -nX chap4/books` to list them, then `git clean -fX chap4/books` |

Deleting the checkpoints also removes the possibility to resume that run.
`git clean -X` deletes only the files ignored by Git, so the 12 committed
books stay; the downloaded ones can be restored with
`python -m chap4.prepare_books`.

## Troubleshooting

* **The training seems stuck, or very slow.** The Mac probably went to sleep:
  the process is suspended and resumes when the Mac wakes up. Run the training
  with `caffeinate -i`, and keep the lid open.
* **"The books ... do not match the manifest"**: run
  `python -m chap4.prepare_books`. It restores missing downloaded books and
  reports modified or extra files; a committed book that is missing is
  restored with `git checkout -- chap4/books/FILE`.
* **"Cannot resume: The checkpoint was saved with different settings"**: the
  message lists what changed. Restore the old values, or start a new run in
  another folder with `--checkpoint-dir`.
* **"... already holds checkpoints of another run"**: resume it with
  `--resume`, or choose another `--checkpoint-dir`.
* **`ModuleNotFoundError: safetensors`** or **`matplotlib`**: install the
  dependencies again with `pip install -r requirements.txt -r requirements-dev.txt`,
  then restart the kernel.
* **`Checksum mismatch`** while downloading the GPT-2 weights: the download
  was corrupted and the partial file removed. Run the cell again.

## Scope and next steps

Greedy decoding always picks the most likely token: the output is
deterministic but repeats itself. The next steps are:

* temperature and top-k sampling, for more varied text;
* the results of the full run, with a plot of the losses from `metrics.csv`;
* other ways to train better on a laptop: gradient accumulation, for larger
  effective batches, and the WSD schedule, which allows longer runs;
* perplexity, to compare models in a more readable unit than the loss.

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

The tests run without network access: they use tiny models, small fake books,
and fake downloads. A few tests, marked `network`, need it: for example, one
checks, without downloading the weights, that Hugging Face still publishes
the expected file. Run them with:

```bash
pytest -m network
```

The test suite does not execute the notebooks. Verify them separately with a
fresh kernel and **Run All Cells**.

## Changelog

* **2026-10-31: chapter README rewritten.** The README describes the two
  parts of the chapter, pretrained weights and training from scratch, with a
  quick start for the training.
* **2026-10-31: cosine schedule and metrics file.** Training uses the cosine
  learning rate schedule (warmup over 2% of the steps, peak 4e-4, minimum
  4e-5), shown in the progress table. `NUM_EPOCHS` and the schedule are now
  checked when a run resumes, so a run can no longer be extended by raising
  `NUM_EPOCHS`. Every evaluation is written to `metrics.csv` next to the
  checkpoints; see [The metrics file](#the-metrics-file-losses-to-plot).
  Ctrl-C stops with the command to resume. The checkpoints of the baseline run
  were deleted.
* **2026-10-31: the manifest defines the corpus.** `chap4/books_manifest.json`
  lists all 31 books, including the 12 committed to the repository, with their
  checksums. `prepare_books` verifies all of them, and `first_training.py`
  reads the list from the manifest and stops if a book is missing or
  modified, or if the folder holds a `.txt` file that is not listed; see
  [the books README](books/README.md). The code is in
  `build_llm/book_corpus.py`.
* **2026-10-31: complete a prompt from a checkpoint.**
  `generate_from_checkpoint.py` completes an English prompt with the most
  recent checkpoint, or a given one; see
  [Test a checkpoint](#test-a-checkpoint-complete-a-prompt).
* **2026-10-31: notebook on learning rate schedules.**
  `learning_rate_schedules.ipynb` explains and draws the cosine and WSD
  schedules. `matplotlib` is added to `requirements-dev.txt` for its charts.
* **2026-10-31: gradient clipping and learning rate schedules.** Training
  clips the gradient norm to 1.0 before every optimizer step
  (`GRAD_CLIP_NORM`). `build_llm/lr_schedule.py` adds the cosine and WSD
  schedules.
* **2026-10-31: checkpoints and resume.** The script saves a checkpoint at
  the end of every epoch, keeps the 2 most recent ones, and resumes from one
  with `--resume`; see [Checkpoints and resume](#checkpoints-and-resume).
* **2026-10-31: fixed validation sample spread over all the books.** Until
  this change, every periodic evaluation computed the validation loss on the
  first 5 batches of the validation loader, always the same 5,120 tokens of
  two books. Now it uses a fixed sample of 100 sequences, chosen once at
  regular intervals inside each of the 31 books. The validation losses of
  earlier runs are not comparable with the new ones.
