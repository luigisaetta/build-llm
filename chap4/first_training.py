"""
Author: L. Saetta
Date Last Modified: 2026-10-31
License: Apache-2.0
Description: A first training loop: pre-train a GPT-2 Small-sized model, from
    random weights, on public-domain books in books/ (see NOTICE).

Origin: derived from the code of Chapter 5 of LLMs-from-scratch
(https://github.com/rasbt/LLMs-from-scratch), Copyright 2023-2026 Sebastian
Raschka, licensed under the Apache License 2.0 (see
LICENSES/Apache-2.0-LLMs-from-scratch.txt). Modified by L. Saetta (2026):
docstrings added, the data path made independent of the working directory,
the code organized in functions with a main entry point, a training and
validation split made per book, and weight tying enabled (TIE_EMBEDDINGS).

Run it from the repository root:

    python -m chap4.first_training

A checkpoint is saved at the end of every epoch, in CHECKPOINT_DIR. To resume
an interrupted training from the most recent checkpoint, or from a given one:

    python -m chap4.first_training --resume
    python -m chap4.first_training --resume PATH/TO/epoch-002.pt

Weight tying: the output head shares its weights with the token embeddings,
as in the original GPT-2 and unlike the book. The model has about 124M
parameters instead of about 163M, with the same architecture; see
TIE_EMBEDDINGS below.

The dataset (over 10 million GPT-2 tokens) is still small for a model of about 124M
parameters, so after some epochs the model tends to memorize the text: the
training loss keeps going down, while the validation loss stops improving.
That is expected here.
"""

import argparse
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import tiktoken
import torch
from torch.utils.data import DataLoader

from build_llm.checkpoints import (
    checkpoint_path,
    latest_checkpoint,
    list_checkpoints,
    load_checkpoint,
    remove_old_checkpoints,
    save_checkpoint,
)
from build_llm.dataloaders import (
    END_OF_TEXT,
    create_dataloader_v1,
    split_document_parts,
)
from build_llm.device import get_device
from build_llm.gpt2 import GPTModel, generate_text_simple
from build_llm.model_parameters import count_parameters
from build_llm.training_progress import TrainingProgress
from build_llm.validation_sample import create_validation_sample_loader

# The dimensions of GPT-2 Small, with a context of 256 tokens instead of 1024:
# shorter sequences make each training step faster and use less memory.
GPT_CONFIG_SMALL_256 = {
    "vocab_size": 50257,  # Vocabulary size of the GPT-2 BPE tokenizer
    "context_length": 256,  # Shortened context length (original: 1024)
    "emb_dim": 768,  # Embedding dimension
    "n_heads": 12,  # Number of attention heads
    "n_layers": 12,  # Number of transformer blocks
    "drop_rate": 0.1,  # Dropout rate
    "qkv_bias": False,  # Bias in the query, key, and value projections
}

# Weight tying, an addition to the book (see build_llm/gpt2.py). The output
# head turns each final vector into one score per vocabulary token, and the
# token embedding table turns each token into a vector: both are
# (vocab_size, emb_dim) matrices. With TIE_EMBEDDINGS = True they are one
# shared matrix, as in GPT-2:
# * the model has vocab_size * emb_dim = 50,257 * 768 = 38,597,376 fewer
#   parameters: about 124M instead of about 163M;
# * every update of the output head also updates the embeddings of the
#   tokens, which helps when, as here, the training data is small;
# * GPTModel initializes the shared matrix as GPT-2 does (standard deviation
#   0.02), so that the first loss is close to ln(vocab_size), about 10.8.
# Set it to False to train the model of the book, with a separate output head.
TIE_EMBEDDINGS = True

# Load the local .txt books dynamically; split each one independently for validation.
BOOKS_DIR = Path(__file__).resolve().parent / "books"

# Checkpoints, one at the end of every epoch. They go under models/, which is
# ignored by Git. With weight tying, a checkpoint holds the weights (about
# 0.5 GB) and the two AdamW running averages (about 1 GB): about 1.5 GB each.
CHECKPOINT_DIR = (
    Path(__file__).resolve().parent.parent / "models" / "checkpoints" / "first_training"
)
# The most recent checkpoints to keep; older ones are deleted after each save.
# Set it to None to keep them all.
KEEP_CHECKPOINTS: int | None = 2

TRAIN_RATIO = 0.90  # Fraction of each book used for training; the rest validates
# doubled
BATCH_SIZE = 4
# reduced
NUM_EPOCHS = 6
LEARNING_RATE = 0.0004
WEIGHT_DECAY = 0.1
# Gradient clipping: before each optimizer step, if the norm of all the
# gradients together is larger than GRAD_CLIP_NORM, the gradients are scaled
# down to that norm. Their direction does not change, only the size of the
# step. With batches of 4 sequences the gradients are noisy, and a rare batch
# can produce a very large gradient that undoes many steps of progress; the
# clipping limits the damage. 1.0 is the usual value for GPT models. Set it to
# None to disable it, as in the book and in the runs saved before this option.
GRAD_CLIP_NORM: float | None = 1.0
EVAL_FREQ = 50  # Evaluate every EVAL_FREQ optimizer steps
EVAL_ITER = 5  # Training batches used for each evaluation
# The validation loss of each evaluation uses a fixed sample of validation
# sequences, chosen once at regular intervals inside every book: 100 sequences
# are 25 batches of 4. The same sample at every evaluation makes the losses
# comparable from one step to the next; spreading it over all the books makes
# it represent the whole validation set. See build_llm/validation_sample.py.
VAL_SAMPLE_SEQUENCES = 100
PROMPT_TOKENS = 50  # Tokens of the validation sample used as the prompt
NEW_TOKENS = 50  # Tokens generated after the prompt


@dataclass
class TrainingState:
    """Progress of a training run: what a checkpoint needs to continue it.

    Attributes:
        global_step: Zero-based index of the last optimizer step; -1 before
            the first one.
        tokens_seen: Input tokens processed so far.
        train_losses: Training loss of each evaluation.
        val_losses: Validation loss of each evaluation.
        track_tokens_seen: Tokens seen at each evaluation.
    """

    global_step: int = -1
    tokens_seen: int = 0
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    track_tokens_seen: list[int] = field(default_factory=list)


#
# Functions
#
def find_book_paths() -> list[Path]:
    """Find local training books at runtime in a deterministic order.

    Returns:
        Regular .txt files directly inside BOOKS_DIR, sorted by filename.
        An empty list if the directory is missing or contains no text files.
    """
    return sorted(path for path in BOOKS_DIR.glob("*.txt") if path.is_file())


def calc_loss_batch(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: torch.nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """Compute the cross-entropy loss of one batch.

    Args:
        input_batch: Token IDs of shape (batch_size, num_tokens).
        target_batch: The next token for each input position, same shape.
        model: A model that returns logits of shape
            (batch_size, num_tokens, vocab_size).
        device: The device of the model.

    Returns:
        The mean loss over all the token positions of the batch, as a scalar
        tensor that supports backward().
    """
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    # cross_entropy expects (N, vocab_size) logits and (N,) targets: flatten the
    # batch and token dimensions into one.
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss


def calc_loss_loader(
    data_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    num_batches: int | None = None,
) -> float:
    """Compute the average loss over the first batches of a data loader.

    Args:
        data_loader: Yields (inputs, targets) batches.
        model: The model to evaluate.
        device: The device of the model.
        num_batches: Number of batches to use; all of them if None, or if it is
            larger than the number of batches.

    Returns:
        The average batch loss, or NaN if the data loader is empty.
    """
    if len(data_loader) == 0:
        return float("nan")
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    total_loss = 0.0
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()
    return total_loss / num_batches


def evaluate_model(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    eval_iter: int,
) -> tuple[float, float]:
    """Compute the training and validation losses, without dropout or gradients.

    The training loss uses the first eval_iter batches of the shuffled training
    loader, so a different random subset each time. The validation loss uses
    all the batches of val_loader: pass the fixed validation sample, so that
    every evaluation measures the same sequences, spread over all the books.
    The model is switched to evaluation mode, then back to training mode.

    Args:
        model: The model being trained.
        train_loader: Training batches.
        val_loader: The fixed validation sample, evaluated in full.
        device: The device of the model.
        eval_iter: Number of training batches to use.

    Returns:
        The training loss and the validation loss.
    """
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(val_loader, model, device)
    model.train()
    return train_loss, val_loss


def synchronize(device: torch.device) -> None:
    """Wait until all the queued operations on the device are finished.

    On a GPU (MPS or CUDA), PyTorch queues the operations and returns before
    they run. Without waiting, a timer would stop before the last training
    steps are actually done. On the CPU, operations already run in order.

    Args:
        device: The device of the model.
    """
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def generate_from_validation_sample(  # pylint: disable=too-many-arguments
    model: GPTModel,
    val_loader: torch.utils.data.DataLoader,
    tokenizer: tiktoken.Encoding,
    device: torch.device,
    *,
    prompt_tokens: int,
    new_tokens: int,
) -> None:
    """Continue a random validation passage and compare it with the original.

    A random sequence of the validation set is chosen; its first prompt_tokens
    tokens are the prompt. The model generates new_tokens tokens with greedy
    decoding, and the real continuation from the book is printed next to them.
    The validation text was never used for training, so the model has not
    memorized it.

    Args:
        model: The trained model, already on device.
        val_loader: Validation batches; its dataset holds the sequences.
        tokenizer: The GPT-2 tokenizer.
        device: The device of the model.
        prompt_tokens: Number of tokens of the prompt.
        new_tokens: Number of tokens to generate.
    """
    dataset = val_loader.dataset
    # torch.randint follows the seed set in main(), so the choice is repeatable
    index = int(torch.randint(len(dataset), (1,)))
    sequence = dataset[index][0]  # Input token IDs, shape (context_length,)
    prompt = sequence[:prompt_tokens]
    real_continuation = sequence[prompt_tokens : prompt_tokens + new_tokens]

    model.eval()  # Disable dropout for generation
    output = generate_text_simple(
        model,
        prompt.unsqueeze(0).to(device),  # Add the batch dimension: (1, tokens)
        max_new_tokens=new_tokens,
        context_size=model.pos_emb.num_embeddings,
    )
    generated = output[0, prompt_tokens:]

    print(f"Validation sample {index} of {len(dataset)}")
    print("Prompt:")
    print(tokenizer.decode(prompt.tolist()))
    print()
    print("Generated continuation:")
    print(tokenizer.decode(generated.tolist()))
    print()
    print("Real continuation, from the book:")
    print(tokenizer.decode(real_continuation.tolist()))


#
# Training loop
#
def train_model_simple(  # pylint: disable=too-many-arguments,too-many-locals
    model: GPTModel,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
    grad_clip_norm: float | None = None,
    start_epoch: int = 0,
    state: TrainingState | None = None,
    on_epoch_end: Callable[[int, TrainingState], None] | None = None,
) -> TrainingState:
    """Train the model, evaluating it periodically.

    Every eval_freq optimizer steps, the training and validation losses are
    computed and shown in a progress table. The final step of each epoch is
    also evaluated. Each epoch ends with loss trends and its duration,
    evaluations included.

    Args:
        model: The model to train, already on device.
        train_loader: Training batches.
        val_loader: The fixed validation sample, evaluated in full at every
            evaluation.
        optimizer: The optimizer of the model parameters.
        device: The device of the model.
        num_epochs: Number of passes over the training data.
        eval_freq: Number of optimizer steps between two evaluations.
        eval_iter: Number of batches used for each evaluation.
        grad_clip_norm: Maximum norm of the gradients before each optimizer
            step; None disables gradient clipping.
        start_epoch: Number of epochs already completed, when resuming from a
            checkpoint; training continues with the next one.
        state: The progress restored from a checkpoint; a new one if None.
        on_epoch_end: Called after each completed epoch with its one-based
            number and the current state, for example to save a checkpoint.

    Returns:
        The final training state, with the losses of every evaluation.
    """
    state = TrainingState() if state is None else state
    progress = TrainingProgress()
    if state.val_losses:  # Resumed: the best validation loss so far is known
        progress.best_validation = min(state.val_losses)

    for epoch in range(start_epoch, num_epochs):
        progress.start_epoch(epoch + 1, num_epochs)
        model.train()  # Enable dropout
        epoch_start = time.perf_counter()

        for batch_number, (input_batch, target_batch) in enumerate(
            train_loader, start=1
        ):
            optimizer.zero_grad()  # Reset the gradients of the previous step

            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Compute the gradients of the loss
            if grad_clip_norm is not None:
                # Scale all the gradients together if their norm is too large
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()  # Update the weights using the gradients

            state.tokens_seen += input_batch.numel()
            state.global_step += 1

            # Include the final batch so every epoch reports its completed progress.
            if state.global_step % eval_freq == 0 or batch_number == len(train_loader):
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                state.train_losses.append(train_loss)
                state.val_losses.append(val_loss)
                state.track_tokens_seen.append(state.tokens_seen)
                progress.update(
                    batch_number,
                    len(train_loader),
                    state.global_step + 1,
                    train_loss,
                    val_loss,
                )

        synchronize(device)  # Count the steps still running on the GPU
        epoch_seconds = time.perf_counter() - epoch_start
        progress.finish_epoch(epoch_seconds, state.train_losses, state.val_losses)
        if on_epoch_end is not None:
            on_epoch_end(epoch + 1, state)

    return state


def create_loaders(
    tokenizer: tiktoken.Encoding, text_paths: list[Path], progress: TrainingProgress
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Read and split the books, and create the training and validation loaders.

    Each book is split on its own: the first TRAIN_RATIO of every book trains,
    the rest validates. Splitting the concatenated text once would put only
    the end of the last book in the validation set. Inside each set, the books
    are separated by the GPT-2 end-of-text token. The books, and the token and
    batch counts of each set, are printed with progress.

    Args:
        tokenizer: The GPT-2 tokenizer.
        text_paths: The book files.
        progress: The printer of the training summary.

    Returns:
        The training loader (shuffled), the full validation loader (not
        shuffled, used for the final generation), and the fixed validation
        sample used by the periodic evaluations.
    """
    books = [path.read_text(encoding="utf-8") for path in text_paths]
    progress.show_books(
        [
            (path.name, len(book), len(tokenizer.encode(book)))
            for path, book in zip(text_paths, books)
        ]
    )

    train_parts, val_parts = split_document_parts(books, TRAIN_RATIO)
    train_text = END_OF_TEXT.join(train_parts)
    val_text = END_OF_TEXT.join(val_parts)
    context_length = GPT_CONFIG_SMALL_256["context_length"]

    train_loader = create_dataloader_v1(
        tokenizer,
        train_text,
        batch_size=BATCH_SIZE,
        max_length=context_length,
        stride=context_length,
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )
    val_loader = create_dataloader_v1(
        tokenizer,
        val_text,
        batch_size=BATCH_SIZE,
        max_length=context_length,
        stride=context_length,
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )
    # The fixed sample of the periodic validation loss: the same sequences at
    # every evaluation, taken at regular intervals inside every book
    val_sample_loader = create_validation_sample_loader(
        tokenizer,
        val_parts,
        num_sequences=VAL_SAMPLE_SEQUENCES,
        batch_size=BATCH_SIZE,
        max_length=context_length,
    )

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text, allowed_special={END_OF_TEXT}))

    progress.show_split(
        [
            ("Training", count_tokens(train_text), len(train_loader)),
            ("Validation", count_tokens(val_text), len(val_loader)),
            (
                "Val. sample",
                len(val_sample_loader.dataset) * context_length,
                len(val_sample_loader),
            ),
        ]
    )
    return train_loader, val_loader, val_sample_loader


#
# Checkpoints and command line
#
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the command-line options.

    Args:
        argv: The arguments, without the program name; sys.argv if None.

    Returns:
        The options: resume (None, "latest", or a checkpoint path) and
        checkpoint_dir.
    """
    parser = argparse.ArgumentParser(
        description="Train a GPT-2 Small-sized model on the books in chap4/books.",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        default=None,
        metavar="CHECKPOINT",
        help=(
            "resume from a checkpoint: the most recent one in --checkpoint-dir "
            "if no path is given"
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=CHECKPOINT_DIR,
        help=f"folder of the checkpoints of this run (default: {CHECKPOINT_DIR})",
    )
    return parser.parse_args(argv)


def training_settings(text_paths: list[Path]) -> dict:
    """Collect the settings that must not change when a training resumes.

    The model configuration must match the saved weights. The batch size, the
    split, and the books decide the batches of every epoch, and the learning
    rate and weight decay are stored in the optimizer state, and gradient
    clipping changes every update: changing them in the middle of a run would
    make it a different training. NUM_EPOCHS is not included, so a finished
    run can be extended with more epochs.

    A setting that is None now and absent from an older checkpoint counts as
    equal, so runs saved before an option existed can still be resumed with
    the option disabled.

    Args:
        text_paths: The book files used for training.

    Returns:
        The settings, made of plain values that a checkpoint can store.
    """
    return {
        "model_config": dict(GPT_CONFIG_SMALL_256),
        "tie_embeddings": TIE_EMBEDDINGS,
        "batch_size": BATCH_SIZE,
        "train_ratio": TRAIN_RATIO,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip_norm": GRAD_CLIP_NORM,
        # File names and sizes: a cheap fingerprint of the corpus
        "books": [f"{path.name}:{path.stat().st_size}" for path in text_paths],
    }


def find_resume_checkpoint(resume: str, checkpoint_dir: Path) -> Path:
    """Resolve the --resume option to a checkpoint file.

    Args:
        resume: "latest", or the path of a checkpoint.
        checkpoint_dir: The folder searched for the latest checkpoint.

    Returns:
        The checkpoint path.

    Raises:
        FileNotFoundError: If there is no checkpoint to resume from.
    """
    if resume == "latest":
        path = latest_checkpoint(checkpoint_dir)
        if path is None:
            raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}.")
        return path
    path = Path(resume)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def make_checkpoint_saver(  # pylint: disable=too-many-arguments
    checkpoint_dir: Path,
    *,
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    settings: dict,
    keep: int | None,
) -> Callable[[int, TrainingState], None]:
    """Return the function that saves a checkpoint at the end of an epoch.

    Args:
        checkpoint_dir: Folder of the checkpoints of this run.
        model: The model being trained.
        optimizer: Its optimizer.
        device: The device of the model.
        settings: The settings of this run, stored for check_settings.
        keep: Number of most recent checkpoints to keep; None keeps them all.

    Returns:
        A function of the epoch number and the training state, for the
        on_epoch_end argument of train_model_simple.
    """

    def save(epoch: int, state: TrainingState) -> None:
        path = checkpoint_path(checkpoint_dir, epoch)
        save_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            device=device,
            training_state=asdict(state),
            settings=settings,
        )
        print(f"  Checkpoint saved: {path}", flush=True)
        if keep is not None:
            for old in remove_old_checkpoints(checkpoint_dir, keep):
                print(f"  Old checkpoint removed: {old.name}", flush=True)

    return save


def resume_training(
    args: argparse.Namespace,
    *,
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    settings: dict,
) -> tuple[int, TrainingState, str]:
    """Restore a checkpoint chosen with --resume into the model and optimizer.

    Args:
        args: The command-line options.
        model: The model, already on device.
        optimizer: Its optimizer.
        device: The device of the model.
        settings: The settings of this run, compared with the saved ones.

    Returns:
        The number of completed epochs, the restored training state, and a
        description for the summary.

    Raises:
        FileNotFoundError: If there is no checkpoint to resume from.
        ValueError: If the checkpoint was saved with different settings.
    """
    path = find_resume_checkpoint(args.resume, args.checkpoint_dir)
    restored = load_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        device=device,
        expected_settings=settings,
    )
    epoch = restored["epoch"]
    state = TrainingState(**restored["training_state"])
    return epoch, state, f"resumed from {path.name} | epoch {epoch} done"


def show_summary(
    progress: TrainingProgress,
    device: torch.device,
    model: GPTModel,
    start: str,
    checkpoint_dir: Path,
) -> None:
    """Print the title and the configuration of the run.

    Args:
        progress: The printer of the training summary.
        device: The device of the model.
        model: The model, to show its parameter count.
        start: How the run starts: new, or resumed from a checkpoint.
        checkpoint_dir: Folder of the checkpoints of this run.
    """
    tying = "weight tying" if TIE_EMBEDDINGS else "separate output head"
    progress.start_demo(
        "A First GPT Training Run",
        [
            ("Device", str(device)),
            ("Model", f"GPT-2 Small | {tying}"),
            ("Parameters", f"{count_parameters(model):,}"),
            ("Start", start),
            ("Training", f"{NUM_EPOCHS} epochs | batch size {BATCH_SIZE}"),
            ("Context", f"{GPT_CONFIG_SMALL_256['context_length']} tokens"),
            ("Learning rate", f"{LEARNING_RATE:g}"),
            (
                "Gradient clipping",
                "off" if GRAD_CLIP_NORM is None else f"max norm {GRAD_CLIP_NORM:g}",
            ),
            (
                "Split per book",
                f"{TRAIN_RATIO:.0%} training / {1 - TRAIN_RATIO:.0%} validation",
            ),
            ("Checkpoints", str(checkpoint_dir)),
        ],
    )


#
# Main
#
# main only runs the steps, in order; each step is a function above.
def main(argv: list[str] | None = None) -> int:  # pylint: disable=too-many-locals
    """Train the model on the books, print the progress, then generate text.

    A new run refuses to start if the checkpoint folder already holds
    checkpoints, so that it never overwrites another run: resume it with
    --resume, or choose another --checkpoint-dir.

    Args:
        argv: The command-line arguments; sys.argv if None.

    Returns:
        The exit code: 0 on success, 1 if the books or the checkpoint to
        resume from are missing or do not match.
    """
    args = parse_args(argv)
    text_paths = find_book_paths()
    if not text_paths:
        print(
            f"No .txt books found in {BOOKS_DIR}. "
            "Run python -m chap4.prepare_books before training.",
            file=sys.stderr,
        )
        return 1
    if args.resume is None and list_checkpoints(args.checkpoint_dir):
        print(
            f"{args.checkpoint_dir} already holds checkpoints of another run. "
            "Resume it with --resume, or choose another --checkpoint-dir.",
            file=sys.stderr,
        )
        return 1

    torch.manual_seed(123)  # Same initial weights and batch order at every run
    device = get_device()

    # The model is created before the summary, to show its parameter count
    model = GPTModel(GPT_CONFIG_SMALL_256, tie_embeddings=TIE_EMBEDDINGS)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    settings = training_settings(text_paths)

    # Resume: restore weights, optimizer, generators, and progress
    start_epoch, state, start = 0, TrainingState(), "new run | random weights"
    if args.resume is not None:
        try:
            start_epoch, state, start = resume_training(
                args, model=model, optimizer=optimizer, device=device, settings=settings
            )
        except (FileNotFoundError, ValueError) as error:
            print(f"Cannot resume: {error}", file=sys.stderr)
            return 1

    progress = TrainingProgress()
    show_summary(progress, device, model, start, args.checkpoint_dir)

    tokenizer = tiktoken.get_encoding("gpt2")
    train_loader, val_loader, val_sample_loader = create_loaders(
        tokenizer, text_paths, progress
    )

    if start_epoch >= NUM_EPOCHS:
        print(
            f"All {NUM_EPOCHS} epochs are already done: no training. "
            "Raise NUM_EPOCHS to train more."
        )
    else:
        train_model_simple(
            model,
            train_loader,
            val_sample_loader,
            optimizer,
            device,
            num_epochs=NUM_EPOCHS,
            eval_freq=EVAL_FREQ,
            eval_iter=EVAL_ITER,
            grad_clip_norm=GRAD_CLIP_NORM,
            start_epoch=start_epoch,
            state=state,
            on_epoch_end=make_checkpoint_saver(
                args.checkpoint_dir,
                model=model,
                optimizer=optimizer,
                device=device,
                settings=settings,
                keep=KEEP_CHECKPOINTS,
            ),
        )

    print("")
    generate_from_validation_sample(
        model,
        val_loader,
        tokenizer,
        device,
        prompt_tokens=PROMPT_TOKENS,
        new_tokens=NEW_TOKENS,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
