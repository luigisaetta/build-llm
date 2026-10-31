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

Weight tying: the output head shares its weights with the token embeddings,
as in the original GPT-2 and unlike the book. The model has about 124M
parameters instead of about 163M, with the same architecture; see
TIE_EMBEDDINGS below.

The dataset (over 10 million GPT-2 tokens) is still small for a model of about 124M
parameters, so after some epochs the model tends to memorize the text: the
training loss keeps going down, while the validation loss stops improving.
That is expected here.
"""

import sys
import time
from pathlib import Path

import tiktoken
import torch

from build_llm.dataloaders import create_dataloader_v1, split_documents
from build_llm.device import get_device
from build_llm.gpt2 import GPTModel, generate_text_simple
from build_llm.model_parameters import count_parameters
from build_llm.training_progress import TrainingProgress

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

TRAIN_RATIO = 0.90  # Fraction of each book used for training; the rest validates
# doubled
BATCH_SIZE = 4
# reduced
NUM_EPOCHS = 6
LEARNING_RATE = 0.0004
WEIGHT_DECAY = 0.1
EVAL_FREQ = 10  # Evaluate every EVAL_FREQ optimizer steps
EVAL_ITER = 5  # Number of batches used for each evaluation
PROMPT_TOKENS = 50  # Tokens of the validation sample used as the prompt
NEW_TOKENS = 50  # Tokens generated after the prompt


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

    The model is switched to evaluation mode, then back to training mode.

    Args:
        model: The model being trained.
        train_loader: Training batches.
        val_loader: Validation batches.
        device: The device of the model.
        eval_iter: Number of batches of each loader to use.

    Returns:
        The training loss and the validation loss.
    """
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
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
) -> tuple[list[float], list[float], list[int]]:
    """Train the model, evaluating it periodically.

    Every eval_freq optimizer steps, the training and validation losses are
    computed and shown in a progress table. The final step of each epoch is
    also evaluated. Each epoch ends with loss trends and its duration,
    evaluations included.

    Args:
        model: The model to train, already on device.
        train_loader: Training batches.
        val_loader: Validation batches.
        optimizer: The optimizer of the model parameters.
        device: The device of the model.
        num_epochs: Number of passes over the training data.
        eval_freq: Number of optimizer steps between two evaluations.
        eval_iter: Number of batches used for each evaluation.

    Returns:
        The training losses, the validation losses, and the number of tokens
        seen at each evaluation.
    """
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    progress = TrainingProgress()

    for epoch in range(num_epochs):
        progress.start_epoch(epoch + 1, num_epochs)
        model.train()  # Enable dropout
        epoch_start = time.perf_counter()

        for batch_number, (input_batch, target_batch) in enumerate(
            train_loader, start=1
        ):
            optimizer.zero_grad()  # Reset the gradients of the previous step

            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Compute the gradients of the loss
            optimizer.step()  # Update the weights using the gradients

            tokens_seen += input_batch.numel()
            global_step += 1

            # Include the final batch so every epoch reports its completed progress.
            if global_step % eval_freq == 0 or batch_number == len(train_loader):
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                progress.update(
                    batch_number,
                    len(train_loader),
                    global_step + 1,
                    train_loss,
                    val_loss,
                )

        synchronize(device)  # Count the steps still running on the GPU
        epoch_seconds = time.perf_counter() - epoch_start
        progress.finish_epoch(epoch_seconds, train_losses, val_losses)

    return train_losses, val_losses, track_tokens_seen


#
# Main
#
def main() -> int:
    """Train the model on the books, print the progress, then generate text.

    Returns:
        The exit code: 0 on success.
    """
    text_paths = find_book_paths()
    if not text_paths:
        print(
            f"No .txt books found in {BOOKS_DIR}. "
            "Run python -m chap4.prepare_books before training.",
            file=sys.stderr,
        )
        return 1
    torch.manual_seed(123)  # Same initial weights and batch order at every run
    device = get_device()

    # The model is created before the summary, to show its parameter count
    model = GPTModel(GPT_CONFIG_SMALL_256, tie_embeddings=TIE_EMBEDDINGS)
    tying = "weight tying" if TIE_EMBEDDINGS else "separate output head"

    progress = TrainingProgress()
    progress.start_demo(
        "A First GPT Training Run",
        [
            ("Device", str(device)),
            ("Model", f"GPT-2 Small | random weights | {tying}"),
            ("Parameters", f"{count_parameters(model):,}"),
            ("Training", f"{NUM_EPOCHS} epochs | batch size {BATCH_SIZE}"),
            ("Context", f"{GPT_CONFIG_SMALL_256['context_length']} tokens"),
            ("Learning rate", f"{LEARNING_RATE:g}"),
            (
                "Split per book",
                f"{TRAIN_RATIO:.0%} training / {1 - TRAIN_RATIO:.0%} validation",
            ),
        ],
    )

    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    tokenizer = tiktoken.get_encoding("gpt2")

    books = [path.read_text(encoding="utf-8") for path in text_paths]
    progress.show_books(
        [
            (path.name, len(book), len(tokenizer.encode(book)))
            for path, book in zip(text_paths, books)
        ]
    )

    # Split each book on its own: the first 90% of every book trains, the last
    # 10% validates. Splitting the concatenated text once would put only the
    # end of the last book in the validation set. Inside each set, the books
    # are separated by the GPT-2 end-of-text token.
    train_text, val_text = split_documents(books, TRAIN_RATIO)
    token_counts = [
        len(tokenizer.encode(text, allowed_special={"<|endoftext|>"}))
        for text in (train_text, val_text)
    ]

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
    progress.show_split(
        [
            ("Training", token_counts[0], len(train_loader)),
            ("Validation", token_counts[1], len(val_loader)),
        ]
    )

    train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs=NUM_EPOCHS,
        eval_freq=EVAL_FREQ,
        eval_iter=EVAL_ITER,
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
