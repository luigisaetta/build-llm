"""
Author: L. Saetta
Date Last Modified: 2026-10-30
License: Apache-2.0
Description: A first training loop: pre-train a GPT-2 Small-sized model, from
    random weights, on a single short text file.

Origin: derived from the code of Chapter 5 of LLMs-from-scratch
(https://github.com/rasbt/LLMs-from-scratch), Copyright 2023-2026 Sebastian
Raschka, licensed under the Apache License 2.0 (see
LICENSES/Apache-2.0-LLMs-from-scratch.txt). Modified by L. Saetta (2026):
docstrings added, the data path made independent of the working directory,
and the code organized in functions with a main entry point.

Run it from the repository root:

    python -m chap4.first_training

The dataset is tiny (about 5,000 tokens) and the model large (about 163M
parameters), so the model memorizes the text: the training loss keeps going
down, while the validation loss soon stops improving. That is expected here.
"""

import sys
from pathlib import Path

import tiktoken
import torch

from build_llm.dataloaders import create_dataloader_v1
from build_llm.device import get_device
from build_llm.gpt2 import GPTModel, generate_text_simple

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

# Located from this file, so that the script works from any working directory.
TEXT_PATH = Path(__file__).resolve().parent.parent / "chap1" / "the-verdict.txt"

TRAIN_RATIO = 0.90  # Fraction of the text used for training; the rest validates
BATCH_SIZE = 2
NUM_EPOCHS = 10
LEARNING_RATE = 0.0004
WEIGHT_DECAY = 0.1
EVAL_FREQ = 5  # Evaluate every EVAL_FREQ optimizer steps
EVAL_ITER = 5  # Number of batches used for each evaluation
START_CONTEXT = "Every effort moves you"
SAMPLE_TOKENS = 50  # Tokens generated for the sample printed after each epoch


#
# Functions
#
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


def generate_and_print_sample(
    model: GPTModel,
    tokenizer: tiktoken.Encoding,
    device: torch.device,
    start_context: str,
) -> None:
    """Print a greedy continuation of start_context, to follow the training.

    The model is switched to evaluation mode, then back to training mode.

    Args:
        model: The model being trained.
        tokenizer: The GPT-2 tokenizer.
        device: The device of the model.
        start_context: The prompt to continue.
    """
    model.eval()
    context_size = model.pos_emb.num_embeddings
    encoded = torch.tensor([tokenizer.encode(start_context)], device=device)
    token_ids = generate_text_simple(
        model, encoded, max_new_tokens=SAMPLE_TOKENS, context_size=context_size
    )
    decoded_text = tokenizer.decode(token_ids[0].tolist())
    print(decoded_text.replace("\n", " "))  # One line, for a compact log
    model.train()


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
    start_context: str,
    tokenizer: tiktoken.Encoding,
) -> tuple[list[float], list[float], list[int]]:
    """Train the model, evaluating it periodically.

    Every eval_freq optimizer steps, the training and validation losses are
    computed and printed. After each epoch, a sample continuation of
    start_context is printed.

    Args:
        model: The model to train, already on device.
        train_loader: Training batches.
        val_loader: Validation batches.
        optimizer: The optimizer of the model parameters.
        device: The device of the model.
        num_epochs: Number of passes over the training data.
        eval_freq: Number of optimizer steps between two evaluations.
        eval_iter: Number of batches used for each evaluation.
        start_context: The prompt of the samples printed after each epoch.
        tokenizer: The GPT-2 tokenizer.

    Returns:
        The training losses, the validation losses, and the number of tokens
        seen at each evaluation.
    """
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()  # Enable dropout

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset the gradients of the previous step

            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Compute the gradients of the loss
            optimizer.step()  # Update the weights using the gradients

            tokens_seen += input_batch.numel()
            global_step += 1

            # Evaluation step, every eval_freq steps (step 0 included)
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                )

        # Show what the model writes at the end of each epoch
        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen


#
# Main
#
def main() -> int:
    """Train the model on the text file and print the progress.

    Returns:
        The exit code: 0 on success.
    """
    torch.manual_seed(123)  # Same initial weights and batch order at every run
    device = get_device()
    print(f"Device: {device}")

    model = GPTModel(GPT_CONFIG_SMALL_256)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    tokenizer = tiktoken.get_encoding("gpt2")

    text_data = TEXT_PATH.read_text(encoding="utf-8")
    print("Characters:", len(text_data))
    print("Tokens:", len(tokenizer.encode(text_data)))

    # Split by characters: the first 90% trains, the last 10% validates
    split_idx = int(TRAIN_RATIO * len(text_data))
    context_length = GPT_CONFIG_SMALL_256["context_length"]

    train_loader = create_dataloader_v1(
        tokenizer,
        text_data[:split_idx],
        batch_size=BATCH_SIZE,
        max_length=context_length,
        stride=context_length,
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )
    val_loader = create_dataloader_v1(
        tokenizer,
        text_data[split_idx:],
        batch_size=BATCH_SIZE,
        max_length=context_length,
        stride=context_length,
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )
    print(f"Batches: {len(train_loader)} training, {len(val_loader)} validation")

    train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs=NUM_EPOCHS,
        eval_freq=EVAL_FREQ,
        eval_iter=EVAL_ITER,
        start_context=START_CONTEXT,
        tokenizer=tokenizer,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
