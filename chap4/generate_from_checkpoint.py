"""
Author: L. Saetta
Date Last Modified: 2026-10-31
License: MIT
Description: Complete an English prompt with a model saved by first_training.py,
    by default from the most recent checkpoint, to test what it has learned.

Run it from the repository root, with the build-llm environment active.

Complete a prompt with the most recent checkpoint, 50 new tokens:

    python -m chap4.generate_from_checkpoint "It was a dark and stormy night"

Choose the number of new tokens, a given checkpoint, or another folder:

    python -m chap4.generate_from_checkpoint "The ship" --max-new-tokens 100
    python -m chap4.generate_from_checkpoint "The ship" \\
        --checkpoint models/checkpoints/first_training/epoch-001.pt
    python -m chap4.generate_from_checkpoint "The ship" \\
        --checkpoint-dir models/checkpoints/run-2

Run it on the CPU, for example while a training is using the GPU:

    python -m chap4.generate_from_checkpoint "The ship" --device cpu

The script only reads the checkpoint. It rebuilds the model from the
configuration stored in it (dimensions and weight tying), so it does not
depend on the constants of first_training.py, and checkpoints of older runs
keep working. The completion uses greedy decoding (generate_text_simple): the
same prompt and checkpoint always give the same text. Generation stops after
--max-new-tokens tokens, or earlier if the model writes the end-of-text token
that separates the books in the training data.

Checkpoints are saved with an atomic rename, so the most recent one is always
complete, even while a training is running and writing the next one.
"""

import argparse
import sys
from pathlib import Path

import tiktoken
import torch

from build_llm.checkpoints import latest_checkpoint, read_checkpoint
from build_llm.dataloaders import END_OF_TEXT
from build_llm.device import get_device
from build_llm.gpt2 import GPTModel, generate_text_simple

# The folder where first_training.py saves its checkpoints by default
CHECKPOINT_DIR = (
    Path(__file__).resolve().parent.parent / "models" / "checkpoints" / "first_training"
)
DEFAULT_NEW_TOKENS = 50


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the command-line options.

    Args:
        argv: The arguments, without the program name; sys.argv if None.

    Returns:
        The options: prompt, max_new_tokens, checkpoint, checkpoint_dir, and
        device.
    """
    parser = argparse.ArgumentParser(
        description="Complete an English prompt with a saved training checkpoint.",
    )
    parser.add_argument("prompt", help="the English text to complete")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_NEW_TOKENS,
        help=f"maximum number of tokens to generate (default: {DEFAULT_NEW_TOKENS})",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="checkpoint file to use (default: the most recent in --checkpoint-dir)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=CHECKPOINT_DIR,
        help=(
            "folder searched for the most recent checkpoint "
            f"(default: {CHECKPOINT_DIR})"
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
        help="device for the model; auto picks MPS, then CUDA, then CPU",
    )
    args = parser.parse_args(argv)
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")
    return args


def load_model(path: Path, device: torch.device) -> tuple[GPTModel, dict]:
    """Rebuild the model saved in a checkpoint, ready for generation.

    Only the model weights are used; the optimizer state is not loaded.

    Args:
        path: The checkpoint file.
        device: The device for the model.

    Returns:
        The model in evaluation mode, on device, and the checkpoint.

    Raises:
        ValueError: If the checkpoint has an unsupported version.
    """
    checkpoint = read_checkpoint(path)
    settings = checkpoint["settings"]
    model = GPTModel(
        settings["model_config"],
        tie_embeddings=settings.get("tie_embeddings", False),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()  # Disable dropout: the same input always gives the same output
    return model, checkpoint


def complete(
    model: GPTModel,
    tokenizer: tiktoken.Encoding,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
) -> str:
    """Return the text generated after the prompt, with greedy decoding.

    Args:
        model: A model in evaluation mode, on device.
        tokenizer: The GPT-2 tokenizer.
        prompt: The text to complete.
        max_new_tokens: Maximum number of tokens to generate.
        device: The device of the model.

    Returns:
        The generated text only, without the prompt. It ends before the first
        end-of-text token, if the model generates one, and never with an
        incomplete character.
    """
    prompt_ids = tokenizer.encode(prompt)
    # A batch of one sequence: (1, num_tokens). Prompts longer than the
    # context are cropped to their last tokens by generate_text_simple.
    idx = torch.tensor([prompt_ids], device=device)
    output = generate_text_simple(
        model,
        idx,
        max_new_tokens=max_new_tokens,
        context_size=model.pos_emb.num_embeddings,
    )
    new_ids = output[0, len(prompt_ids) :].tolist()
    end_of_text = tokenizer.encode(END_OF_TEXT, allowed_special={END_OF_TEXT})[0]
    if end_of_text in new_ids:
        new_ids = new_ids[: new_ids.index(end_of_text)]
    # A BPE token can hold part of a character: a curly quote is 3 bytes in
    # UTF-8, and may be split over 2 tokens. If generation stops in the middle
    # of a character, decode() would end with the replacement character, so
    # the incomplete bytes are dropped instead.
    return tokenizer.decode_bytes(new_ids).decode("utf-8", errors="ignore")


def describe(path: Path, checkpoint: dict) -> str:
    """Summarize a checkpoint: file, epoch, steps, and validation losses."""
    state = checkpoint["training_state"]
    text = f"{path} | epoch {checkpoint['epoch']}"
    text += f" | {state['global_step'] + 1:,} optimizer steps"
    if state.get("val_losses"):
        text += f" | last validation loss {state['val_losses'][-1]:.3f}"
        text += f" | best {min(state['val_losses']):.3f}"
    return text


def main(argv: list[str] | None = None) -> int:
    """Complete the prompt with the chosen checkpoint and print the result.

    Args:
        argv: The command-line arguments; sys.argv if None.

    Returns:
        The exit code: 0 on success, 1 if there is no usable checkpoint.
    """
    args = parse_args(argv)
    path = args.checkpoint or latest_checkpoint(args.checkpoint_dir)
    if path is None or not Path(path).is_file():
        missing = (
            path if path is not None else f"any checkpoint in {args.checkpoint_dir}"
        )
        print(
            f"Cannot find {missing}. Train with python -m chap4.first_training.",
            file=sys.stderr,
        )
        return 1

    device = get_device() if args.device == "auto" else torch.device(args.device)
    try:
        model, checkpoint = load_model(Path(path), device)
    except ValueError as error:
        print(f"Cannot use the checkpoint: {error}", file=sys.stderr)
        return 1

    tokenizer = tiktoken.get_encoding("gpt2")
    completion = complete(model, tokenizer, args.prompt, args.max_new_tokens, device)

    print(f"Checkpoint: {describe(Path(path), checkpoint)}")
    print(f"Device:     {device}")
    print(
        f"Prompt:     {len(tokenizer.encode(args.prompt))} tokens; "
        f"up to {args.max_new_tokens} new tokens, greedy decoding"
    )
    print()
    print(args.prompt + completion)
    return 0


if __name__ == "__main__":
    sys.exit(main())
