"""
Author: L. Saetta
Date Last Modified: 2026-10-31
License: MIT
Description: Tests of the utility that completes a prompt from a checkpoint.
"""

import pytest
import torch

from build_llm.checkpoints import checkpoint_path, save_checkpoint
from build_llm.gpt2 import GPTModel
from chap4 import generate_from_checkpoint as generator

TINY_CONFIG = {
    "vocab_size": 261,  # The 261 tokens of the tiny_encoding fixture
    "context_length": 16,
    "emb_dim": 12,
    "n_heads": 3,
    "n_layers": 1,
    "drop_rate": 0.0,
    "qkv_bias": False,
}
END_OF_TEXT_ID = 260  # The end-of-text token of the tiny encoding


@pytest.fixture(name="run_dir")
def fixture_run_dir(tmp_path, monkeypatch, tiny_encoding):
    """A folder with two checkpoints of a tiny tied model, and an offline tokenizer."""
    monkeypatch.setattr(generator.tiktoken, "get_encoding", lambda name: tiny_encoding)
    model = GPTModel(TINY_CONFIG, tie_embeddings=True)
    optimizer = torch.optim.AdamW(model.parameters())
    for epoch in (1, 2):
        save_checkpoint(
            checkpoint_path(tmp_path, epoch),
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            device=torch.device("cpu"),
            # global_step is zero-based: 10 steps per epoch end at 9, 19, ...
            training_state={"global_step": 10 * epoch - 1, "val_losses": [5.0, 4.5]},
            settings={"model_config": TINY_CONFIG, "tie_embeddings": True},
        )
    return tmp_path


def test_latest_checkpoint_completes_the_prompt(run_dir, capsys):
    """Without --checkpoint, the most recent one is used and the text printed."""
    code = generator.main(
        [
            "hello",
            "--checkpoint-dir",
            str(run_dir),
            "--max-new-tokens",
            "5",
            "--device",
            "cpu",
        ]
    )

    output = capsys.readouterr().out
    assert code == 0
    assert "epoch-002.pt | epoch 2 | 20 optimizer steps" in output
    assert "best 4.500" in output
    assert output.rstrip().splitlines()[-1].startswith("hello")


def test_a_given_checkpoint_is_used(run_dir, capsys):
    """--checkpoint chooses the file."""
    path = checkpoint_path(run_dir, 1)

    assert generator.main(["hello", "--checkpoint", str(path), "--device", "cpu"]) == 0
    assert "epoch-001.pt | epoch 1" in capsys.readouterr().out


def test_missing_checkpoints_fail_clearly(tmp_path, capsys):
    """An empty folder gives exit code 1 and a hint on how to train."""
    code = generator.main(["hello", "--checkpoint-dir", str(tmp_path / "none")])

    assert code == 1
    assert "python -m chap4.first_training" in capsys.readouterr().err


def test_completion_stops_at_end_of_text(monkeypatch, tiny_encoding):
    """Tokens after the end-of-text token are not part of the completion."""
    prompt_ids = tiny_encoding.encode("hello")
    generated = [ord("!"), END_OF_TEXT_ID, ord("x")]

    def fake_generate(*_args, **_kwargs):
        return torch.tensor([prompt_ids + generated])

    monkeypatch.setattr(generator, "generate_text_simple", fake_generate)
    model = GPTModel(TINY_CONFIG)

    text = generator.complete(model, tiny_encoding, "hello", 3, torch.device("cpu"))

    assert text == "!"


def test_incomplete_characters_are_dropped(monkeypatch, tiny_encoding):
    """Stopping in the middle of a UTF-8 character leaves no replacement mark."""
    prompt_ids = tiny_encoding.encode("hello")
    curly_quote = "“".encode("utf-8")  # 3 bytes: e2 80 9c
    generated = [ord("a"), curly_quote[0], curly_quote[1]]  # The quote is cut

    def fake_generate(*_args, **_kwargs):
        return torch.tensor([prompt_ids + generated])

    monkeypatch.setattr(generator, "generate_text_simple", fake_generate)
    model = GPTModel(TINY_CONFIG)

    text = generator.complete(model, tiny_encoding, "hello", 3, torch.device("cpu"))

    assert text == "a"


def test_max_new_tokens_must_be_positive():
    """A non-positive number of tokens is rejected by the option parser."""
    with pytest.raises(SystemExit):
        generator.parse_args(["hello", "--max-new-tokens", "0"])
