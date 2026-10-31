"""
Author: L. Saetta
Date Last Modified: 2026-10-31
License: MIT
Description: Tests for saving, listing, restoring, and pruning checkpoints.
"""

import pytest
import torch
import torch.nn.functional as F

from build_llm.checkpoints import (
    check_settings,
    checkpoint_path,
    latest_checkpoint,
    list_checkpoints,
    load_checkpoint,
    remove_old_checkpoints,
    save_checkpoint,
)
from build_llm.gpt2 import GPTModel

TINY_CONFIG = {
    "vocab_size": 30,
    "context_length": 8,
    "emb_dim": 16,
    "n_heads": 4,
    "n_layers": 1,
    "drop_rate": 0.0,
    "qkv_bias": False,
}
CPU = torch.device("cpu")
SETTINGS = {"batch_size": 2, "model_config": TINY_CONFIG}


def trained_model_and_optimizer(tie_embeddings=False):
    """A tiny model and an AdamW optimizer after one step, so both have state."""
    model = GPTModel(TINY_CONFIG, tie_embeddings=tie_embeddings)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    token_ids = torch.randint(0, 30, (2, 6))
    loss = F.cross_entropy(model(token_ids).flatten(0, 1), token_ids.flatten())
    loss.backward()
    optimizer.step()
    return model, optimizer


def save(path, model, optimizer, epoch=1, settings=None):
    """Save a checkpoint with a small training state."""
    arguments = {
        "model": model,
        "optimizer": optimizer,
        "epoch": epoch,
        "device": CPU,
        "training_state": {"global_step": 5, "val_losses": [3.5, 3.1]},
        "settings": settings or SETTINGS,
    }
    save_checkpoint(path, **arguments)


def test_round_trip_restores_weights_optimizer_and_state(tmp_path):
    """A new model and optimizer get back exactly what was saved."""
    torch.manual_seed(0)
    model, optimizer = trained_model_and_optimizer()
    path = checkpoint_path(tmp_path, 1)
    save(path, model, optimizer)

    torch.manual_seed(1)  # Different initial weights
    new_model = GPTModel(TINY_CONFIG)
    new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=0.01)
    restored = load_checkpoint(
        path, model=new_model, optimizer=new_optimizer, device=CPU
    )

    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, new_model.state_dict()[name]), name
    saved_moments = optimizer.state_dict()["state"][0]["exp_avg"]
    loaded_moments = new_optimizer.state_dict()["state"][0]["exp_avg"]
    assert torch.equal(saved_moments, loaded_moments)
    assert restored["epoch"] == 1
    assert restored["training_state"]["val_losses"] == [3.5, 3.1]
    assert restored["settings"] == SETTINGS
    assert not list(tmp_path.glob("*.tmp"))  # The temporary file was renamed


def test_generator_state_is_restored(tmp_path):
    """After loading, random numbers continue as after saving: same batch order."""
    model, optimizer = trained_model_and_optimizer()
    path = checkpoint_path(tmp_path, 1)
    save(path, model, optimizer)
    expected = torch.rand(5)

    torch.manual_seed(42)  # A different generator state
    load_checkpoint(path, model=model, optimizer=optimizer, device=CPU)

    assert torch.equal(torch.rand(5), expected)


def test_tied_weights_stay_tied_after_loading(tmp_path):
    """A model with weight tying keeps one shared matrix after a restore."""
    model, optimizer = trained_model_and_optimizer(tie_embeddings=True)
    path = checkpoint_path(tmp_path, 1)
    save(path, model, optimizer)

    new_model = GPTModel(TINY_CONFIG, tie_embeddings=True)
    new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=0.01)
    load_checkpoint(path, model=new_model, optimizer=new_optimizer, device=CPU)

    assert new_model.out_head.weight is new_model.tok_emb.weight
    assert torch.equal(new_model.tok_emb.weight, model.tok_emb.weight)


def test_different_settings_are_rejected_before_restoring(tmp_path):
    """An incompatible checkpoint raises an error and leaves the model untouched."""
    model, optimizer = trained_model_and_optimizer()
    path = checkpoint_path(tmp_path, 1)
    save(path, model, optimizer)
    other_model = GPTModel(TINY_CONFIG)
    other_optimizer = torch.optim.AdamW(other_model.parameters(), lr=0.01)
    before = other_model.tok_emb.weight.detach().clone()

    with pytest.raises(ValueError, match="batch_size: checkpoint 2, now 4"):
        load_checkpoint(
            path,
            model=other_model,
            optimizer=other_optimizer,
            device=CPU,
            expected_settings={**SETTINGS, "batch_size": 4},
        )
    assert torch.equal(other_model.tok_emb.weight, before)


def test_check_settings_reports_missing_and_extra_keys():
    """A setting present on one side only is a difference too."""
    with pytest.raises(ValueError, match="learning_rate"):
        check_settings({"batch_size": 2}, {"batch_size": 2, "learning_rate": 0.1})
    check_settings({"batch_size": 2}, {"batch_size": 2})  # No error


def test_option_disabled_now_matches_an_older_checkpoint():
    """A new setting set to None equals its absence in an older checkpoint."""
    check_settings({"batch_size": 2}, {"batch_size": 2, "grad_clip_norm": None})
    with pytest.raises(ValueError, match="grad_clip_norm"):
        check_settings({"batch_size": 2}, {"batch_size": 2, "grad_clip_norm": 1.0})


def test_unsupported_version_is_rejected(tmp_path):
    """A checkpoint with another format version is not loaded."""
    model, optimizer = trained_model_and_optimizer()
    path = tmp_path / "old.pt"
    torch.save({"version": 0}, path)

    with pytest.raises(ValueError, match="version"):
        load_checkpoint(path, model=model, optimizer=optimizer, device=CPU)


def test_listing_sorts_by_epoch_and_ignores_other_files(tmp_path):
    """Only epoch-NNN.pt files are checkpoints, sorted by epoch number."""
    for name in ("epoch-010.pt", "epoch-002.pt", "notes.txt", "epoch-x.pt"):
        (tmp_path / name).write_bytes(b"")

    names = [path.name for path in list_checkpoints(tmp_path)]

    assert names == ["epoch-002.pt", "epoch-010.pt"]
    assert latest_checkpoint(tmp_path).name == "epoch-010.pt"
    assert latest_checkpoint(tmp_path / "missing") is None


def test_remove_old_checkpoints_keeps_the_newest(tmp_path):
    """Pruning deletes the oldest checkpoints only, and no other file."""
    for epoch in (1, 2, 3):
        checkpoint_path(tmp_path, epoch).write_bytes(b"")
    (tmp_path / "notes.txt").write_bytes(b"")

    removed = remove_old_checkpoints(tmp_path, keep=2)

    assert [path.name for path in removed] == ["epoch-001.pt"]
    remaining = sorted(path.name for path in tmp_path.iterdir())
    assert remaining == ["epoch-002.pt", "epoch-003.pt", "notes.txt"]
    with pytest.raises(ValueError):
        remove_old_checkpoints(tmp_path, keep=0)
