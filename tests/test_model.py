"""Tests for modsae.model: architecture, frozen biases, constant attention, training signal."""

import torch

from modsae import data, model
from modsae.config import BaseModelConfig

CFG = BaseModelConfig(num_epochs=30, log_every=10, checkpoint_every=10)


def test_shapes_and_frozen_biases():
    m = model.build_model(CFG, "cpu")
    tokens, _ = data.make_dataset(CFG.p)
    logits = m(tokens[:5])
    assert logits.shape == (5, 3, CFG.p)
    assert m.W_in.shape[-2:] == (128, 512) and m.W_U.shape == (128, 113)
    for name, p in m.named_parameters():
        if name.endswith("bias"):
            assert not p.requires_grad and torch.all(p == 0)


def test_native_and_bridge_agree():
    m = model.build_model(CFG, "cpu")
    tokens, _ = data.make_dataset(CFG.p)
    assert torch.allclose(m(tokens[:50]), m.original_model(tokens[:50]), atol=1e-5)


def test_constant_attention_is_uniform():
    m = model.build_model(BaseModelConfig(constant_attention=True), "cpu")
    tokens, _ = data.make_dataset(113)
    _, cache = m.run_with_cache(tokens[:4], names_filter=lambda n: n.endswith("hook_pattern"))
    pattern = cache["blocks.0.attn.hook_pattern"]  # (batch, head, query, key)
    assert torch.allclose(pattern[:, :, 2, :], torch.full_like(pattern[:, :, 2, :], 1 / 3), atol=1e-6)


def test_short_training_reduces_loss(tmp_path, monkeypatch):
    monkeypatch.setenv("CHECKPOINT_DIR", str(tmp_path))
    _, hist = model.train(CFG, "cpu", use_trackio=False)
    assert hist["train_loss"][-1] < hist["train_loss"][0]
