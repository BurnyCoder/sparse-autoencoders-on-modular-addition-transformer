"""Tests for modsae.sae_train: SAELens training on synthetic sparse data, save/load round trip."""

import torch

from modsae import sae_train
from modsae.config import SAEConfig


def _synthetic(n=6000, d_in=32, n_atoms=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    atoms = torch.nn.functional.normalize(torch.randn(n_atoms, d_in, generator=g), dim=1)
    codes = (torch.rand(n, n_atoms, generator=g) < 0.05).float() * torch.rand(n, n_atoms, generator=g)
    return codes @ atoms


def test_batchtopk_sae_learns_and_reloads(tmp_path):
    x = _synthetic()
    cfg = SAEConfig(site="toy", d_sae=64, k=4, total_samples=150_000, batch_size=256)
    sae, info = sae_train.train_sae(cfg, x[:5000], x[5000:], "cpu", out_dir=tmp_path, use_trackio=False)
    assert info["val_fvu"] < 0.5  # far better than the trivial mean predictor (FVU = 1)
    reloaded = sae_train.load_sae(tmp_path, "cpu")
    with torch.no_grad():
        assert torch.allclose(reloaded.encode(x[:100]), sae.encode(x[:100]))
        l0 = (sae.encode(x[5000:]) > 0).float().sum(-1).mean().item()
    assert 1 <= l0 <= 12  # inference JumpReLU threshold keeps L0 near k
    assert torch.allclose(sae.W_dec.norm(dim=1), torch.ones(64), atol=1e-4)  # unit-norm decoder rows


def test_centering_fold_is_exact_on_raw_inputs(tmp_path):
    x = _synthetic() + 5.0  # large shared offset, like non-negative MLP activations
    cfg = SAEConfig(site="toy", d_sae=64, k=4, total_samples=150_000, batch_size=256, center=True)
    sae, info = sae_train.train_sae(cfg, x[:5000], x[5000:], "cpu", out_dir=tmp_path, use_trackio=False)
    assert info["val_fvu"] < 0.5  # evaluated on raw (uncentred) inputs after folding μ into b_dec
    with torch.no_grad():
        recon = sae.decode(sae.encode(x[5000:]))
    assert (recon - x[5000:]).pow(2).mean() < x[5000:].var(0).sum() / 32 * 0.5
