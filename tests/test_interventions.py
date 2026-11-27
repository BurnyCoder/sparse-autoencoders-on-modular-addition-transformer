"""Tests for interventions, metrics and stats: identities that must hold exactly."""

import math

import numpy as np
import torch

from modsae import activations, interventions, metrics, model, rings, stats
from modsae.config import BaseModelConfig

CFG = BaseModelConfig()


def _model_and_grid():
    m = model.build_model(CFG, "cpu")
    grid = activations.run_grid(m, CFG.p)
    return m, grid, activations.weights(m)


def test_analytic_paths_match_hooked_forward():
    m, grid, w = _model_and_grid()
    idx = torch.arange(0, 12769, 97)
    x = grid["resid_mid"][idx] + 0.3 * torch.randn(len(idx), 128)  # an arbitrary edited resid_mid
    assert torch.allclose(interventions.logits_from_resid_mid(x, w), interventions.hooked_logits(m, CFG.p, idx, "resid_mid", x), atol=1e-4)
    mlp_only = interventions.logits_from_resid_mid(x, w, skip=grid["resid_mid"][idx])
    assert torch.allclose(mlp_only, interventions.hooked_logits(m, CFG.p, idx, "mlp_input", x), atol=1e-4)
    h = torch.relu(grid["mlp_post"][idx] + torch.randn(len(idx), 512))
    assert torch.allclose(interventions.logits_from_mlp_post(h, grid["resid_mid"][idx], w),
                          interventions.hooked_logits(m, CFG.p, idx, "mlp_post", h), atol=1e-4)
    assert torch.allclose(interventions.logits_from_resid_mid(grid["resid_mid"], w), grid["logits"], atol=1e-4)


def test_mlp_post_steering_is_exactly_linear():
    _, grid, w = _model_and_grid()
    d = torch.randn(512)
    base = interventions.logits_from_mlp_post(grid["mlp_post"][:50], grid["resid_mid"][:50], w)
    steered = interventions.logits_from_mlp_post(grid["mlp_post"][:50] + 3.0 * d, grid["resid_mid"][:50], w)
    assert torch.allclose(steered - base, (3.0 * d @ w["W_L"]).expand(50, -1), atol=1e-4)


def test_identity_transport_and_zero_rotation_are_noops():
    labels = torch.tensor([0, 0, 0, 1, 1, 1, -1])
    angle = torch.tensor([0.0, 2.1, 4.2, 0.0, 2.1, 4.2, float("nan")])
    T = rings.transport_matrix(labels, angle, 0.0)
    assert torch.equal(T, torch.eye(7))
    x = torch.randn(10, 6)
    P = torch.linalg.qr(torch.randn(6, 2)).Q.T
    assert torch.allclose(interventions.rotate_in_planes(x, {1: P}, {1: 0.0}, x.mean(0)), x, atol=1e-6)
    full = interventions.rotate_in_planes(x, {1: P}, {1: 2 * math.pi}, x.mean(0))
    assert torch.allclose(full, x, atol=1e-5)


def test_min_norm_oracle_hits_target():
    W_L = torch.randn(512, 113)
    t = torch.randn(113)
    dh = interventions.min_norm_mlp_post(t, W_L)
    assert torch.allclose(dh @ W_L.double(), t.double(), atol=1e-6)
    U, S, Vh = torch.linalg.svd(W_L.double(), full_matrices=False)
    t_low = 3.0 * Vh[0] + 2.0 * Vh[1]  # a target inside the top-2 output directions
    dh2 = interventions.min_norm_mlp_post(t_low.float(), W_L, rank=2)
    assert torch.allclose(dh2 @ W_L.double(), t_low, atol=1e-6)


def test_metrics():
    x = torch.randn(100, 8)
    assert metrics.fvu(x, x) == 0.0 and abs(metrics.fvu(x, x.mean(0).expand_as(x)) - 1.0) < 1e-9
    W_L = torch.zeros(4, 3)
    W_L[0, 0] = 1.0
    frac = metrics.null_space_fraction(torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0]]), W_L)
    assert torch.allclose(frac, torch.tensor([0.0, 1.0]))
    labels = torch.tensor([0, 1])
    lg = torch.tensor([[5.0, 0, 0], [0, 5.0, 0]])
    om = metrics.output_metrics(lg, labels, clean_logits=lg)
    assert om["acc"] == 1.0 and abs(om["kl"]) < 1e-9 and om["margin"] == 5.0


def test_stats():
    rng = np.random.default_rng(0)
    groups = [[rng.normal(1.0, 1.0, 200) for _ in range(3)] for _ in range(8)]
    boot = stats.hierarchical_bootstrap(groups, n_boot=500)
    lo, hi = stats.percentile_ci(boot)
    assert lo < 1.0 < hi + 0.3
    assert stats.one_sided_p(boot, 0.0) < 0.01 and stats.one_sided_p(boot, 2.0) > 0.99
    assert stats.holm({"a": 0.001, "b": 0.04, "c": 0.5}) == {"a": True, "b": False, "c": False}
    lo, hi = stats.wilson(50, 100)
    assert lo < 0.5 < hi
