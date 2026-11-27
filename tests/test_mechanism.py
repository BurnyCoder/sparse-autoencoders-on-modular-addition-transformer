"""Tests for modsae.mechanism on synthetic functions with known answers."""

import math

import torch

from modsae import data, mechanism

P = 113


def _ab():
    a, b = data.index_to_ab(torch.arange(P * P), P)
    return a.double(), b.double()


def test_fve_identifies_frequency_of_degree2_polynomials():
    a, b = _ab()
    w = 2 * math.pi * 14 / P
    neuron = torch.relu(torch.cos(w * a) + torch.cos(w * b))  # rectified, not purely degree-2
    poly = torch.cos(w * a) * torch.sin(w * b) + 0.5 * torch.cos(w * b)  # exactly degree-2 at k = 14
    fve = mechanism.fve_by_frequency(torch.stack([poly, neuron]).float(), P)
    assert fve[0, 14] > 0.999 and fve[0].argmax() == 14
    assert fve[1].argmax() == 14 and fve[1, 14] > 0.5


def test_line_fractions_are_orthogonal_and_exact():
    a, b = _ab()
    w = 2 * math.pi * 5 / P
    maps = torch.stack([torch.cos(w * a), torch.sin(w * b), torch.cos(w * (a + b)), torch.cos(w * (a - b)) ** 3]).float()
    fr = mechanism.line_fractions(maps, P)
    assert fr["a"][0] > 0.999 and fr["b"][1] > 0.999 and fr["sum"][2] > 0.999 and fr["diff"][3] > 0.999
    total = sum(fr[n] for n in ["a", "b", "sum", "diff", "other"])
    assert torch.allclose(total, torch.ones(4), atol=1e-4)


def test_key_frequencies_rule():
    a, b = _ab()
    rows = []
    for k, n in [(9, 12), (33, 15), (40, 3)]:  # 40 has too few neurons to count
        w = 2 * math.pi * k / P
        for j in range(n):
            rows.append(torch.cos(w * a + j) * torch.cos(w * b + j))
    out = mechanism.key_frequencies(torch.stack(rows).float(), P)
    assert out["K"] == [9, 33]


def test_restricted_logits_keep_ideal_algorithm():
    a, b = _ab()
    c = torch.arange(P).double()
    keys = [9, 33]
    logits = sum(torch.cos(2 * math.pi * k / P * ((a + b)[:, None] - c[None, :])) for k in keys).float()
    restricted, excluded = mechanism.restricted_and_excluded_logits(logits, keys, P)
    assert torch.allclose(restricted, logits, atol=1e-4)
    assert excluded.abs().max() < 1e-4  # the ideal logits are entirely key sum-line components
    _, labels = data.make_dataset(P)
    assert mechanism.loss_and_acc(restricted, labels)[1] == 1.0


def test_analytic_coords_identity():
    a, b = _ab()
    psi, r = mechanism.analytic_coords(P, 7)
    w = 2 * math.pi * 7 / P
    assert torch.allclose(r.double(), 2 * torch.cos(w * (a - b) / 2).abs(), atol=1e-5)
    z = r.double() * torch.exp(1j * psi.double())
    assert torch.allclose(z, torch.exp(1j * w * a) + torch.exp(1j * w * b), atol=1e-5)


def test_distance_irrelevance_extremes():
    a, b = _ab()
    _, labels = data.make_dataset(P)
    base = torch.zeros(P * P, P)
    pizza = base.clone()
    pizza[torch.arange(P * P), labels] = (torch.cos(math.pi * 3 * (a - b) / P).abs() + 0.1).float()
    clock = base.clone()
    clock[torch.arange(P * P), labels] = torch.cos(2 * math.pi * 3 * a / P).float() + torch.cos(2 * math.pi * 3 * b / P).float()
    assert mechanism.distance_irrelevance(pizza, P) < 0.05
    assert mechanism.distance_irrelevance(clock, P) > 0.8


def test_resid_regression_detects_a_plus_b_rank2():
    a, b = _ab()
    w = 2 * math.pi * 11 / P
    u, v = torch.randn(64, dtype=torch.float64), torch.randn(64, dtype=torch.float64)
    X = (torch.cos(w * a) + torch.cos(w * b))[:, None] * u + (torch.sin(w * a) + torch.sin(w * b))[:, None] * v
    out = mechanism.resid_regression(X.float(), [11], P)[11]
    assert out["r2_a_plus_b"] > 0.999 and out["sv3_over_sv1"] < 1e-3
