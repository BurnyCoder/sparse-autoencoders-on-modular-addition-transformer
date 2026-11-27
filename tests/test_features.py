"""Tests for modsae.features and modsae.rings on synthetic ring features with known answers."""

import math

import torch

from modsae import data, features, mechanism, rings

P = 113


def _split():
    s = data.sae_split(P, seed=0)
    return s["train"], s["test"]


def test_arc_fit_recovers_frequency_gamma_and_angle():
    train, test = _split()
    k, phi0 = 9, 0.7
    psi, r = mechanism.analytic_coords(P, k)
    arc = r * torch.relu(torch.cos(psi - phi0))  # Pizza-style ring feature: activation ∝ r, tuned at φ0
    out = features.characterize(arc[:, None], P, train, test, keys=[9, 33], H=6)
    assert out["k_best"][0] == 9 and out["r2_best"][0] > 0.98
    assert out["gamma"][0] == 1.0
    assert rings.circular_distance(out["preferred_angle"][:1], torch.tensor([phi0]))[0] < 0.05
    assert 0.25 < out["width"][0] < 0.4  # half-wave rectified cosine: g >= max/2 on 1/3 of the circle
    assert out["odd_fraction"][0] > 0.5
    assert out["r2_nonkey_median"][0] < 0.2


def test_sum_feature_has_even_harmonics_and_gamma0():
    train, test = _split()
    a, b = data.index_to_ab(torch.arange(P * P), P)
    w = 2 * math.pi * 33 / P
    feat = torch.relu(torch.cos(w * (a + b).double() - 1.0)).float()  # depends on a+b only (Clock-style)
    out = features.characterize(feat[:, None], P, train, test, keys=[9, 33], H=6)
    assert out["k_best"][0] == 33
    assert out["gamma"][0] == 0.0
    assert out["odd_fraction"][0] < 0.05
    assert out["line_sum"][0] > 0.99


def test_frequency_phase_convention():
    c = torch.arange(P).double()
    v = torch.cos(2 * math.pi * 5 * c / P - 1.3)[None, :].float()
    assert abs(features.frequency_phase(v, torch.tensor([5]))[0].item() - 1.3) < 1e-4


def test_rings_recovered_and_transport_moves_along_ring():
    torch.manual_seed(0)
    d = 32
    planes = torch.linalg.qr(torch.randn(d, 4)).Q.T  # two orthogonal 2-D planes
    rows, truth, ang = [], [], []
    for ring in range(2):
        for j in range(12):
            t = 2 * math.pi * j / 12
            rows.append(math.cos(t) * planes[2 * ring] + math.sin(t) * planes[2 * ring + 1])
            truth.append(ring)
            ang.append(t)
    W = torch.stack(rows)
    out = rings.recover_rings(W, torch.ones(24, dtype=torch.bool), n_clusters=2)
    labels = out["labels"]
    assert len(set(labels[:12].tolist())) == 1 and len(set(labels[12:].tolist())) == 1 and labels[0] != labels[12]
    T = rings.transport_matrix(labels, out["angle"], math.pi)  # antipodal swap
    j_to = T[0].argmax().item()
    assert j_to in range(12) and rings.circular_distance(torch.tensor(ang[j_to]), torch.tensor(ang[0] + math.pi)) < 0.1
    assert torch.allclose(T.sum(1), torch.ones(24))
