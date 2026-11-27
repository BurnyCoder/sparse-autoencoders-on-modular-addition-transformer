"""The known Fourier mechanism: frequencies, neuron FVE, restricted logits, Clock/Pizza diagnostics.

Local: exact 2-D DFT computations over the full (a, b) grid, plus mechanism coordinates
(ψ_k, r_k) for the "A+B" / Pizza picture and the regression checks of preregistration §4.1.

Global context: these are the ground truth against which SAE features are scored.
Sources: Nanda et al. 2023 (https://arxiv.org/abs/2301.05217) for neuron FVE, key
frequencies, restricted/excluded loss; Zhong et al. 2023 (https://arxiv.org/abs/2306.17844)
for gradient symmetricity and distance irrelevance; Lin 2025
(https://arxiv.org/abs/2510.00468) for the A+B features cos(w a)+cos(w b).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from modsae import data


def omega(k: int | torch.Tensor, p: int):
    """Angular frequency w_k = 2πk/p."""
    return 2 * math.pi * k / p


# ---------------------------------------------------------------- 2-D Fourier analysis over the grid

def grid_energy(maps: torch.Tensor, p: int) -> torch.Tensor:
    """|DFT|² over (a, b) of maps given as (n, p*p) rows (idx = a*p + b) -> (n, p, p)."""
    grids = maps.reshape(-1, p, p).double()
    return torch.fft.fft2(grids).abs().pow(2)


def freq_set_mask(k: int, p: int) -> torch.Tensor:
    """Mask of the 8 DFT bins {(±k,0), (0,±k), (±k,±k)}: Nanda's degree-2 sin/cos terms at frequency k."""
    m = torch.zeros(p, p, dtype=torch.bool)
    for fa, fb in [(k, 0), (-k, 0), (0, k), (0, -k), (k, k), (k, -k), (-k, k), (-k, -k)]:
        m[fa % p, fb % p] = True
    return m


def fve_by_frequency(maps: torch.Tensor, p: int) -> torch.Tensor:
    """FVE_k for k = 0..(p-1)//2 (column 0 unused): fraction of non-DC variance on freq_set_mask(k)."""
    energy = grid_energy(maps, p)
    total = energy.sum((-2, -1)) - energy[:, 0, 0]
    out = torch.zeros(energy.shape[0], (p - 1) // 2 + 1, dtype=torch.float64)
    for k in range(1, (p - 1) // 2 + 1):
        out[:, k] = energy[:, freq_set_mask(k, p)].sum(-1) / total.clamp_min(1e-30)
    return out.float()


def line_masks(p: int) -> dict[str, torch.Tensor]:
    """Masks for the a, b, sum, diff lines through the origin of Z_p x Z_p (DC excluded).

    A function of a alone has DFT support on (f, 0); of b alone on (0, f); of a+b on
    (f, f); of a−b on (f, −f). For prime p, every nonzero bin lies on exactly one of the
    p+1 lines through the origin, so these energy fractions are orthogonal.
    """
    fa, fb = torch.meshgrid(torch.arange(p), torch.arange(p), indexing="ij")
    nonzero = (fa != 0) | (fb != 0)
    masks = {
        "a": nonzero & (fb == 0),
        "b": nonzero & (fa == 0),
        "sum": nonzero & (fa == fb),
        "diff": nonzero & ((fa + fb) % p == 0) & (fa != 0),
    }
    masks["other"] = nonzero & ~(masks["a"] | masks["b"] | masks["sum"] | masks["diff"])
    return masks


def line_fractions(maps: torch.Tensor, p: int) -> dict[str, torch.Tensor]:
    """Fraction of each map's non-DC energy on the a / b / sum / diff / other lines."""
    energy = grid_energy(maps, p)
    total = energy.sum((-2, -1)) - energy[:, 0, 0]
    return {name: (energy[:, m].sum(-1) / total.clamp_min(1e-30)).float() for name, m in line_masks(p).items()}


# ---------------------------------------------------------------- key frequencies

def key_frequencies(neuron_maps: torch.Tensor, p: int, fve_threshold: float = 0.85, min_neurons: int = 10) -> dict:
    """Preregistered key-frequency rule (§4.1) plus per-neuron FVE summary.

    neuron_maps: (n_neurons, p*p) MLP post-activations at '='. A neuron is single-frequency
    if its best FVE_k >= fve_threshold; K = frequencies that are best for >= min_neurons
    single-frequency neurons.
    """
    fve = fve_by_frequency(neuron_maps, p)
    best_fve, best_k = fve[:, 1:].max(-1)
    best_k = best_k + 1
    single = best_fve >= fve_threshold
    counts = torch.bincount(best_k[single], minlength=fve.shape[1])
    keys = [k for k in range(1, fve.shape[1]) if counts[k] >= min_neurons]
    return {"K": keys, "neuron_best_k": best_k, "neuron_best_fve": best_fve, "n_single_frequency": int(single.sum()),
            "cluster_sizes": {k: int(counts[k]) for k in keys}}


def embedding_fourier_norms(W_E: torch.Tensor, p: int) -> torch.Tensor:
    """Nanda et al.'s embedding rule: ||DFT_f(W_E)||² summed over cos/sin, for f = 0..(p-1)//2."""
    spec = torch.fft.fft(W_E[:p].double(), dim=0).abs().pow(2).sum(-1)  # (p,)
    half = (p - 1) // 2
    out = torch.zeros(half + 1, dtype=torch.float64)
    out[0] = spec[0]
    out[1:] = spec[1:half + 1] + spec[p - half:].flip(0)
    return out.float()


# ---------------------------------------------------------------- restricted / excluded logits

def _sum_line_mask(keys: list[int], p: int, include_dc: bool) -> torch.Tensor:
    m = torch.zeros(p, p, dtype=torch.bool)
    m[0, 0] = include_dc
    for k in keys:
        m[k % p, k % p] = True
        m[(-k) % p, (-k) % p] = True
    return m


def restricted_and_excluded_logits(logits: torch.Tensor, keys: list[int], p: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep / remove the cos(w_k(a+b)), sin(w_k(a+b)) components of each output logit over (a, b).

    logits: (p*p, p) with rows idx = a*p + b. Restricted keeps DC + key sum-line bins;
    excluded removes the key sum-line bins (DC kept), as in Nanda et al. §5.
    """
    grid = logits.T.reshape(p, p, p).double()  # (c, a, b)
    spec = torch.fft.fft2(grid)
    keep = _sum_line_mask(keys, p, include_dc=True)
    key_only = _sum_line_mask(keys, p, include_dc=False)
    restricted = torch.fft.ifft2(spec * keep).real
    excluded = grid - torch.fft.ifft2(spec * key_only).real
    to_rows = lambda g: g.reshape(p, p * p).T.float()  # back to (p*p, c)
    return to_rows(restricted), to_rows(excluded)


def loss_and_acc(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    return F.cross_entropy(logits.double(), labels).item(), (logits.argmax(-1) == labels).float().mean().item()


# ---------------------------------------------------------------- Clock vs Pizza diagnostics

def gradient_symmetricity(model, p: int, n: int = 100, seed: int = 0) -> float:
    """Zhong et al.: mean cosine similarity of dQ/dE_a and dQ/dE_b, Q = correct-answer logit.

    E_a, E_b are the token-embedding outputs at positions 0 and 1 (hook_embed).
    """
    tokens, labels = data.make_dataset(p)
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(p * p, generator=gen)[:n]
    device = next(model.parameters()).device
    store = {}

    def keep(act, hook):
        act.retain_grad()
        store["embed"] = act
        return act

    with torch.enable_grad():
        logits = model.run_with_hooks(tokens[idx].to(device), fwd_hooks=[("hook_embed", keep)])
        q = logits[torch.arange(n), -1, labels[idx].to(device)].sum()
        q.backward()
    g = store["embed"].grad
    model.zero_grad(set_to_none=True)
    return F.cosine_similarity(g[:, 0], g[:, 1], dim=-1).mean().item()


def distance_irrelevance(logits: torch.Tensor, p: int) -> float:
    """Zhong et al.'s q = mean_d std_i(L[i, i+d]) / std(L) for the correct logit L[a, b].

    Near 0: the correct logit depends on a−b (Pizza). Near 1: it does not (Clock).
    """
    _, labels = data.make_dataset(p)
    L = logits[torch.arange(p * p), labels].reshape(p, p).double()
    i = torch.arange(p)
    stds = torch.stack([L[i, (i + d) % p].std() for d in range(p)])
    return (stds.mean() / L.std()).item()


def fourier_basis(p: int, k: int) -> dict[str, torch.Tensor]:
    """Columns over the grid (rows idx = a*p + b) used by the regression checks."""
    a, b = data.index_to_ab(torch.arange(p * p), p)
    w = omega(k, p)
    a, b = a.double(), b.double()
    return {
        "ab": torch.stack([torch.cos(w * a), torch.sin(w * a), torch.cos(w * b), torch.sin(w * b)], 1),
        "a_plus_b_feats": torch.stack([torch.cos(w * a) + torch.cos(w * b), torch.sin(w * a) + torch.sin(w * b)], 1),
        "sum": torch.stack([torch.cos(w * (a + b)), torch.sin(w * (a + b))], 1),
    }


def _r2(X: torch.Tensor, B: torch.Tensor) -> float:
    coef = torch.linalg.lstsq(B, X).solution
    return 1 - ((X - B @ coef).pow(2).sum() / X.pow(2).sum()).item()


def resid_regression(X: torch.Tensor, keys: list[int], p: int) -> dict:
    """A+B check at resid_mid (§4.1): per key k, R² on [cos/sin w_k a, cos/sin w_k b], its singular values,
    R² on the A+B features, and R² on cos/sin(w_k(a+b))."""
    Xc = (X - X.mean(0)).double()
    out = {}
    for k in keys:
        basis = fourier_basis(p, k)
        coef = torch.linalg.lstsq(basis["ab"], Xc).solution  # (4, d)
        sv = torch.linalg.svdvals(coef)
        out[k] = {"r2_ab": _r2(Xc, basis["ab"]), "r2_a_plus_b": _r2(Xc, basis["a_plus_b_feats"]),
                  "r2_sum": _r2(Xc, basis["sum"]), "singular_values": sv.tolist(), "sv3_over_sv1": (sv[2] / sv[0]).item()}
    return out


def ab_asymmetry(X: torch.Tensor, p: int) -> float:
    """Mean ||x(a,b) − x(b,a)|| / ||x(a,b) − mean|| over the grid (0 = perfectly symmetric)."""
    idx = torch.arange(p * p)
    a, b = data.index_to_ab(idx, p)
    swap = b * p + a
    Xc = X - X.mean(0)
    return ((Xc - Xc[swap]).norm(dim=-1) / Xc.norm(dim=-1).clamp_min(1e-8)).mean().item()


# ---------------------------------------------------------------- mechanism coordinates

def analytic_coords(p: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """ψ_k = arg(e^{i w a} + e^{i w b}) and r_k = |e^{i w a} + e^{i w b}| = 2|cos(w(a−b)/2)| over the grid."""
    a, b = data.index_to_ab(torch.arange(p * p), p)
    w = omega(k, p)
    z = torch.polar(torch.ones(p * p, dtype=torch.float64), w * a.double()) + torch.polar(torch.ones(p * p, dtype=torch.float64), w * b.double())
    return torch.angle(z).float(), z.abs().float()


def fitted_plane(X: torch.Tensor, k: int, p: int) -> torch.Tensor:
    """Top-2 right singular vectors (2, d) of the regression of centered X on [cos/sin w_k a, cos/sin w_k b]."""
    Xc = (X - X.mean(0)).double()
    coef = torch.linalg.lstsq(fourier_basis(p, k)["ab"], Xc).solution
    return torch.linalg.svd(coef, full_matrices=False).Vh[:2].float()


def fitted_coords(X: torch.Tensor, plane: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """ψ^fit and r^fit from projecting centered X onto a fitted 2-D plane."""
    z = (X - X.mean(0)) @ plane.T
    return torch.atan2(z[:, 1], z[:, 0]), z.norm(dim=-1)
