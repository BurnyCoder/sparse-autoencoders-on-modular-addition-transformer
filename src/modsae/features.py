"""Per-feature characterization against the known mechanism (preregistration §4.4).

Local: for every dictionary feature (SAE latent, neuron, PCA component, ...) with an
activation map F over the 113x113 grid, fit the arc model F ≈ r_k^γ · g(ψ_k) at every
frequency k, pick the best frequency, and derive tuning width, preferred angle,
odd-harmonic fraction, line fractions and (for mlp_post) the direct logit effect.

Global context: these statistics test H1 (ring features), H1b (tiling) and feed H2
(antipodal pairs). Fits use SAE-train rows and R² is reported on SAE-test rows.
"""

from __future__ import annotations

import math

import torch

from modsae import mechanism

GAMMAS = (0.0, 0.5, 1.0, 1.5, 2.0)


def harmonic_basis(psi: torch.Tensor, H: int) -> torch.Tensor:
    """[1, cos ψ, sin ψ, ..., cos Hψ, sin Hψ] as columns (N, 2H+1), float64."""
    cols = [torch.ones_like(psi, dtype=torch.float64)]
    for h in range(1, H + 1):
        cols += [torch.cos(h * psi.double()), torch.sin(h * psi.double())]
    return torch.stack(cols, 1)


def _lstsq_r2(B: torch.Tensor, Y: torch.Tensor, train: torch.Tensor, test: torch.Tensor):
    """Least squares on train rows; returns (coef, train SSE, test R²) for all columns of Y."""
    coef = torch.linalg.lstsq(B[train], Y[train]).solution
    pred = B @ coef
    sse_tr = (Y[train] - pred[train]).pow(2).sum(0)
    yt = Y[test]
    sst = (yt - yt.mean(0)).pow(2).sum(0)
    r2 = 1 - (yt - pred[test]).pow(2).sum(0) / sst.clamp_min(1e-30)
    r2[sst <= 1e-20] = float("nan")  # feature never varies on the test rows
    return coef, sse_tr, r2


def arc_fit(F: torch.Tensor, psi: torch.Tensor, r: torch.Tensor, train: torch.Tensor, test: torch.Tensor, H: int = 4):
    """Fit F ≈ r^γ g(ψ) for every feature (columns of F: (N, n_feat)); γ chosen by train SSE.

    Returns dict with test R² (n_feat,), chosen γ (n_feat,) and g coefficients (2H+1, n_feat).
    """
    Y = F.double()
    base = harmonic_basis(psi, H)
    best = None
    for gamma in GAMMAS:
        B = base * r.double().pow(gamma)[:, None]
        coef, sse, r2 = _lstsq_r2(B, Y, train, test)
        if best is None:
            best = {"sse": sse, "r2": r2, "gamma": torch.full_like(sse, gamma), "coef": coef}
        else:
            better = sse < best["sse"]
            best["sse"] = torch.where(better, sse, best["sse"])
            best["r2"] = torch.where(better, r2, best["r2"])
            best["gamma"] = torch.where(better, torch.full_like(sse, gamma), best["gamma"])
            best["coef"] = torch.where(better[None, :], coef, best["coef"])
    return best


def torus_fit(F: torch.Tensor, k: int, p: int, train: torch.Tensor, test: torch.Tensor) -> torch.Tensor:
    """Held-out R² of Nanda's 9 degree-2 sin/cos functions of (θ_a, θ_b) at frequency k (Clock-style torus model)."""
    ab = mechanism.fourier_basis(p, k)["ab"]  # cos a, sin a, cos b, sin b
    ca, sa, cb, sb = ab.T
    B = torch.stack([torch.ones_like(ca), ca, sa, cb, sb, ca * cb, ca * sb, sa * cb, sa * sb], 1)
    return _lstsq_r2(B, F.double(), train, test)[2]


def features_basis_k(p: int, k: int, H: int) -> torch.Tensor:
    """r_k · [1, cos hψ_k, sin hψ_k]_{h<=H} over the grid (γ = 1), one block of the additive multi-frequency model."""
    psi, r = mechanism.analytic_coords(p, k)
    return harmonic_basis(psi, H) * r.double()[:, None]


def tuning_stats(coef: torch.Tensor, H: int, n_grid: int = 360) -> dict:
    """From g's Fourier coefficients: width (fraction of circle with g >= 0.5 max g), preferred angle, odd-harmonic fraction."""
    theta = torch.linspace(-math.pi, math.pi, n_grid + 1)[:-1]
    g = harmonic_basis(theta, H) @ coef  # (n_grid, n_feat)
    gmax, arg = g.max(0)
    width = (g >= 0.5 * gmax[None, :]).double().mean(0)
    width[gmax <= 0] = float("nan")
    power = coef[1:].reshape(H, 2, -1).pow(2).sum(1)  # (H, n_feat): c_h² + s_h²
    odd = power[0::2].sum(0) / power.sum(0).clamp_min(1e-30)
    return {"width": width.float(), "preferred_angle": theta[arg].float(), "odd_fraction": odd.float()}


def characterize(F: torch.Tensor, p: int, train: torch.Tensor, test: torch.Tensor, keys: list[int], H: int = 4,
                 alive: torch.Tensor | None = None) -> dict:
    """Full table for features given as activation maps F (N = p*p rows, n_feat columns).

    Returns per-feature tensors: r2_by_k (n_feat, (p-1)//2+1), k_best, r2_best, r2_key_best,
    r2_nonkey_median, gamma, width, preferred_angle, odd_fraction, torus_r2, line fractions.
    """
    n_feat = F.shape[1]
    half = (p - 1) // 2
    r2_by_k = torch.full((n_feat, half + 1), float("nan"), dtype=torch.float64)
    fits = {}
    for k in range(1, half + 1):
        psi, r = mechanism.analytic_coords(p, k)
        fit = arc_fit(F, psi, r, train, test, H)
        r2_by_k[:, k] = fit["r2"]
        if k in keys:
            fits[k] = fit
    filled = torch.nan_to_num(r2_by_k[:, 1:], nan=-1.0)
    r2_best, k_best = filled.max(1)
    k_best = k_best + 1
    key_cols = torch.tensor(keys) if keys else torch.zeros(0, dtype=torch.long)
    nonkey = [k for k in range(1, half + 1) if k not in keys]
    out = {"r2_by_k": r2_by_k.float(), "k_best": k_best, "r2_best": r2_best.float(),
           "r2_nonkey_median": torch.nanmedian(r2_by_k[:, nonkey], dim=1).values.float()}
    if keys:
        key_r2 = torch.nan_to_num(r2_by_k[:, key_cols], nan=-1.0)
        r2_key_best, which = key_r2.max(1)
        k_key = key_cols[which]
        out["r2_key_best"], out["k_key"] = r2_key_best.float(), k_key
        gamma = torch.zeros(n_feat)
        width, pref, odd, torus = (torch.full((n_feat,), float("nan")) for _ in range(4))
        for k in keys:
            sel = k_key == k
            if sel.any():
                st = tuning_stats(fits[k]["coef"][:, sel], H)
                gamma[sel] = fits[k]["gamma"][sel].float()
                width[sel], pref[sel], odd[sel] = st["width"], st["preferred_angle"], st["odd_fraction"]
                torus[sel] = torus_fit(F[:, sel], k, p, train, test).float()
        out.update(gamma=gamma, width=width, preferred_angle=pref, odd_fraction=odd, torus_r2=torus)
        # exploratory: additive multi-frequency arc model Σ_k r_k g_k(ψ_k) over all key frequencies
        # (a feature that is a *conjunction* across rings fits neither the single-k nor the additive model)
        cols = [features_basis_k(p, k, H) for k in keys]
        out["additive_multi_k_r2"] = _lstsq_r2(torch.cat(cols, 1), F.double(), train, test)[2].float()
    out.update({f"line_{n}": v for n, v in mechanism.line_fractions(F.T, p).items()})
    if alive is not None:
        out["alive"] = alive
    return out


def logit_effects(W_dec: torch.Tensor, W_L: torch.Tensor) -> torch.Tensor:
    """Direct logit effect of mlp_post decoder rows, centered over outputs: (n_feat, p)."""
    L = W_dec.double() @ W_L.double()
    return (L - L.mean(1, keepdim=True)).float()


def frequency_phase(vectors: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Phase ϕ of the k-th DFT component of each row, defined so that row(c) ∝ cos(w_k c − ϕ)."""
    spec = torch.fft.fft(vectors.double(), dim=1)
    comp = spec[torch.arange(vectors.shape[0]), k]
    return (-torch.angle(comp)).float()  # fft uses e^{-i w c}; cos(w c − ϕ) has coefficient ∝ e^{-iϕ}
