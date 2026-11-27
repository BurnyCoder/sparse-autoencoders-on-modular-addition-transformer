"""Steering experiments (preregistration §5): H3 antipodal swap, H4 ring transport, H5 signed steering,
and the mlp_post sanity checks with their baselines.

Local: all logits come from the exact analytic paths in interventions.py (unit-tested
against the hooked TransformerLens forward pass). resid_mid edits replace the full residual
stream (TransformerLens-3 semantics); the MLP-input-only variant is reported alongside.

Global context: calibration (ring orientation and frequency) uses SAE-train inputs and their
labels only; every success rate is computed on SAE-test inputs, and confirmatory runs use the
test half of δ / s* (dev half on the dev seed only).
"""

from __future__ import annotations

import json
import math

import numpy as np
import torch

from modsae import data, interventions, mechanism, rings
from modsae.config import Paths, device
from modsae.logging_utils import get_logger
from modsae.phases import P, _done, _write, grid_cache, load_sae_for, mechanism_phase

C_GRID = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
ALPHA_GRID = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)


def _labels():
    a, b = data.index_to_ab(torch.arange(P * P), P)
    return a, b, (a + b) % P


def _logits(x: torch.Tensor, w: dict, x_orig: torch.Tensor, variant: str) -> torch.Tensor:
    return interventions.logits_from_resid_mid(x, w, skip=None if variant == "full" else x_orig)


# ---------------------------------------------------------------- ring calibration

def calibrate_angles(theta: torch.Tensor, s: torch.Tensor) -> dict:
    """Find frequency k, sign and family so that an angle tracks the sum s (SAE-train inputs only).

    family 'full': θ ≈ sgn·w_k s + c; family 'half' (A+B / Pizza): 2θ ≈ sgn·w_k s + c (θ known mod π).
    Score = circular resultant length R. Returns the shift rule Δ(δ) = slope·δ.
    """
    ok = ~torch.isnan(theta)
    th, ss = theta[ok].double(), s[ok].double()
    best = {"R": -1.0}
    for k in range(1, (P - 1) // 2 + 1):
        w = 2 * math.pi * k / P
        for sgn in (1, -1):
            for family, mult in (("full", 1.0), ("half", 2.0)):
                R = abs(torch.exp(1j * (mult * th - sgn * w * ss)).mean()).item()
                if R > best["R"]:
                    best = {"R": R, "k": k, "sign": sgn, "family": family, "slope": sgn * w / mult}
    return best


def input_ring_angle(z: torch.Tensor, labels: torch.Tensor, angle: torch.Tensor, ring: int) -> torch.Tensor:
    """Activation-weighted circular mean of active ring members' angles (nan if none active)."""
    m = (labels == ring) & ~torch.isnan(angle)
    zz = z[:, m]
    vec = (zz.double() * torch.exp(1j * angle[m].double())[None, :]).sum(1)
    out = torch.angle(vec).float()
    out[zz.sum(1) <= 0] = float("nan")
    return out


# ---------------------------------------------------------------- resid_mid experiments

@torch.no_grad()
def resid_mid_phase(run_id: str, force: bool = False, use_dev_split: bool = True) -> dict:
    """H3, H4, H5 for a resid_mid SAE."""
    out_dir = Paths().make().results / run_id
    out = out_dir / "steering.json"
    if _done(out, force):
        return json.loads(out.read_text())
    log = get_logger(f"steer_{run_id}")
    dev = device()
    cfg = json.loads((out_dir / "metrics.json").read_text())["config"]
    assert cfg["site"] == "resid_mid", "resid_mid_phase expects a resid_mid SAE"
    grid = grid_cache(cfg["base_run"], dev)
    w, X = grid["weights"], grid["resid_mid"]
    K = mechanism_phase(cfg["base_run"])["K"]
    split = data.sae_split(P, seed=0)
    tr, te = split["train"], split["test"]
    a, b, s = _labels()
    sae = load_sae_for(run_id, dev)
    feat = torch.load(out_dir / "features.pt", weights_only=False)
    a_idx, rg = feat["alive_idx"], feat["rings"]
    d_sae = sae.W_dec.shape[0]
    labels = torch.full((d_sae,), -1, dtype=torch.long)
    angle = torch.full((d_sae,), float("nan"))
    labels[a_idx], angle[a_idx] = rg["labels"], rg["angle"]
    Z = interventions.encode(sae, X)
    res = {"run_id": run_id, "K": K, "variants": {}}
    # calibration of each recovered ring on SAE-train inputs
    calib = {}
    for ring in range(rg["n_rings"]):
        theta = input_ring_angle(Z[tr], labels, angle, ring)
        calib[ring] = calibrate_angles(theta, s[tr])
    res["ring_calibration"] = calib
    good = {r: c for r, c in calib.items() if c["R"] >= 0.5}  # rings whose angle tracks the sum
    res["n_calibrated_rings"] = len(good)
    deltas_dev, deltas_test = data.dev_test_split(torch.arange(1, P), seed=0)
    deltas = deltas_dev if use_dev_split else deltas_test
    res["delta_split"] = "dev" if use_dev_split else "test"
    xt, zt = X[te], Z[te]
    st = s[te]
    clean_acc = (grid["logits"][te].argmax(-1) == st).float().mean().item()
    for variant in ("full", "mlp_input"):
        r = {"clean_acc": clean_acc}
        for mode in ("spliced", "error"):
            recon = interventions.sae_write(sae, xt, zt, mode)
            r[f"{mode}_recon_acc"] = (_logits(recon, w, xt, variant).argmax(-1) == st).float().mean().item()
            # H3: antipodal vs quarter-turn swap
            for name, shift in (("antipodal", math.pi), ("quarter", math.pi / 2)):
                T = rings.transport_matrix(labels, angle, {rr: shift for rr in good} | {rr: 0.0 for rr in calib if rr not in good})
                x2 = interventions.sae_write(sae, xt, interventions.apply_transport(zt, T), mode)
                r[f"H3_{name}_{mode}_acc"] = (_logits(x2, w, xt, variant).argmax(-1) == st).float().mean().item()
                if name == "antipodal":
                    act0 = (xt @ w["W_in"] + w["b_in"]) > 0
                    act1 = (x2 @ w["W_in"] + w["b_in"]) > 0
                    r[f"H3_neuron_flip_fraction_{mode}"] = ((act0 ^ act1).sum(1) / (act0 | act1).sum(1).clamp_min(1)).float().mean().item()
            # H4: ring transport by δ (calibrated), 2x transport, random within-ring permutation
            succ, succ2, succ_2d, succ_perm = [], [], [], []
            perm_T = random_ring_permutation(labels, angle, good, seed=0)
            for dlt in deltas.tolist():
                target = (st + dlt) % P
                T = rings.transport_matrix(labels, angle, {rr: c["slope"] * dlt for rr, c in good.items()} | {rr: 0.0 for rr in calib if rr not in good})
                pred = _logits(interventions.sae_write(sae, xt, interventions.apply_transport(zt, T), mode), w, xt, variant).argmax(-1)
                succ.append((pred == target).float().mean().item())
                T2 = rings.transport_matrix(labels, angle, {rr: 2 * c["slope"] * dlt for rr, c in good.items()} | {rr: 0.0 for rr in calib if rr not in good})
                pred2 = _logits(interventions.sae_write(sae, xt, interventions.apply_transport(zt, T2), mode), w, xt, variant).argmax(-1)
                succ2.append((pred2 == target).float().mean().item())
                succ_2d.append((pred2 == (st + 2 * dlt) % P).float().mean().item())
                predp = _logits(interventions.sae_write(sae, xt, interventions.apply_transport(zt, perm_T), mode), w, xt, variant).argmax(-1)
                succ_perm.append((predp == target).float().mean().item())
            r[f"H4_transport_{mode}_success"] = float(np.mean(succ))
            r[f"H4_transport_{mode}_success_by_delta"] = succ
            r[f"H4_double_transport_{mode}_hits_plus_delta"] = float(np.mean(succ2))
            r[f"H4_double_transport_{mode}_hits_plus_2delta"] = float(np.mean(succ_2d))
            r[f"H4_random_permutation_{mode}_success"] = float(np.mean(succ_perm))
        r.update(transport_baselines(X, w, K, tr, te, s, deltas, variant))
        r.update(signed_steering(sae, xt, st, w, feat, a_idx, K, variant, X))
        res["variants"][variant] = r
        log.info("%s [%s]: recon acc %.3f | H3 antipodal %.3f quarter %.3f | H4 transport %.3f (perm %.3f, oracle %.3f, pca %.3f) | H5 %.3f",
                 run_id, variant, r["spliced_recon_acc"], r["H3_antipodal_spliced_acc"], r["H3_quarter_spliced_acc"],
                 r["H4_transport_spliced_success"], r["H4_random_permutation_spliced_success"], r.get("oracle_rotation_success", float("nan")),
                 r.get("pca_rotation_success", float("nan")), r.get("H5_sign_symmetry", float("nan")))
    _write(out, res)
    return res


def random_ring_permutation(labels: torch.Tensor, angle: torch.Tensor, good: dict, seed: int) -> torch.Tensor:
    """Control: permute features randomly within each calibrated ring (keeps ring membership, destroys order)."""
    g = torch.Generator().manual_seed(seed)
    n = labels.shape[0]
    T = torch.eye(n)
    for ring in good:
        m = ((labels == ring) & ~torch.isnan(angle)).nonzero().flatten()
        perm = m[torch.randperm(len(m), generator=g)]
        T[m] = 0.0
        T[m, perm] = 1.0
    return T


def transport_baselines(X, w, K, tr, te, s, deltas, variant) -> dict:
    """Oracle rotation in fitted Fourier planes, rotation in PCA planes, and the fixed-vector DiffMean shift."""
    out = {}
    mu = X.mean(0)
    xt, st = X[te], s[te]
    # oracle: fitted planes for the true key frequencies, orientation/family calibrated on SAE-train
    planes, slopes = {}, {}
    for k in K:
        plane = mechanism.fitted_plane(X, k, P)
        theta, _ = mechanism.fitted_coords(X[tr], plane)
        c = calibrate_angles(theta, s[tr])
        planes[k], slopes[k] = plane, c["slope"]
        out[f"oracle_plane_calibration_k{k}"] = c
    succ = []
    for dlt in deltas.tolist():
        x2 = interventions.rotate_in_planes(xt, planes, {k: slopes[k] * dlt for k in K}, mu)
        succ.append((_logits(x2, w, xt, variant).argmax(-1) == (st + dlt) % P).float().mean().item())
    out["oracle_rotation_success"] = float(np.mean(succ))
    # PCA planes: consecutive principal-component pairs, each calibrated on SAE-train
    _, _, Vh = torch.linalg.svd((X - mu).double(), full_matrices=False)
    pca_planes, pca_slopes = {}, {}
    for j in range(len(K)):
        plane = Vh[2 * j:2 * j + 2].float()
        theta, _ = mechanism.fitted_coords(X[tr], plane)
        c = calibrate_angles(theta, s[tr])
        pca_planes[j], pca_slopes[j] = plane, c["slope"]
    succ = []
    for dlt in deltas.tolist():
        x2 = interventions.rotate_in_planes(xt, pca_planes, {j: pca_slopes[j] * dlt for j in pca_planes}, mu)
        succ.append((_logits(x2, w, xt, variant).argmax(-1) == (st + dlt) % P).float().mean().item())
    out["pca_rotation_success"] = float(np.mean(succ))
    # fixed additive vector: mean over SAE-train of x(a+57δ, b+57δ) − x(a, b) (cannot be input-dependent)
    a, b = data.index_to_ab(torch.arange(P * P), P)
    succ = []
    for dlt in deltas.tolist():
        shifted = ((a[tr] + 57 * dlt) % P) * P + (b[tr] + 57 * dlt) % P
        v = (X[shifted] - X[tr]).mean(0)
        succ.append((_logits(xt + v, w, xt, variant).argmax(-1) == (st + dlt) % P).float().mean().item())
    out["fixed_vector_shift_success"] = float(np.mean(succ))
    return out


def signed_steering(sae, xt, st, w, feat, a_idx, K, variant, X, n_inputs: int = 256) -> dict:
    """H5: x + c·||x̄||·d_i for ring features; sign symmetry of the frequency-k phase of the logit change."""
    table, ring_mask = feat["table"], feat["ring_mask"]
    if not K or not ring_mask.any():
        return {}
    scale = X.norm(dim=-1).mean().item()
    x = xt[:n_inputs]
    base = _logits(x, w, x, variant)
    W_dec = sae.W_dec.detach().float().cpu()
    sym, dose = [], {c: [] for c in C_GRID}
    for j in ring_mask.nonzero().flatten().tolist():
        k = int(table["k_key"][j])
        d = W_dec[a_idx[j]]
        phases = {}
        for c in C_GRID:
            for sgn in (1, -1):
                dl = _logits(interventions.signed_steer(x, d, sgn * c, scale), w, x, variant) - base
                spec = torch.fft.fft(dl.double(), dim=1)[:, k]
                phases[(sgn, c)] = -torch.angle(spec)
            dose[c].append(torch.cos(phases[(1, c)] - phases[(-1, c)]).mean().item())
        sym.append(dose[C_GRID[-1]][-1])
    return {"H5_sign_symmetry": float(np.mean(sym)), "H5_sign_symmetry_by_c": {str(c): float(np.mean(v)) for c, v in dose.items()},
            "H5_n_features": len(sym), "H5_per_feature": sym}


# ---------------------------------------------------------------- mlp_post sanity checks

@torch.no_grad()
def mlp_post_phase(run_id: str, force: bool = False, use_dev_split: bool = True) -> dict:
    """Sanity checks at mlp_post (exactly linear in the logits): answer rewriting with one ring feature per
    key frequency versus min-norm oracle, DiffMean, top neurons and random directions at matched norm."""
    out_dir = Paths().make().results / run_id
    out = out_dir / "steering.json"
    if _done(out, force):
        return json.loads(out.read_text())
    log = get_logger(f"steer_{run_id}")
    dev = device()
    cfg = json.loads((out_dir / "metrics.json").read_text())["config"]
    grid = grid_cache(cfg["base_run"], dev)
    w, H, R = grid["weights"], grid["mlp_post"], grid["resid_mid"]
    K = mechanism_phase(cfg["base_run"])["K"]
    split = data.sae_split(P, seed=0)
    tr, te = split["train"], split["test"]
    _, _, s = _labels()
    sae = load_sae_for(run_id, dev)
    feat = torch.load(out_dir / "features.pt", weights_only=False)
    table, ring_mask, a_idx = feat["table"], feat["ring_mask"], feat["alive_idx"]
    W_dec = sae.W_dec.detach().float().cpu()
    Z = interventions.encode(sae, H)
    q99 = torch.quantile(Z[tr], 0.99, dim=0)
    L = (W_dec @ w["W_L"])
    L = L - L.mean(1, keepdim=True)
    targets_dev, targets_test = data.dev_test_split(torch.arange(P), seed=0)
    targets = targets_dev if use_dev_split else targets_test
    ring_feats = a_idx[ring_mask]
    k_of = {int(a_idx[j]): int(table["k_key"][j]) for j in ring_mask.nonzero().flatten().tolist()}
    rows = {m: {a: [] for a in ALPHA_GRID} for m in ("sae", "oracle", "diffmean", "neurons", "random")}
    g = torch.Generator().manual_seed(0)
    for t in targets.tolist():
        sel = te[s[te] != t]
        h0, r0 = H[sel], R[sel]
        chosen = [max((f for f in ring_feats.tolist() if k_of[f] == k), key=lambda f: L[f, t].item(), default=None) for k in K]
        chosen = [f for f in chosen if f is not None]
        if not chosen:
            continue
        ideal = sum(torch.cos(2 * math.pi * k / P * (torch.arange(P) - t)) for k in K)  # Nanda's ideal logits for answer t
        oracle_dir = interventions.min_norm_mlp_post(ideal, w["W_L"], rank=2 * len(K)).float()
        dm_dir = interventions.diffmean_direction(H[tr], s[tr], t)
        top_neurons = (w["W_L"][:, t] - w["W_L"].mean(1)).topk(len(chosen)).indices
        nr_dir = torch.zeros(H.shape[1])
        nr_dir[top_neurons] = 1.0
        rnd_dir = torch.randn(H.shape[1], generator=g)
        for alpha in ALPHA_GRID:
            dz = torch.zeros(len(sel), W_dec.shape[0])
            z0 = interventions.encode(sae, h0)
            for f in chosen:
                dz[:, f] = alpha * q99[f] - z0[:, f]  # clamp z_f to alpha * q99_f (error term preserved)
            dh = dz @ W_dec
            norm = dh.norm(dim=-1, keepdim=True)
            for name, delta in (("sae", dh), ("oracle", oracle_dir), ("diffmean", dm_dir), ("neurons", nr_dir), ("random", rnd_dir)):
                if name != "sae":
                    delta = norm * torch.nn.functional.normalize(delta, dim=0)[None, :]
                pred = interventions.logits_from_mlp_post(h0 + delta, r0, w).argmax(-1)
                rows[name][alpha].append((pred == t).float().mean().item())
    res = {"run_id": run_id, "K": K, "target_split": "dev" if use_dev_split else "test",
           "rewrite_success": {m: {str(a): float(np.mean(v)) for a, v in d.items()} for m, d in rows.items()}}
    _write(out, res)
    log.info("%s rewrite success by alpha: %s", run_id, res["rewrite_success"])
    return res
