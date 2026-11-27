"""Pipeline phases called by the CLI (run.py): train-base, mechanism, train-sae, characterize, steer.

Local: each phase reads its inputs from checkpoints/ or cache/, computes, and writes JSON
(and .pt tables) to results/<run_id>/. Phases are idempotent: a finished phase (its output
file exists) is skipped unless `force=True`, which makes sweeps resumable after a crash.

Global context: this is the wiring between the modular components (data, model, mechanism,
sae_train, features, rings, interventions, metrics). Definitions refer to
reports/preregistration.md sections.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from modsae import activations, data, features, interventions, mechanism, metrics, model, rings, sae_train
from modsae.config import BaseModelConfig, Paths, SAEConfig, device
from modsae.logging_utils import get_logger

P = 113
RING_R2 = 0.8  # tentative ring-feature threshold (preregistration §6; frozen in stage 2)


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


def _done(path: Path, force: bool) -> bool:
    return path.exists() and not force


# ---------------------------------------------------------------- models

def get_model(run: str, dev: str):
    """Model registry: base_s{seed}[_constattn], base_s{seed}_mem, base_s{seed}_init, hf_burnycoder."""
    paths = Paths()
    if run == "hf_burnycoder":
        return model.load_hf_legacy_checkpoint("BurnyCoder/grokking-modular-addition-transformer", "grokking_demo.pth", dev)[0]
    if run.endswith("_init"):
        seed = int(run.split("_s")[1].split("_")[0])
        return model.build_model(BaseModelConfig(seed=seed), dev)
    if run.endswith("_mem"):
        base = run[: -len("_mem")]
        m, cfg, hist = model.load_model(paths.checkpoints / base / "model.pt", dev)
        epoch = memorization_epoch(hist)
        ck = torch.load(paths.checkpoints / base / "checkpoints.pt", map_location="cpu", weights_only=True)
        return model.load_state(m, ck["checkpoints"][ck["epochs"].index(epoch)])
    return model.load_model(paths.checkpoints / run / "model.pt", dev)[0]


def memorization_epoch(hist: dict) -> int:
    """Earliest logged epoch (multiple of 100) with train accuracy 100% and test accuracy <= 5% (§2)."""
    for e, tr, te in zip(hist["epoch"], hist["train_acc"], hist["test_acc"]):
        if e % 100 == 0 and tr >= 1.0 and te <= 0.05:
            return e
    raise ValueError("no memorization-phase checkpoint found")


def train_base_phase(cfg: BaseModelConfig, force: bool = False) -> dict:
    out = Paths().make().results / cfg.run_id / "training.json"
    if _done(out, force):
        return json.loads(out.read_text())
    log = get_logger(f"train_{cfg.run_id}")
    log.info("training %s with %s", cfg.run_id, cfg.to_dict())
    _, hist = model.train(cfg, device(), logger=log)
    _write(out, hist)
    log.info("done %s: final train acc %.4f test acc %.4f (%.0fs)", cfg.run_id, hist["train_acc"][-1], hist["test_acc"][-1], hist["train_seconds"])
    return hist


# ---------------------------------------------------------------- mechanism + activation cache

def grid_cache(run: str, dev: str) -> dict:
    """Activations/logits on the full grid for a model run (cached in cache/<run>/grid.pt)."""
    path = Paths().make().cache / run / "grid.pt"
    if path.exists():
        return torch.load(path, weights_only=True)
    m = get_model(run, dev)
    grid = activations.run_grid(m, P)
    grid["weights"] = activations.weights(m)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(grid, path)
    return grid


def mechanism_phase(run: str, force: bool = False) -> dict:
    """Preregistration §4.1 for one model: key frequencies, FVE, restricted/excluded loss, Clock/Pizza diagnostics."""
    out = Paths().make().results / run / "mechanism.json"
    if _done(out, force):
        return json.loads(out.read_text())
    log = get_logger(f"mechanism_{run}")
    dev = device()
    grid = grid_cache(run, dev)
    w = grid["weights"]
    _, labels = data.make_dataset(P)
    tr, te = data.base_split(P, 0.3, 598)
    kf = mechanism.key_frequencies(grid["mlp_post"].T, P)
    K = kf["K"]
    res = {"run": run, "K": K, "cluster_sizes": kf["cluster_sizes"], "n_single_frequency_neurons": kf["n_single_frequency"],
           "embedding_fourier_norms": mechanism.embedding_fourier_norms(w["W_E"], P).tolist()}
    res["clean"] = {"train": mechanism.loss_and_acc(grid["logits"][tr], labels[tr]), "test": mechanism.loss_and_acc(grid["logits"][te], labels[te])}
    if K:
        restricted, excluded = mechanism.restricted_and_excluded_logits(grid["logits"], K, P)
        res["restricted"] = {"train": mechanism.loss_and_acc(restricted[tr], labels[tr]), "test": mechanism.loss_and_acc(restricted[te], labels[te])}
        res["excluded"] = {"train": mechanism.loss_and_acc(excluded[tr], labels[tr]), "test": mechanism.loss_and_acc(excluded[te], labels[te])}
        res["resid_mid_regression"] = mechanism.resid_regression(grid["resid_mid"], K, P)
    res["gradient_symmetricity"] = mechanism.gradient_symmetricity(get_model(run, dev), P)
    res["distance_irrelevance"] = mechanism.distance_irrelevance(grid["logits"], P)
    res["ab_asymmetry"] = {s: mechanism.ab_asymmetry(grid[s], P) for s in ("resid_mid", "mlp_post", "resid_post")}
    res["inclusion"] = {"test_acc>=0.999": res["clean"]["test"][1] >= 0.999, "|K|>=2": len(K) >= 2,
                        "restricted_test_acc>=0.99": bool(K) and res["restricted"]["test"][1] >= 0.99}
    res["included"] = all(res["inclusion"].values())
    _write(out, res)
    log.info("%s: K=%s clusters=%s single-freq neurons=%d | test acc %.4f | restricted test acc %s | sym %.3f | dist-irrel %.3f | included=%s",
             run, K, kf["cluster_sizes"], kf["n_single_frequency"], res["clean"]["test"][1],
             res.get("restricted", {}).get("test"), res["gradient_symmetricity"], res["distance_irrelevance"], res["included"])
    return res


# ---------------------------------------------------------------- SAE training + evaluation

def spliced_logits(site: str, x_hat: torch.Tensor, idx: torch.Tensor, grid: dict) -> torch.Tensor:
    """Logits when the site activation of inputs `idx` is replaced by x_hat (exact analytic paths, unit-tested)."""
    w = grid["weights"]
    if site == "resid_mid":
        return interventions.logits_from_resid_mid(x_hat, w)
    if site == "mlp_post":
        return interventions.logits_from_mlp_post(x_hat, grid["resid_mid"][idx], w)
    if site == "resid_post":
        return x_hat @ w["W_U"] + w["b_U"]
    raise ValueError(site)


def sae_phase(cfg: SAEConfig, force: bool = False) -> dict:
    """Train one SAE on SAE-train activations and evaluate it on SAE-test (§4.3)."""
    out = Paths().make().results / cfg.run_id / "metrics.json"
    if _done(out, force):
        return json.loads(out.read_text())
    log = get_logger(f"sae_{cfg.run_id}")
    dev = device()
    grid = grid_cache(cfg.base_run, dev)
    split = data.sae_split(P, seed=0)
    acts = grid[cfg.site]
    if cfg.variant == "surrogate":
        acts = surrogate_activations(cfg.base_run, cfg.site, dev)
    sae, info = sae_train.train_sae(cfg, acts[split["train"]], acts[split["val"]], dev, logger=log)
    res = {"run_id": cfg.run_id, "config": cfg.to_dict(), "train_seconds": info["train_seconds"]}
    res.update(evaluate_sae(sae, cfg.site, grid, split, acts))
    _write(out, res)
    log.info("%s | test FVU %.4f | logit FVU %.4f | L0 %.2f | dead %.3f | spliced acc %.4f | KL %.2e",
             cfg.run_id, res["test_fvu"], res["test_logit_fvu"], res["test_l0"], res["dead_fraction"], res["spliced"]["acc"], res["spliced"]["kl"])
    return res


@torch.no_grad()
def evaluate_sae(sae, site: str, grid: dict, split: dict, acts: torch.Tensor | None = None) -> dict:
    acts = grid[site] if acts is None else acts
    _, labels = data.make_dataset(P)
    tr, te = split["train"], split["test"]
    z_tr, z_te = interventions.encode(sae, acts[tr]), interventions.encode(sae, acts[te])
    xh_tr = interventions.sae_write(sae, acts[tr], z_tr, "spliced")
    xh_te = interventions.sae_write(sae, acts[te], z_te, "spliced")
    clean = grid["logits"][te]
    spl = spliced_logits(site, xh_te, te, grid)
    zero = spliced_logits(site, torch.zeros_like(xh_te), te, grid)
    res = {"train_fvu": metrics.fvu(acts[tr], xh_tr), "test_fvu": metrics.fvu(acts[te], xh_te),
           "test_l0": (z_te > 0).float().sum(-1).mean().item(), "dead_fraction": ((z_tr > 0).sum(0) == 0).float().mean().item(),
           "test_logit_fvu": metrics.fvu(clean, spl), "clean": metrics.output_metrics(clean, labels[te]),
           "spliced": metrics.output_metrics(spl, labels[te], clean), "zero_ablation": metrics.output_metrics(zero, labels[te], clean)}
    res["train_test_fvu_gap"] = res["test_fvu"] - res["train_fvu"]
    res["loss_recovered"] = metrics.loss_recovered(res["spliced"]["ce"], res["clean"]["ce"], res["zero_ablation"]["ce"])
    # SAE-val block: the only numbers allowed for configuration selection (preregistration §3)
    va = split["val"]
    z_va = interventions.encode(sae, acts[va])
    xh_va = interventions.sae_write(sae, acts[va], z_va, "spliced")
    spl_va = spliced_logits(site, xh_va, va, grid)
    res["val"] = {"fvu": metrics.fvu(acts[va], xh_va), "logit_fvu": metrics.fvu(grid["logits"][va], spl_va),
                  "l0": (z_va > 0).float().sum(-1).mean().item(), "spliced": metrics.output_metrics(spl_va, labels[va], grid["logits"][va])}
    if site == "mlp_post":
        W_L = grid["weights"]["W_L"]
        alive = (z_tr > 0).any(0)
        res["decoder_null_fraction_mean"] = metrics.null_space_fraction(sae.W_dec.detach().float().cpu()[alive], W_L).mean().item()
        centered = acts - acts.mean(0)
        _, S, Vh = torch.linalg.svd(centered.double(), full_matrices=False)
        var_null = metrics.null_space_fraction(Vh, W_L).double()
        res["activation_variance_null_fraction"] = ((S.pow(2) * var_null).sum() / S.pow(2).sum()).item()
    return res


def surrogate_activations(base_run: str, site: str, dev: str) -> torch.Tensor:
    """Idealized-mechanism activations (§4.5): replace each site activation by its projection onto the
    frequency-k sub-mechanism, x̃ = mean + Σ_k f_k(z_k), fitted per key frequency with the arc basis
    (r_k^γ g(ψ_k), H = 8, γ = 1) by least squares over the full grid."""
    grid = grid_cache(base_run, dev)
    K = mechanism_phase(base_run)["K"]
    X = grid[site].double()
    mu = X.mean(0)
    approx = mu.expand_as(X).clone()
    for k in K:
        psi, r = mechanism.analytic_coords(P, k)
        B = features.harmonic_basis(psi, 8)[:, 1:] * r.double()[:, None]
        coef = torch.linalg.lstsq(B, X - mu).solution
        approx = approx + B @ coef
    if site == "mlp_post":
        approx = approx.clamp_min(0)  # neuron activations are non-negative
    return approx.float()


# ---------------------------------------------------------------- characterization

def load_sae_for(run_id: str, dev: str):
    return sae_train.load_sae(Paths().checkpoints / run_id, dev)


def feature_maps(sae, acts: torch.Tensor) -> torch.Tensor:
    return interventions.encode(sae, acts)  # (N, d_sae)


def characterize_phase(run_id: str, force: bool = False) -> dict:
    """Preregistration §4.4 for one SAE: per-feature table, rings, H1/H1b/H2 metrics."""
    out_dir = Paths().make().results / run_id
    out = out_dir / "characterization.json"
    if _done(out, force):
        return json.loads(out.read_text())
    log = get_logger(f"characterize_{run_id}")
    dev = device()
    cfg = json.loads((out_dir / "metrics.json").read_text())["config"]
    site, base_run = cfg["site"], cfg["base_run"]
    grid = grid_cache(base_run, dev)
    K = mechanism_phase(base_run)["K"]
    split = data.sae_split(P, seed=0)
    sae = load_sae_for(run_id, dev)
    F = feature_maps(sae, grid[site])
    alive = (F[split["train"]] > 0).any(0)
    a_idx = alive.nonzero().flatten()
    table = features.characterize(F[:, a_idx], P, split["train"], split["test"], K, H=4)
    W_dec = sae.W_dec.detach().float().cpu()
    ring_mask = (table["r2_key_best"] >= RING_R2) if K else torch.zeros(len(a_idx), dtype=torch.bool)
    res = {"run_id": run_id, "site": site, "K": K, "n_alive": int(alive.sum()), "n_ring": int(ring_mask.sum()),
           "H1_fraction_ring": ring_mask.float().mean().item(),
           "H1_median_nonkey_r2": torch.nanmedian(table["r2_nonkey_median"]).item(),
           "median_r2_best_any_k": torch.nanmedian(table["r2_best"]).item(),
           "median_r2_key_best": torch.nanmedian(table["r2_key_best"]).item() if K else None,
           "median_torus_r2_ring": torch.nanmedian(table["torus_r2"][ring_mask]).item() if ring_mask.any() else None,
           "median_gamma_ring": table["gamma"][ring_mask].median().item() if ring_mask.any() else None,
           "median_odd_fraction_ring": torch.nanmedian(table["odd_fraction"][ring_mask]).item() if ring_mask.any() else None,
           "median_width_ring": torch.nanmedian(table["width"][ring_mask]).item() if ring_mask.any() else None,
           "median_line_fractions_ring": {n: torch.nanmedian(table[f"line_{n}"][ring_mask]).item() for n in ("a", "b", "sum", "diff", "other")} if ring_mask.any() else None,
           "ring_k_counts": {int(k): int(((table["k_key"] == k) & ring_mask).sum()) for k in K} if K else {}}
    res["H1b_reference_width"] = reference_width(site, grid, split, K)
    # unsupervised rings on alive decoders
    rg = rings.recover_rings(W_dec[a_idx], torch.ones(len(a_idx), dtype=torch.bool), max_clusters=max(2 * len(K) + 4, 6))
    res["n_rings_recovered"] = rg["n_rings"]
    res["ring_validation"] = validate_rings(rg, table, ring_mask)
    if site == "mlp_post" and K:
        L = features.logit_effects(W_dec[a_idx], grid["weights"]["W_L"])
        res["H2"] = antipodal_pairs(table, ring_mask, W_dec[a_idx], L)
        enc_phase = 2 * table["preferred_angle"]
        dec_phase = features.frequency_phase(L, table["k_key"]) if K else None
        if K and ring_mask.any():
            res["readwrite_phase_agreement_median_cos"] = torch.cos(enc_phase[ring_mask] - dec_phase[ring_mask]).median().item()
    torch.save({"alive_idx": a_idx, "table": table, "ring_mask": ring_mask, "rings": rg}, out_dir / "features.pt")
    _write(out, res)
    log.info("%s: alive %d | ring fraction %.3f | median key R2 %s | nonkey %.3f | gamma %s | odd %s | width %s vs ref %s | rings %d | H2 %s",
             run_id, res["n_alive"], res["H1_fraction_ring"], res["median_r2_key_best"], res["H1_median_nonkey_r2"],
             res["median_gamma_ring"], res["median_odd_fraction_ring"], res["median_width_ring"], res["H1b_reference_width"],
             rg["n_rings"], res.get("H2"))
    return res


def reference_width(site: str, grid: dict, split: dict, K: list[int], n_dirs: int = 64, seed: int = 0) -> float | None:
    """H1b reference: median tuning width of half-wave features (neurons at mlp_post; ReLU of random
    in-plane directions at resid_mid / resid_post)."""
    if not K:
        return None
    if site == "mlp_post":
        F = grid["mlp_post"]
    else:
        X = grid[site]
        g = torch.Generator().manual_seed(seed)
        cols = []
        for k in K:
            plane = mechanism.fitted_plane(X, k, P)
            ang = torch.rand(n_dirs, generator=g) * 2 * math.pi
            dirs = torch.cos(ang)[:, None] * plane[0] + torch.sin(ang)[:, None] * plane[1]
            cols.append(torch.relu((X - X.mean(0)) @ dirs.T))
        F = torch.cat(cols, 1)
    tab = features.characterize(F, P, split["train"], split["test"], K, H=4)
    mask = tab["r2_key_best"] >= RING_R2
    return torch.nanmedian(tab["width"][mask]).item() if mask.any() else None


def validate_rings(rg: dict, table: dict, ring_mask: torch.Tensor) -> dict:
    """Agreement of unsupervised rings with ground truth: purity of key frequency per ring, and
    circular correlation between ring angle and preferred angle (up to rotation/reflection)."""
    from sklearn.metrics import adjusted_rand_score

    labels = rg["labels"]
    if "k_key" not in table or not ring_mask.any():
        return {}
    sel = ring_mask & (labels >= 0)
    ari = adjusted_rand_score(table["k_key"][sel].numpy(), labels[sel].numpy())
    corr = []
    for ring in labels[sel].unique():
        m = sel & (labels == ring)
        if m.sum() < 4:
            continue
        a, b = rg["angle"][m].double(), table["preferred_angle"][m].double()
        best = max(abs(torch.exp(1j * (a - sgn * b)).mean()).item() for sgn in (1, -1))  # rotation-invariant
        corr.append(best)
    return {"adjusted_rand_vs_key_frequency": ari, "median_circular_angle_agreement": float(np.median(corr)) if corr else None}


def antipodal_pairs(table: dict, ring_mask: torch.Tensor, W_dec: torch.Tensor, L: torch.Tensor, n_null: int = 100, seed: int = 0) -> dict:
    """H2: fraction of ring features whose (same-frequency) antipode partner has decoder cos <= 0.1 and
    logit-effect cos >= 0.9, versus a random same-frequency pairing null."""
    Dn = torch.nn.functional.normalize(W_dec, dim=1)
    Ln = torch.nn.functional.normalize(L, dim=1)
    idx = ring_mask.nonzero().flatten()
    g = torch.Generator().manual_seed(seed)
    hits, null_hits = [], []
    for i in idx.tolist():
        same = idx[(table["k_key"][idx] == table["k_key"][i]) & (idx != i)]
        if len(same) == 0:
            continue
        dist = rings.circular_distance(table["preferred_angle"][same], table["preferred_angle"][i] + math.pi)
        j = same[dist.argmin()]
        ok = lambda j: bool((Dn[i] @ Dn[j] <= 0.1) and (Ln[i] @ Ln[j] >= 0.9))
        hits.append(bool(dist.min() <= math.pi / 8) and ok(j))
        null_hits.append(np.mean([ok(same[torch.randint(len(same), (1,), generator=g)].item()) for _ in range(n_null)]))
    return {"fraction_with_partner": float(np.mean(hits)) if hits else None, "null_fraction": float(np.mean(null_hits)) if null_hits else None,
            "n": len(hits)}
