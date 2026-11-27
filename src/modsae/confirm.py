"""Confirmatory runs and aggregation (preregistration stage 2).

Local: `run_seed` trains one confirmatory base model and its frozen-config SAEs (3 SAE seeds per
site), characterizes them and runs the steering experiments on the *test* halves of δ and s*.
`aggregate` applies the inclusion rule, computes the seed-level metric of every confirmatory
hypothesis, runs the hierarchical bootstrap (base seed → SAE seed → innermost units), applies Holm
correction and writes results/confirmatory/summary.json plus a markdown table.

Global context: nothing here may be tuned after the stage-2 commit; the frozen configuration lives in
FROZEN below and in reports/preregistration.md (stage 2).
"""

from __future__ import annotations

import json
import math

import numpy as np

from modsae import phases, stats, steering
from modsae.config import BaseModelConfig, Paths, SAEConfig
from modsae.logging_utils import get_logger

# Frozen in stage 2 (see reports/preregistration.md). Filled in when stage 2 is committed.
FROZEN = {
    "resid_mid": {"arch": "batchtopk", "d_sae": 256, "k": 8, "center": True},
    "mlp_post": {"arch": "batchtopk", "d_sae": 256, "k": 8, "center": True},
}
SAE_SEEDS = (0, 1, 2)
CONFIRM_SEEDS = tuple(range(10))
N_BOOT = 10_000

# hypothesis -> (description, threshold, direction, falsified-if rule)
HYPOTHESES = {
    "S1_a_plus_b_cone": ("resid_mid variance explained by A+B features (sum over K)", 0.80, "greater", lambda v: v < 0.5),
    "H1_ring_fraction_resid_mid": ("fraction of alive resid_mid SAE features that are ring features", 0.80, "greater", lambda v: v < 0.5),
    "H1_ring_fraction_mlp_post": ("fraction of alive mlp_post SAE features that are ring features", 0.80, "greater", lambda v: v < 0.5),
    "H3_antipodal_accuracy": ("spliced accuracy after antipodal swap of calibrated rings (resid_mid)", 0.95, "greater", lambda v: v <= 0.5),
    "H4_transport_success": ("P(output = a+b+δ) after ring transport, test δ (resid_mid)", 0.80, "greater", lambda v: v <= 0.2),
    "H5_sign_symmetry": ("mean cos(φ+ − φ−) of ring features at c = ±8 (resid_mid)", 0.50, "greater", lambda v: v <= 0.0),
    "S3_answer_writing": ("write rate of own preferred sum at c = +8 minus shuffled-sum null (resid_mid)", 0.0, "greater", lambda v: v <= 0.0),
}


def sae_cfg(base_run: str, site: str, seed: int) -> SAEConfig:
    return SAEConfig(base_run=base_run, site=site, seed=seed, **FROZEN[site])


def run_seed(seed: int) -> None:
    log = get_logger(f"confirm_s{seed}")
    base = BaseModelConfig(seed=seed)
    phases.train_base_phase(base)
    mech = phases.mechanism_phase(base.run_id)
    log.info("seed %d included=%s K=%s", seed, mech["included"], mech["K"])
    if not mech["included"]:
        return
    for site in FROZEN:
        for r in SAE_SEEDS:
            cfg = sae_cfg(base.run_id, site, r)
            phases.sae_phase(cfg)
            phases.characterize_phase(cfg.run_id)
            if site == "resid_mid":
                steering.resid_mid_phase(cfg.run_id, use_dev_split=False)
            else:
                steering.mlp_post_phase(cfg.run_id, use_dev_split=False)


def _load(run_id: str, name: str):
    p = Paths().results / run_id / name
    return json.loads(p.read_text()) if p.exists() else None


def collect(seeds=CONFIRM_SEEDS) -> dict:
    """Nested values per hypothesis: {hyp: [[innermost array per SAE seed] per included base seed]}."""
    out = {h: [] for h in HYPOTHESES}
    included, excluded = [], []
    for seed in seeds:
        run = f"base_s{seed}"
        mech = _load(run, "mechanism.json")
        if not mech:
            continue
        if not mech["included"]:
            excluded.append(seed)
            continue
        included.append(seed)
        out["S1_a_plus_b_cone"].append([np.array([sum(v["r2_a_plus_b"] for v in mech["resid_mid_regression"].values())])])
        groups = {h: [] for h in HYPOTHESES if h != "S1_a_plus_b_cone"}
        for r in SAE_SEEDS:
            for site in FROZEN:
                rid = sae_cfg(run, site, r).run_id
                ch = _load(rid, "characterization.json")
                feat_path = Paths().results / rid / "features.pt"
                if ch and feat_path.exists():
                    import torch

                    ring = torch.load(feat_path, weights_only=False)["ring_mask"].float().numpy()
                    groups[f"H1_ring_fraction_{site}"].append(ring if len(ring) else np.array([0.0]))
            st = _load(sae_cfg(run, "resid_mid", r).run_id, "steering.json")
            if st:
                v = st["variants"]["full"]
                groups["H3_antipodal_accuracy"].append(np.array([v["H3_antipodal_spliced_acc"]]))
                groups["H4_transport_success"].append(np.array(v["H4_transport_spliced_success_by_delta"]))
                if v.get("H5_per_feature"):
                    groups["H5_sign_symmetry"].append(np.array(v["H5_per_feature"]))
                cmax = str(steering.C_GRID[-1])
                diffs = [row["write_rate"][cmax] - row["write_rate_shuffled"][cmax] for row in v["single_feature_stats"] if "write_rate" in row]
                if diffs:
                    groups["S3_answer_writing"].append(np.array(diffs))
        for h, g in groups.items():
            if g:
                out[h].append(g)
    return {"values": out, "included": included, "excluded": excluded}


def aggregate(seeds=CONFIRM_SEEDS) -> dict:
    data = collect(seeds)
    res, pvals = {}, {}
    for h, (desc, thr, direction, falsified) in HYPOTHESES.items():
        groups = data["values"][h]
        if not groups:
            res[h] = {"description": desc, "status": "not testable (no data)"}
            continue
        point = float(np.mean([np.mean([g.mean() for g in seed]) for seed in groups]))
        boot = stats.hierarchical_bootstrap(groups, n_boot=N_BOOT, seed=0)
        lo, hi = stats.percentile_ci(boot)
        p = stats.one_sided_p(boot, thr, direction)
        pvals[h] = p
        per_seed = [float(np.mean([g.mean() for g in seed])) for seed in groups]
        res[h] = {"description": desc, "threshold": thr, "point": point, "ci95": [lo, hi], "p_one_sided": p,
                  "n_base_seeds": len(groups), "per_seed": per_seed, "falsified": bool(falsified(point))}
    rejected = stats.holm(pvals)
    for h in pvals:
        res[h]["confirmed_holm"] = rejected[h]
        res[h]["status"] = "confirmed" if rejected[h] else ("falsified" if res[h]["falsified"] else "inconclusive")
    summary = {"included_seeds": data["included"], "excluded_seeds": data["excluded"], "hypotheses": res, "frozen_config": FROZEN}
    out = Paths().make().results / "confirmatory"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
