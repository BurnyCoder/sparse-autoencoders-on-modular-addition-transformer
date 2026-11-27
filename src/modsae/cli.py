"""Command-line interface: `uv run python run.py <phase> [options]`.

Phases (each idempotent; pass --force to recompute):
  train-base    train a base model (seed, optional constant attention)
  mechanism     key frequencies, FVE, restricted/excluded loss, Clock/Pizza diagnostics for a model run
  train-sae     train + evaluate one SAE on a model run's activations
  characterize  per-feature table, rings, H1/H1b/H2 metrics for an SAE run
  steer         H3/H4/H5 (resid_mid SAE) or mlp_post sanity checks (mlp_post SAE)
  sweep         train + characterize SAEs over sites x k x m x seeds (resumable)
  dev           the whole dev-seed pipeline (base_s999 + HF cross-check + resid_mid/mlp_post SAEs)
  smoke         tiny end-to-end run in cache/smoke (checks the wiring in a few minutes)
  report        regenerate tables and figures from results/
  confirm       stage-2 confirmatory runs for --confirm-seeds (frozen config, test split)
  aggregate     hierarchical bootstrap + Holm over confirmatory results
  site          export weights, SAEs, tables and figures for the GitHub Pages explorer
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def _sae_cfg(args):
    from modsae.config import SAEConfig

    return SAEConfig(base_run=args.base_run, site=args.site, arch=args.arch, d_sae=args.m, k=args.k, seed=args.seed,
                     total_samples=args.samples, variant=args.variant, l1_coefficient=args.l1, l0_coefficient=args.l0, lr=args.lr, center=args.center)


def run_dev(epochs: int, samples: int, force: bool, sae_seeds=(0,), include_hf: bool = True) -> None:
    from modsae import phases, steering
    from modsae.config import BaseModelConfig, SAEConfig

    base = BaseModelConfig(seed=999, num_epochs=epochs)
    phases.train_base_phase(base, force=force)
    phases.mechanism_phase(base.run_id, force=force)
    if include_hf:
        phases.mechanism_phase("hf_burnycoder", force=force)
    for site, m in (("resid_mid", 256), ("mlp_post", 512)):
        for seed in sae_seeds:
            cfg = SAEConfig(base_run=base.run_id, site=site, d_sae=m, k=16, seed=seed, total_samples=samples)
            phases.sae_phase(cfg, force=force)
            phases.characterize_phase(cfg.run_id, force=force)
            if site == "resid_mid":
                steering.resid_mid_phase(cfg.run_id, force=force)
            else:
                steering.mlp_post_phase(cfg.run_id, force=force)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["train-base", "mechanism", "train-sae", "characterize", "steer", "dev", "sweep", "smoke", "report", "site", "confirm", "aggregate"])
    ap.add_argument("--sites", default="resid_mid,mlp_post")
    ap.add_argument("--ks", default="2,3,4,6,8,12,16")
    ap.add_argument("--ms", default="64,128,256,512")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--seed", type=int, default=999)
    ap.add_argument("--constattn", action="store_true")
    ap.add_argument("--epochs", type=int, default=40_000)
    ap.add_argument("--run", default="base_s999")
    ap.add_argument("--base-run", default="base_s999")
    ap.add_argument("--site", default="resid_mid")
    ap.add_argument("--arch", default="batchtopk")
    ap.add_argument("--m", type=int, default=256)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--samples", type=int, default=3_000_000)
    ap.add_argument("--variant", default="")
    ap.add_argument("--l1", type=float, default=5.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--center", action="store_true")
    ap.add_argument("--l0", type=float, default=1.0)
    ap.add_argument("--sae", default="")
    ap.add_argument("--test-split", action="store_true", help="use test δ / s* (confirmatory) instead of dev")
    ap.add_argument("--status", default="")
    ap.add_argument("--confirm-seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    if args.phase == "smoke":  # isolate smoke outputs before any module reads the paths
        root = Path(__file__).resolve().parents[2] / "cache" / "smoke"
        for key, sub in (("CHECKPOINT_DIR", "checkpoints"), ("CACHE_DIR", "cache"), ("RESULTS_DIR", "results"), ("LOGS_DIR", "logs")):
            os.environ[key] = str(root / sub)
        run_dev(epochs=16_000, samples=300_000, force=args.force, include_hf=False)
        return

    from modsae import phases, steering
    from modsae.config import BaseModelConfig

    if args.phase == "train-base":
        phases.train_base_phase(BaseModelConfig(seed=args.seed, num_epochs=args.epochs, constant_attention=args.constattn), force=args.force)
    elif args.phase == "mechanism":
        phases.mechanism_phase(args.run, force=args.force)
    elif args.phase == "train-sae":
        phases.sae_phase(_sae_cfg(args), force=args.force)
    elif args.phase == "characterize":
        phases.characterize_phase(args.sae, force=args.force)
    elif args.phase == "steer":
        site = "resid_mid" if "_resid_mid_" in args.sae else "mlp_post"
        fn = steering.resid_mid_phase if site == "resid_mid" else steering.mlp_post_phase
        fn(args.sae, force=args.force, use_dev_split=not args.test_split)
    elif args.phase == "dev":
        run_dev(epochs=args.epochs, samples=args.samples, force=args.force)
    elif args.phase == "sweep":
        from modsae.config import SAEConfig

        for site in args.sites.split(","):
            for k in map(int, args.ks.split(",")):
                for m in map(int, args.ms.split(",")):
                    for seed in map(int, args.seeds.split(",")):
                        cfg = SAEConfig(base_run=args.base_run, site=site, arch=args.arch, d_sae=m, k=k, seed=seed,
                                        total_samples=args.samples, variant=args.variant, lr=args.lr, center=args.center)
                        phases.sae_phase(cfg, force=args.force)
                        phases.characterize_phase(cfg.run_id, force=args.force)
    elif args.phase == "confirm":
        from modsae import confirm

        for seed in map(int, args.confirm_seeds.split(",")):
            confirm.run_seed(seed)
    elif args.phase == "aggregate":
        from modsae import confirm

        summary = confirm.aggregate()
        for h, r in summary["hypotheses"].items():
            print(h, r.get("status"), r.get("point"), r.get("ci95"))
    elif args.phase == "site":
        from modsae import site_export

        print(site_export.build_site(args.run, [r for r in args.sae.split(",") if r], status_html=args.status))
    elif args.phase == "report":
        from modsae import report

        report.main()


if __name__ == "__main__":
    main()
