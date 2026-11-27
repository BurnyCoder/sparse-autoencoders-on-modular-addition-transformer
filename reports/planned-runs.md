# Planned runs (run matrix)

Author: Libor "Burny" Burian. Date: 2026-11-27. Definitions: [preregistration.md](preregistration.md).

Every run is a config in `configs/` plus a results folder `results/<run_id>/` that holds `config.json`, `metrics.json` and the raw arrays. Runs are idempotent: a finished run, detected by its `metrics.json`, is skipped on re-launch, so interrupted sweeps resume.

## Shared settings

| Item | Value |
|---|---|
| Task | (a + b) mod 113, all 12,769 pairs, input [a, b, 113] |
| Base split | 30% train / 70% test, `torch.randperm` under data seed 598 (TransformerLens demo) |
| SAE split | unordered pairs {a ≤ b} shuffled with seed 0, 80/10/10 → SAE-train / SAE-val / SAE-test |
| δ / s* split | shuffled with seed 0; first half dev, second half test |
| Activation position | last token ('=') only |
| GPU | RTX 5070 Laptop 8 GB; PyTorch 2.11 + CUDA 12.8 |

## Phase A: base models

| Run id | Model seed | Architecture | Training | Purpose |
|---|---|---|---|---|
| `base_s999` | 999 | 1L, 4 heads, d_model 128, d_mlp 512, ReLU, no LN, biases 0 | AdamW lr 1e-3, wd 1, β (0.9, 0.98), full batch, 40k epochs, checkpoint every 100 | dev model (all exploration) |
| `base_s0` … `base_s9` | 0–9 | same | same | confirmatory, trained after stage 2 |
| `base_s999_constattn` | 999 | same, W_Q = W_K = 0 frozen (uniform attention) | same | Pizza-prone control (Zhong et al. Model A) |
| `base_s999_mem` | 999 | same | checkpoint of `base_s999`: first epoch with train acc 100% and test acc ≤ 5% | memorization control |
| `base_s999_init` | 999 | same | no training | untrained control |
| `hf_burnycoder` | (999, TL3 init) | user's HF checkpoint, 25k epochs | none (loaded) | reproduction cross-check |

## Phase B: SAEs on the dev model (exploration, then frozen in stage 2)

| Run id pattern | Site | Arch | m | k | SAE seeds | Training |
|---|---|---|---|---|---|---|
| `sae_s999_residmid_btk_m256_k16_r{0,1,2}` | resid_mid | BatchTopK | 256 | 16 | 3 | SAELens SAETrainer, Adam lr 3e-4, batch 512, 3M samples (~290 epochs over SAE-train) |
| `sae_s999_mlppost_btk_m512_k16_r{0,1,2}` | mlp_post | BatchTopK | 512 | 16 | 3 | same |
| `sae_s999_residpost_btk_m256_k16_r0` | resid_post | BatchTopK | 256 | 16 | 1 | same (exploratory) |

Training budget: 3M samples was chosen so each SAE sees every training vector hundreds of times on a dataset of about 10k vectors. It is checked on the dev seed via convergence of SAE-val FVU and frozen in stage 2.

## Phase C: analyses on the dev model

| Analysis | Inputs | Output |
|---|---|---|
| Mechanism report (§4.1) | `base_s999`, `hf_burnycoder`, controls | key frequencies, neuron FVE, restricted/excluded loss, Clock/Pizza diagnostics |
| Feature characterization (§4.4) | each SAE + baseline dictionaries | per-feature table (arc R², k_i, γ̂, width, φ_i, line fractions, logit effects, rings) |
| Interventions I1–I4 | resid_mid / mlp_post SAEs + baselines | dose–response curves, success rates, side effects |

## Phase D: confirmatory (after stage 2)

| Run id pattern | Count | Notes |
|---|---|---|
| `base_s{0..9}` | 10 | inclusion rule applied; excluded seeds reported |
| `sae_s{seed}_{site}_{frozen cfg}_r{0,1,2}` | 10 × 2 sites × 3 | frozen config |
| H1–H5 tests | per included seed | test δ / s* only; hierarchical bootstrap; Holm |

## Phase E: controls that need training, and baseline dictionaries

| Run id pattern | Description |
|---|---|
| `sae_s999_{site}_init_*`, `sae_s999_{site}_mem_*`, `sae_s999_{site}_constattn_*` | frozen-config SAEs on the 3 control models (3 SAE seeds each) |
| `dict_s{seed}_{site}_{pca,ica,nmf,kmeans,supervised}` | baseline dictionaries (no training or closed form) |
| `sae_s999_{site}_surrogate_r{0,1,2}` | SAE on activations synthesized from the idealized mechanism |

## Phase F: ablations and sensitivity (exploratory)

| Sweep | Grid | Seeds |
|---|---|---|
| Architecture | BatchTopK, TopK, JumpReLU, ReLU+L1 (Standard); optional end-to-end KL SAE | dev model, 3 SAE seeds |
| Dictionary size × k | m ∈ {128, 256, 512, 1024} × k ∈ {4, 8, 16, 32} | dev model, 1 SAE seed (3 for frozen config) |
| Site | resid_mid, mlp_post, resid_post | dev model |
| Base-seed sensitivity | the confirmatory seeds themselves | 10 |

## Estimated cost

About 3 minutes per base model (40k full-batch steps), and well under a minute to a few minutes per SAE (3M samples). With roughly 13 base models and about 150 SAEs, the total is under about 4 GPU-hours.
