<!-- Copy of the approved plan (2026-11-27), kept up to date as the project evolves. -->

# Plan: Sparse autoencoders on the grokked modular-addition transformer, plus feature steering

Author on all write-ups: Libor "Burny" Burian. Date: 2026-11-27.

## Context

The user wants an SAE trained on the 1-layer modular-addition transformer from Nanda et al. 2023 ([arXiv 2301.05217](https://arxiv.org/abs/2301.05217)), and at least one SAE feature that does something interesting when steered. Their global CLAUDE.md asks for the full research pipeline: literature review, a preregistration pushed before any results, TDD, baselines, controls, ablations, seeds, CIs, adversarial reviews, a paper, a GitHub Pages site, GitHub and Hugging Face.

This model is a good testbed because the mechanism is already known, so SAE features can be scored against ground truth instead of LLM auto-interpretation. What Nanda et al. showed:
- Embeddings are sparse in the Fourier basis, with 5 key frequencies.
- 433 of 512 neurons have >85% of variance explained by one frequency, and neurons cluster by frequency.
- cos/sin(w_k(a+b)) are linear in the MLP activations, and logits ≈ Σ_k α_k cos(w_k(a+b−c)) (95% of variance explained).

Prior work and what follows from it (literature searched 2026-11-27; each item is re-verified in the lit-review phase):
- No peer-reviewed SAE steering study on this model turned up. The nearest is Manning-Coe et al. 2026 ([2606.09940](https://arxiv.org/abs/2606.09940)): BatchTopK crosscoders, with this model only as an appendix benchmark. That gives the project room for novelty.
- Lin 2025 ([2510.00468](https://arxiv.org/abs/2510.00468)) finds that before the MLP, a and b sit in one subspace ("A+B" features cos(w_k a)+cos(w_k b)), and after it the features are cos/sin(w_k(a+b)). If so, the pre-MLP state for frequency k is z_k = e^{iθa}+e^{iθb} = 2cos(d/2)·e^{is/2}, where s = a+b and d = a−b, measured in units of w_k. That is a cone over a circle: angle ψ_k = s/2 (mod π) and radius r_k = |2cos(d/2)|.
  - Prediction: each sum maps to two antipodal angles, so features should come in antipodal pairs that have different neurons but the same effect on the answer.
  - Zhong et al. 2023 "Clock vs Pizza" ([2306.17844](https://arxiv.org/abs/2306.17844)) gives two candidate algorithms that predict different steering outcomes. That makes the steering experiments discriminating.
- SAEs tile circular manifolds with localized features (Michaud et al. 2025, [2509.02565](https://arxiv.org/abs/2509.02565); Bhalla et al. 2026, [2604.28119](https://arxiv.org/abs/2604.28119)). Engels et al. 2024 ([2405.14860](https://arxiv.org/abs/2405.14860)) found causal circular features for modular arithmetic in LLMs.
- Baselines: DiffMean is a strong steering baseline (AxBench, [2501.17148](https://arxiv.org/abs/2501.17148)), supervised dictionaries make good SAE baselines (Makelov et al. 2024, [2405.08366](https://arxiv.org/abs/2405.08366)), and end-to-end SAEs come from Braun et al. 2024 ([2405.12241](https://arxiv.org/abs/2405.12241)).
- The user's own HF checkpoint `BurnyCoder/grokking-modular-addition-transformer` uses the TransformerLens demo recipe (seed 999, data seed 598, 25k epochs). It reports key frequencies 9, 33, 36, 38, 55 and 298 of 512 single-frequency neurons, against 433 in Nanda et al. The gap gets explained in the reproduction step.

**Design decision from the review:** additive steering at the MLP output is exactly linear (logits = W_U(resid_mid + W_out h), with no LayerNorm and zero biases). Every mlp_post steering result is therefore computable from the decoder and is only a sanity check. The confirmatory steering happens at **resid_mid, the MLP input**, where interventions must pass through the ReLU, and uses interventions that change direction (sign flips, rotations, transport along a ring), not just amplitude.

## Base models

- Recipe: TransformerLens Grokking demo, which matches the user's checkpoint: p=113, 1 layer, 4 heads, d_model 128, d_head 32, d_mlp 512, ReLU, no LayerNorm, biases frozen at 0, untied embed/unembed, input [a, b, =] with '=' = 113, 30% train split (data seed 598), AdamW lr 1e-3, wd 1.0, betas (0.9, 0.98), full batch, float64 log-softmax CE at the last position.
- Train for **40,000 epochs** as in the paper (the demo uses 25k), checkpointing every 100 epochs. The 25k checkpoint lets me compare directly with the user's HF file.
- **Dev seed 999** is used for all exploration and debugging.
- **Confirmatory seeds 0–9** are trained only after the stage-2 preregistration is frozen. Inclusion rule, fixed now: test accuracy ≥ 99.9% and cleanup finished (restricted/excluded loss as in Nanda et al.). Excluded seeds are reported.
- Validation per seed:
  - Key frequencies by a fixed rule: frequencies that dominate ≥ 10 neurons at FVE ≥ 0.85. The Nanda W_E-norm rule is reported alongside.
  - Neuron FVE reimplemented from Nanda's definition (degree-2 sin/cos terms at one frequency), checked on the HF checkpoint.
  - Restricted and excluded loss.
  - The Clock vs Pizza diagnostics from Zhong et al. (gradient symmetricity, logit dependence on a−b), plus a check of how far resid_mid is A+B: the R² of resid_mid on cos/sin of a, of b, and of a+b, and the a↔b asymmetry ‖h(a,b)−h(b,a)‖/‖h‖.
- Control models (cheap):
  - A constant-attention transformer (W_Q = W_K = 0, frozen), which is Pizza-prone and checks that the diagnostics can tell the two apart.
  - The memorization-phase checkpoint (train accuracy 100%, test accuracy near chance, roughly epoch 1,500).
  - The untrained, randomly initialized model.

## Activations, splits and SAEs

- Sites, final '=' position only: **resid_mid** (`blocks.0.hook_resid_mid`, 128-d, primary for steering), **mlp_post** (`blocks.0.mlp.hook_post`, 512-d, primary for feature geometry) and resid_post (exploratory).
- Split the 12,769 inputs **by unordered pair {a,b}** into SAE-train/val/test 80/10/10, so that (a,b) and (b,a) never straddle splits. Reconstruction metrics use SAE-test. Steering selects and calibrates on SAE-train and evaluates on SAE-test. Target sums and shifts δ are also split into dev and test halves with a seed.
- Primary SAE: BatchTopK (Bussmann et al. 2024, [2412.06410](https://arxiv.org/abs/2412.06410)), trained with SAELens 6.x `SAETrainer` and an in-memory data provider, using `normalize_activations="expected_average_only_in"`. Tentative primary configs:
  - resid_mid: m=256, k=16.
  - mlp_post: m=512, k=16.
  - k ≥ 2 × the number of key frequencies, so the code is not forced into cross-frequency lookup features.
  - These configs may change on the dev seed only, then get frozen in stage 2.
- Metrics:
  - FVU and train–test FVU gap (memorization check).
  - L0 and dead fraction.
  - Spliced-in CE, accuracy, KL and logit margin.
  - **Logit-space FVU** (FVU of W_U W_out h), because about 75% of mlp_post variance is predicted to be logit-null and accuracy sits at ceiling.
  - Fraction of decoder norm in null(W_L), compared with the activation-variance fraction.

## Feature characterization (exact, from the known mechanism)

- **Mechanism coordinates** per key frequency k:
  - Analytic coordinates ψ_k and r_k from z_k = e^{iw_k a}+e^{iw_k b}.
  - Torus coordinates (θa, θb) for the Clock alternative.
  - A fitted version that allows unequal a/b weights and phases.
- Per feature i, with activation map F_i(a,b) over the 113×113 grid:
  - **Arc-model fit**: R² of F_i ≈ r_k^γ·g(ψ_k), where g is a Fourier series in ψ capped at H harmonics. Reported as a curve over H, with assigned frequency k_i = argmax R².
  - Nulls: the same statistic at non-key frequencies, and a random rotation within the frequency cluster that keeps purity but destroys locality.
  - The arc fit is compared with a 2-D (θa, θb) model with a matched parameter count, which separates Pizza-like cone geometry from Clock-like torus geometry.
- Tuning width and preferred angle ψ_i. Line decomposition of the 2-D DFT (each nonzero frequency of Z_p×Z_p lies on one of p+1 lines: a, b, sum, diff, other), used as a descriptive lens only.
- mlp_post only: direct logit effect L_i = W_U^T W_out^T d_i, with its frequency-k phase compared against ψ_i.
- **Unsupervised ring recovery** (ground truth used only for validation): spectral clustering of decoder cosines gives rings, and a 2-D PCA of each ring's decoders gives each feature's angle. Agreement with true k and ψ is reported.
- The same characterization runs on baseline dictionaries:
  - PCA, ICA, NMF (mlp_post is non-negative) and k-means/VQ.
  - A supervised mechanism dictionary of tiles ReLU(cos(ψ−ψ_j))·r.
  - A **surrogate SAE** trained on activations synthesized from the fitted idealized mechanism, which separates artifacts of the manifold geometry from structure specific to this model.

## Hypotheses

Stage 1 is preregistered now. Thresholds and the confirmatory subset get frozen in stage 2, after dev-seed exploration.

- **H1, ring geometry**: within each key frequency, SAE features are localized arcs in ψ_k with activation ∝ r_k. Prediction: median arc-fit R² ≥ 0.8 at a preregistered H. Falsified if the key-frequency R² does not beat both nulls, or if the 2-D torus model fits clearly better (which points to Clock).
- **H2, antipodal pairs (mlp_post)**: a substantial share of features have a partner at ψ+π with decoder cosine ≤ 0.1, logit-effect cosine ≥ 0.9 and complementary firing by the sign of cos(d/2). The null is random pairing within the frequency.
- **H3, antipodal invariance (resid_mid; steering)**: moving every active ring latent to its antipodal partner flips most active MLP neurons yet keeps accuracy ≥ 95%. A quarter-turn moves accuracy to near chance. Both outcomes are predicted from the weights before running (ReLU(−u) = ReLU(u) − u). Under Clock the antipodal swap would change the answer.
- **H4, half-angle transport computes a+b+δ (resid_mid; primary steering)**: moving each active latent along its recovered ring by Δ_k = πkδ/p makes the output (a+b+δ) mod p on held-out inputs and held-out δ. A full-angle rotation by 2πkδ/p is predicted to give a+b+2δ under A+B, and Clock predicts otherwise.
  - Baselines: oracle rotation of each frequency plane (upper bound), rotation within PCA planes, the real shifted input (a+57δ, b+57δ) since 2⁻¹ ≡ 57 mod 113, additive steering of target tiles (predicted to fail, because the map depends on the input) and random permutation within a ring (should give chance).
- **H5, sign-symmetric single-feature steering (resid_mid; the single-feature candidate)**: adding c·d_j over a ± grid of c pushes the frequency-k logit phase toward the feature's sum for both signs, giving a V-shaped dose–response. Linear readout predicts the anti-answer for negative c, and Clock predicts a half-period shift. The metric is the circular phase of the frequency-k logit component. A single ring gives a comb of 17 aliases; the 5-ring version is used to measure argmax.
- **Sanity checks at mlp_post, not hypotheses**:
  - Analytic versus actual steering agreement.
  - Single-feature "Fourier comb" steering, where a feature writes its preferred sum into one frequency.
  - Answer rewriting with one feature per frequency, compared on efficiency against the **minimum-norm oracle** (pseudoinverse of W_L), DiffMean / class-conditional mean E[h|s*], top neurons and norm-matched random directions.
- **Exploratory, labelled post hoc**: case-study features (the "interesting steerable feature" write-ups), the memorization-checkpoint SAE, the constant-attention model, and the share of SAE capacity spent on logit-null directions.

Intervention details fixed now:
- Every steering run reports dose–response curves over a log-spaced ± grid, and α₅₀ normalized to the minimum-norm oracle. No single α is picked.
- The SAE error term is handled two ways: spliced (reconstruction only) is primary for resid_mid transport, and error-preserved is secondary. A diagnostic rotates the error term with the oracle plane rotation.
- Side effects are reported: probability mass on the target and on aliases, and the KL from the clean output.

## Statistics and anti-p-hacking

- The **base seed is the unit of inference**: 10 confirmatory seeds, with 3 SAE seeds nested in each, and a hierarchical bootstrap over base seeds, then SAE seeds, then inputs.
- A hypothesis is confirmed if the lower end of its 95% CI clears the frozen threshold. Holm correction runs across the confirmatory tests. Negative results are reported in full.
- Exploration happens only on dev seed 999 and on dev δ and target values. Stage 2 records the git hash of the preregistration and of the frozen code before any confirmatory seed is trained.
- Every number in the README, paper and site comes from `python run.py report`, which reads `results/`.

## Ablations and sweeps (after the main results)

- SAE architecture: BatchTopK, TopK, ReLU-L1 and JumpReLU in SAELens, with an end-to-end KL SAE as an optional extra.
- m ∈ {128, 256, 512, 1024} × k ∈ {4, 8, 16, 32}, reported as a purity-versus-k transition.
- Site: resid_mid, mlp_post and resid_post.
- SAE seeds, and the control models listed above.

## Environment and repository

- Machine: RTX 5070 Laptop (8 GB, Blackwell sm_120), Windows 11. gh is logged in as BurnyCoder (repo + workflow scopes), and the hf CLI is logged in as BurnyCoder.
- `uv init --python 3.13` (the system Python 3.15 is too new for PyTorch, and sae-lens supports 3.10–3.14). Local `.venv`, with PyTorch from a CUDA 12.8+ index (check `torch.cuda.is_available()` on sm_120).
- Dependencies: `transformer-lens` (4.x; use `HookedTransformer` if still present, else `TransformerBridge.boot_native` as the current demo does), `sae-lens` 6.x (requires transformer-lens ≥3,<5), `scikit-learn`, `numpy`, `scipy`, `pandas`, `matplotlib`, `trackio`, `python-dotenv`, `huggingface_hub`, `typst` (paper) and `pytest`.
- Fallback if the SAELens custom-provider path breaks: a minimal BatchTopK module copied from bartbussmann/BatchTopK, with a citation.
- `.env` (gitignored) and `.env.example`. No secrets are committed.
- **Public** GitHub repo `BurnyCoder/sparse-autoencoders-on-modular-addition-transformer` (public is needed for free Pages). Feature branches go through PRs, which I merge myself.
- **Public** HF model repo plus one collection, holding the base checkpoints and all SAEs in SAELens format. The user's existing checkpoint is linked as related.

```
run.py                     # CLI wrapper: train-base, collect, train-sae, characterize, steer, baselines, report, site
src/modsae/
  config.py  logging_utils.py          # dataclass configs + .env; timestamped logs to logs/ and console
  data.py                              # dataset, base split (demo recipe), unordered-pair SAE split, dev/test targets and δ
  model.py                             # build/train/save/load (TransformerLens), constant-attention variant, HF checkpoint loader
  activations.py                       # cache hook activations at the '=' position
  sae_train.py                         # SAELens SAETrainer + in-memory provider; resumable sweeps (skip finished configs)
  mechanism.py                         # key freqs, neuron FVE, restricted/excluded loss, z_k/psi_k/r_k, Clock-vs-Pizza diagnostics
  features.py  rings.py                # arc fits, nulls, line decomposition, logit effects; unsupervised ring recovery
  dictionaries.py                      # PCA/ICA/NMF/k-means, supervised mechanism dictionary, surrogate data
  interventions.py                     # add / clamp / antipodal swap / ring transport / patching hooks; oracle + DiffMean + min-norm baselines
  metrics.py  stats.py                 # FVU, logit-FVU, KL, spliced metrics; hierarchical bootstrap, Holm
  plots.py  report.py  site_export.py  # all figures and tables from results/; inject tables between markers; site data export
tests/   configs/   results/   checkpoints/ (gitignored, mirrored to HF)   logs/
reports/ literature-review.md state-of-the-art-review.md preregistration.md planned-runs.md plan.md progress.md
         summary.md paper/ (Typst) -> paper.pdf  reviews/  experiments/ (experiments.md, learnings.md, one subfolder per experiment)
docs/    site/ (GitHub Pages: feature browser, steering playground running resid_mid -> MLP -> logits exactly in JS, ring-transport and antipodal demos)
```

## Order of work (follows the CLAUDE.md order)

Every milestone below ends with: `run.py report`, then updating the README, paper, site, experiments docs and `reports/progress.md`, then commits in functional units, a push, and an HF upload of any new weights.

1. **Setup**: uv project, git init, `gh repo create`, `.gitignore`, `.env.example`.
2. **Literature review**: `literature-review.md` and `state-of-the-art-review.md`, followed by a fact-check pass that opens every cited source.
3. **Stage-1 preregistration**: `preregistration.md`, `planned-runs.md` (the run matrix), `plan.md` and README. Push before any training, then fact-check.
4. **TDD implementation**, pushed before any training. Tests cover:
   - Labels, and no split leakage across orderings.
   - The line decomposition and arc fit on synthetic functions.
   - α = 0 and identity transport leave the logits unchanged.
   - Analytic mlp_post steering equals the hooked forward pass.
   - SAELens training reduces loss on toy data.
   
   Then an end-to-end smoke run of the pipeline, reading the logs and fixing problems, and one code-review pass.
5. **Dev base model (seed 999)**: grokking, key frequencies, neuron FVE, restricted/excluded loss, Clock/Pizza diagnostics, and a comparison with the HF checkpoint. Push, and upload to HF.
6. **Dev SAEs and the controls that need no training**: SAEs at resid_mid and mlp_post, characterization, rings, H1–H5 exploration, mlp_post sanity checks, and the steering baselines. Write up case-study features. This is the 14-hour first-run budget: if the SAEs do not learn (high FVU or logit-FVU, many dead latents), stop and fix first. Push.
7. **Review 1** in `reports/reviews/`. Then the **stage-2 preregistration**, which freezes thresholds, configs and the confirmatory hypotheses, records git hashes, and is pushed.
8. **Confirmatory seeds 0–9**: base models, SAEs and every confirmatory test, followed by the trained controls (random-init, memorization checkpoint, constant attention) and the dictionary baselines, including the surrogate SAE. Push after each block.
9. **Review 2**, then the ablations (architectures, m × k, sites), then the SAE-seed sensitivity runs. Push.
10. **Write-up**: the Typst paper compiled to `reports/paper.pdf`, `summary.md`, experiments and learnings docs, the website deployed through a GitHub Actions Pages workflow, the HF collection, and the final README. Then a triple fact-check and **Review 3**. Re-check the plan and the CLAUDE.md compliance.

The paper includes an abstract through appendices, an architecture diagram, an eval table covering method, baselines, controls and ablations, and steering examples, plus an AI statement at the end. trackio provides the live dashboard (`trackio show`) for base-model and SAE training.

Estimated compute: about 3 minutes per base model, seconds to minutes per SAE, under about 4 GPU-hours in total. Most of the wall-clock time goes to writing and reviews.

## Verification

- `uv run pytest` passes. `uv run python run.py all --smoke` runs every phase end-to-end on a tiny config, and the logs show no errors.
- Dev base model: test accuracy ≥ 99.9%, key frequencies and neuron FVE reported next to Nanda's and the HF checkpoint's, and the 25k checkpoint compared against the HF file.
- SAEs: SAE-test FVU, logit-FVU, spliced accuracy, KL and dead fraction in the eval table, with the train–test gap checked.
- Steering: analytic predictions at mlp_post match the hooked forward pass to floating-point tolerance, which validates the intervention code. Confirmatory results come from the hierarchical bootstrap across base seeds.
- `run.py report` regenerates every table and figure, and a diff of the README and paper against the generated tables shows no hand-typed numbers.
- The site is checked in the browser pane (feature browser and steering playground give the same logits as Python for spot-checked inputs) and deployed to GitHub Pages. The HF collection lists the base checkpoints and SAEs, and they reload with `SAE.load_from_disk` / `from_pretrained`.
