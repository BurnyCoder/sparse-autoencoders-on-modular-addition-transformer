# Preregistration: SAE features and steering in the grokked modular-addition transformer

Author: Libor "Burny" Burian. Stage 1 written 2026-11-27, **before any model was trained or any result existed**. The git commit that adds this file is the stage-1 timestamp.

Two-stage design:
- **Stage 1 (this file)** fixes the question, data splits, models, metrics, analysis plan, inclusion rules, the dev/confirmatory separation, and candidate hypotheses with tentative thresholds.
- **Stage 2** is written after exploring the **dev base model only** (seed 999) and before any confirmatory base model (seeds 0–9) is trained. It may only (a) set final numeric thresholds, (b) fix the SAE configuration, (c) fix the intervention details listed under "frozen in stage 2", and (d) select which candidate hypotheses are confirmatory. It cannot change the metrics, splits, inclusion rules or statistical procedure below. Stage 2 is appended to this file with the git hash of the code used for confirmatory runs.

Background, citations and the reasoning behind each prediction: [literature-review.md](literature-review.md), [state-of-the-art-review.md](state-of-the-art-review.md), [plan.md](plan.md).

## 1. Question

Nanda et al. 2023 ([2301.05217](https://arxiv.org/abs/2301.05217)) reverse-engineered the grokked 1-layer modular-addition transformer. Given that known mechanism, we ask:
1. What features do SAEs learn on this model's activations, and how do they relate to the known mechanism coordinates?
2. Can SAE features be steered to produce predictable, non-trivial changes in the computed answer, including changes that pass through the MLP nonlinearity?
3. Do those steering outcomes discriminate between the published Clock and Pizza algorithms (Zhong et al. 2023, [2306.17844](https://arxiv.org/abs/2306.17844)) for this exact model?

## 2. Models

- **Base model recipe**: the TransformerLens 4 Grokking demo, extended to the paper's 40,000 epochs.
  - Architecture: p = 113; 1 layer; 4 heads; d_model 128; d_head 32; d_mlp 512; ReLU; no LayerNorm; all biases frozen at 0; untied embed/unembed; learned positional embeddings; input [a, b, 113].
  - Data: 30% train split from `torch.randperm(p*p)` under data seed 598, identical for all base seeds.
  - Optimization: AdamW, lr 1e-3, wd 1.0, betas (0.9, 0.98), full batch, float64 log-softmax CE at the last position. Checkpoint every 100 epochs.
- **Dev base model**: model seed 999. All exploration happens on this seed.
- **Confirmatory base models**: model seeds 0–9, trained only after stage 2 is committed.
- **Inclusion rule** (fixed now; applied to every base model, excluded seeds reported):
  - final test accuracy ≥ 99.9%;
  - |K| ≥ 2, where K is the key set defined in §4.1;
  - accuracy ≥ 99% on the test split for logits restricted to the key frequencies (§4.1).
- **Control models**:
  - **untrained** (seed-matched random init);
  - **memorization checkpoint**: the earliest checkpoint with train accuracy = 100% and test accuracy ≤ 5%;
  - **constant attention**: W_Q = W_K = 0, frozen, so attention is uniform. This model is Pizza-prone per Zhong et al.'s Model A.

## 3. Activations, splits and SAEs

- **Hook sites**, at the last ('=') position only:
  - `blocks.0.hook_resid_mid` (128-d; the MLP input; primary site for steering);
  - `blocks.0.mlp.hook_post` (512-d; primary site for feature geometry);
  - `blocks.0.hook_resid_post` (128-d; exploratory).
- **SAE split**: shuffle the 6,441 unordered pairs {a ≤ b} with seed 0 and split them 80/10/10 into SAE-train / SAE-val / SAE-test. Both orderings of a pair go to the same split.
  - SAE weights and all calibration/selection steps use SAE-train only.
  - Configuration choices use SAE-val only.
  - Every reported reconstruction and steering number uses SAE-test.
  - Feature activation maps used only for description (heatmaps) use the full grid.
- **Shift and target split**: δ ∈ {1, …, 112} and target sums s* ∈ {0, …, 112} are each shuffled with seed 0. The first half is **dev** (used only on the dev base model) and the second half is **test**. Confirmatory runs use test δ and test s* only.
- **SAE**: BatchTopK (Bussmann et al. 2024, [2412.06410](https://arxiv.org/abs/2412.06410)) via SAELens 6.56.4 `SAETrainer`, with `normalize_activations="expected_average_only_in"`.
  - Tentative configs: resid_mid m = 256, k = 16; mlp_post m = 512, k = 16. The final config is frozen in stage 2.
  - k is never below 2|K|.
  - SAE seeds: 3 per base model in confirmatory runs.
  - SAE runs are **never excluded**. Dead or poor SAEs count as they are.

## 4. Measurements

### 4.1 Mechanism (per base model)

- **Neuron FVE**: for neuron n and frequency k ∈ {1..56}, FVE_k(n) is the fraction of the non-DC variance of its activation map over all 12,769 inputs that lies on the 2-D DFT frequencies {(±k, 0), (0, ±k), (±k, ±k)}. This is the span of Nanda et al.'s degree-2 sin/cos polynomials at frequency k.
- **Key set K**: the frequencies k that are the argmax for at least 10 neurons with FVE ≥ 0.85. The W_E Fourier-norm rule of Nanda et al. is reported alongside.
- **Restricted / excluded logits**: keep / remove the components cos(w_k(a+b)), sin(w_k(a+b)), k ∈ K, of each output logit over the (a, b) grid. The DC component is always kept. Report loss and accuracy.
- **Clock vs Pizza diagnostics** (Zhong et al.):
  - gradient symmetricity (cosine of ∂logit/∂E_a and ∂logit/∂E_b over 100 random inputs);
  - dependence of the correct logit on a−b (distance dependence);
  - at resid_mid, the singular values of the 128×4 regression of resid_mid on [cos w_k a, sin w_k a, cos w_k b, sin w_k b]. A+B predicts rank ≈ 2 (σ3/σ1 ≈ 0); separate encodings of a and b predict rank 4;
  - the a↔b asymmetry ‖x(a,b) − x(b,a)‖ / ‖x‖ at each site.

### 4.2 Mechanism coordinates

For each k ∈ K:
- **Analytic coordinates**: z_k(a,b) = e^{i w_k a} + e^{i w_k b}, ψ_k = arg z_k, r_k = |z_k| = 2|cos(w_k(a−b)/2)|.
- **Fitted coordinates**: project centered resid_mid onto the top-2 left singular vectors of the §4.1 regression coefficients; ψ_k^fit and r_k^fit come from that 2-D projection.
- **Torus coordinates** (Clock alternative): θ_a = w_k a, θ_b = w_k b.

### 4.3 SAE quality (on SAE-test)

- FVU (= normalized MSE) and the train − test FVU gap;
- L0 and the fraction of dead latents (never active on SAE-train);
- **logit-space FVU** = FVU of W_U W_out x at mlp_post, or of the logits obtained by splicing at resid_mid;
- spliced-in CE, accuracy, mean KL(clean ‖ spliced) and mean correct-answer logit margin;
- fraction of decoder squared norm in null(W_L) at mlp_post, compared with the fraction of activation variance in null(W_L).

### 4.4 Feature characterization (per alive feature i, activation map F_i over the grid)

- **Arc-model fit**: for each k, F_i ≈ r_k^γ · g(ψ_k), where g(ψ) = c_0 + Σ_{h=1..H} (c_h cos hψ + s_h sin hψ) and γ ∈ {0, 0.5, 1, 1.5, 2} is chosen by least squares on SAE-train inputs. R² is computed on SAE-test inputs. **Primary H = 4** (9 linear coefficients); the curve over H = 1..12 is reported.
  - Assigned frequency k_i is the argmax of R² over k ∈ {1..56}.
  - A feature is a **ring feature** if k_i ∈ K and R² ≥ the stage-2 threshold.
- **Torus comparison**: held-out R² of a linear fit on the 9 Nanda degree-2 basis functions at k_i (same parameter count as H = 4).
- **Clock/Pizza feature statistics**:
  - γ̂: Pizza predicts ≈ 1, Clock sum features ≈ 0;
  - odd-harmonic fraction of g: Pizza arcs > 0, pure sum features ≈ 0;
  - sum-line fraction of the 2-D DFT energy. In Z_p×Z_p every nonzero frequency lies on exactly one of the p+1 lines through the origin, so energy splits into a, b, sum, diff and other lines.
- **Tuning width**: the fraction of the ψ circle where g ≥ 0.5·max g. **Preferred angle** φ_i = argmax g.
- **mlp_post only**: direct logit effect L_i = W_U^T W_out^T d_i, with its frequency-k_i DFT phase.
- **Rings (unsupervised)**: spectral clustering of absolute decoder cosine similarities, with the number of clusters chosen by eigengap. The ring angle of each feature is the angle of its decoder in the cluster's top-2 PCA plane. Agreement with k_i and φ_i is reported but **not used to build any intervention**.

### 4.5 Baseline dictionaries

These get the same §4.3–4.4 measurements where they apply:
- PCA (m = k components), ICA, NMF (mlp_post only), k-means (m centroids);
- a supervised mechanism dictionary: tiles r_k·ReLU(cos(ψ_k − ψ_j)) for 2|K|·8 angles, decoders fitted by least squares;
- a **surrogate SAE**: the same SAE config trained on activations synthesized from the idealized mechanism fitted to the dev base model;
- SAEs on the three control models.

## 5. Interventions

All interventions act only on the last position, at the named hook. For an SAE intervention that maps the code z to T(z):
- the **spliced** variant writes Dec(T(z));
- the **error-preserved** variant writes Dec(T(z)) + (x − Dec(z)).

Dose–response curves are reported over the full magnitude grid. No single magnitude is selected after seeing results.

- **I1, antipodal swap (resid_mid)**: within each ring, move each active latent's activation to the feature whose ring angle is nearest its own angle + π. A **quarter-turn swap** (+π/2) is the paired control.
- **I2, ring transport (resid_mid)**: move each active ring latent to the feature nearest its ring angle + Δ_ring(δ). The calibration of Δ_ring(δ) from SAE-train inputs and labels (slope and sign of ring angle versus a+b) is frozen in stage 2. Comparisons:
  - full-angle (2×) transport;
  - oracle rotation of the fitted frequency planes;
  - rotation within PCA planes of resid_mid;
  - the real shifted input (a + 57δ, b + 57δ), since 2·57 ≡ 1 mod 113;
  - additive steering toward the target sum's tiles;
  - a random permutation of features within each ring.
- **I3, signed single-feature steering (resid_mid)**: x' = x + c·‖x̄‖·d_i for c ∈ {±0.25, ±0.5, ±1, ±2, ±4, ±8}, where ‖x̄‖ is the mean resid_mid norm. Each ring feature is steered on SAE-test inputs.
- **I4, mlp_post sanity checks (not hypotheses)**:
  - analytic Δlogits = W_U^T W_out^T Δh versus the hooked forward pass;
  - single-feature clamp steering, z_i ← α·max_i with α ∈ {0.5, 1, 2, 4, 8, 16}, where max_i is the 99th percentile of feature i's activation on SAE-train;
  - answer rewriting, clamping one ring feature per key frequency whose preferred sum equals s*.
  
  Baselines for answer rewriting: the minimum-norm oracle Δh = W_L⁺(target logit change), DiffMean / class-conditional mean E[h | s*] − E[h], top neurons, and norm-matched random directions. Efficiency is reported as the α₅₀ ratio to the oracle.
- **Side effects** for every intervention: probability on the target, probability on aliases, and KL from clean.

## 6. Candidate hypotheses

Thresholds are tentative and frozen in stage 2. Each hypothesis gives the predictions of both algorithms where they differ.

| ID | Hypothesis | Primary metric (seed level) | Tentative confirm threshold | Falsified if | Pizza / A+B predicts | Clock predicts |
|---|---|---|---|---|---|---|
| H1 | SAE features at resid_mid and mlp_post are frequency-specific ring features | fraction of alive features with held-out arc R² ≥ 0.8 at a key frequency | ≥ 0.80, and median non-key R² ≤ 0.2 | fraction < 0.5 | γ̂ ≈ 1, odd harmonics present | γ̂ ≈ 0, sum-line dominant |
| H1b | Ring features tile the ring (localized arcs) | median tuning width of ring features | below the width of the reference half-wave features (neurons at mlp_post; ReLU of random in-plane directions at resid_mid) | median width ≥ reference median | — | — |
| H2 | mlp_post ring features come in antipodal pairs (different neurons, same answer) | fraction of ring features with a partner at angle +π (±π/8), decoder cosine ≤ 0.1 and logit-effect cosine ≥ 0.9 | ≥ 0.30 and above the random within-frequency pairing null | not above the null | pairs exist | no logit-equal antipodes |
| H3 | Antipodal swap at resid_mid preserves the answer; quarter-turn destroys it | spliced accuracy after I1 (antipodal) and after the quarter-turn | antipodal ≥ 0.95 and quarter-turn ≤ 0.10 | antipodal ≤ 0.5 | antipodal accuracy ≈ clean | antipodal changes the answer |
| H4 | **Primary steering hypothesis.** Ring transport at resid_mid makes the model compute a+b+δ | transport success = P(argmax = (a+b+δ) mod p) on SAE-test inputs × test δ, spliced | ≥ 0.80, and above random within-ring permutation | ≤ 0.2 or not above the null | half-angle transport gives +δ; full-angle gives +2δ | full-angle gives +δ |
| H5 | Signed single-feature steering at resid_mid is sign-symmetric | mean over ring features of cos(φ⁺ − φ⁻), where φ^± is the frequency-k_i phase of the logit change at c = ±8 | ≥ 0.5 | ≤ 0 | +1 (V-shaped) | −1, as linear readout predicts |

**Exploratory, labelled post hoc in every write-up**:
- case-study "interesting features";
- the memorization-checkpoint SAE and the constant-attention model;
- the dictionary-size × k sweep;
- architecture ablations (TopK, JumpReLU, ReLU+L1, optional end-to-end KL);
- resid_post;
- the share of SAE capacity spent on logit-null directions.

## 7. Statistics

- **Unit of inference**: the base model.
- **Seed-level metric**: averaged over the 3 SAE seeds and over SAE-test inputs × test δ / s*.
- **Uncertainty**: hierarchical percentile bootstrap with 10,000 resamples (base seeds, then SAE seeds within each, then inputs within each), 95% CIs.
- **Decision rule**: a hypothesis is confirmed if its bootstrap one-sided p-value against its threshold survives Holm correction over the confirmatory set at family-wise α = 0.05, i.e. the fraction of bootstrap means on the wrong side of the threshold. It is falsified if the point estimate meets its "falsified if" condition. Otherwise it is inconclusive. All three outcomes are reported.
- **Reporting**:
  - Per-base-seed results are always shown next to the aggregate, as dot plots.
  - Wilson CIs are used only for descriptive within-SAE fractions, never for the confirmatory decision.
  - Negative and inconclusive results are reported with the same prominence as positive ones.

## 8. Frozen in stage 2 (allowed to change after dev exploration)

- the final thresholds in §6;
- the SAE configuration (architecture stays BatchTopK; m and k are set);
- the ring-feature R² threshold;
- the nearest-feature versus interpolated transport rule;
- the Δ_ring(δ) calibration procedure;
- whether H1–H5 are each confirmatory or exploratory.

Anything not in this list is fixed by stage 1.

## 9. Things that would make us distrust our own results

- **Linear tautology**: an mlp_post steering result that is just Δlogits = W_U^T W_out^T Δh. It is reported as a sanity check, never as a finding.
- **Hidden supervision**: interventions built with ground-truth coordinates are labelled "oracle".
- **Leakage**: selecting features or calibrating on SAE-test inputs, or evaluating on dev δ / s*. The code enforces the split by passing index sets, and tests check it.
- **Error term**: success that depends on the spliced versus error-preserved choice. Both are always reported.
- **Pseudo-replication**: claims resting on many correlated features or targets within one seed.

---

## Stage 2

(Appended after dev-seed exploration and before confirmatory runs.)
