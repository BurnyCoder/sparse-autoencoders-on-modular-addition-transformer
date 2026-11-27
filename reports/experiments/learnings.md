# Learnings

Author: Libor "Burny" Burian. Running log of what was learned, what went wrong, why, and how it was fixed. Numbers live in the generated tables (`results/tables/`); this file explains them.

## From the literature review

- The mechanism is known well enough to derive predictions before training. Nanda et al. ([2301.05217](https://arxiv.org/abs/2301.05217)) give the key-frequency rule, neuron FVE and restricted/excluded loss. Lin ([2510.00468](https://arxiv.org/abs/2510.00468)) suggests the MLP input carries "A+B" features. Zhong et al. ([2306.17844](https://arxiv.org/abs/2306.17844)) give two competing algorithms, Clock and Pizza, with diagnostics.
- SAEs on circles tend to tile them with localized latents ([2509.02565](https://arxiv.org/abs/2509.02565), [2604.28119](https://arxiv.org/abs/2604.28119)). That made "ring features" the natural hypothesis.
- DiffMean is a strong steering baseline, ahead of SAEs ([2501.17148](https://arxiv.org/abs/2501.17148)), so every steering result needs it as a comparison.

## From the design review (before any code)

- **Steering after the MLP is exactly linear here.** There is no LayerNorm and all biases are 0, so Δlogits = W_U^T W_out^T Δh. Any mlp_post steering result is computable from the decoder vector and is not a finding. We moved confirmatory steering to the MLP input and to interventions that change direction (sign flips, rotations).
- **Our first purity metric was wrong.** Every k is invertible mod 113, so "energy on harmonics of k" with a harmonic cap really measures smoothness, not frequency. It was replaced by fits in mechanism coordinates (ψ_k, r_k).
- **Pseudo-replication.** The 113 targets and the many features inside one SAE are not independent. The base model is the unit of inference, which is why there are 10 confirmatory base seeds.

## From implementation

- **TransformerLens 4 removed `HookedTransformer`.** Models are built with `TransformerBridge.boot_native`, as in the current official Grokking demo.
- **TL4 hook alias semantics (important).** `blocks.0.hook_resid_mid` is an alias of `blocks.0.ln2.hook_in`, so overwriting it changes only the MLP input, not the residual skip path. A unit test caught this when the analytic logits disagreed with the hooked forward pass. Fix: overwrite `hook_attn_out` with (x − resid_pre) to replace the full residual stream. Recorded as a stage-1 clarification in the preregistration before any results.
- **Training speed.** The hooked bridge costs about 47 ms per full-batch step against about 7 ms for the underlying `NativeModel`, which shares the same parameters. Training uses the native model; hooks are used only for analysis.
- **SAELens API details.** `SAETrainerConfig` requires `lr_end` even with a constant schedule, otherwise an assertion fails. BatchTopK SAEs are saved as JumpReLU SAEs with a learned threshold and unit-norm decoder rows, so inference L0 is close to k but not exactly k. SAELens logs through `wandb`; a small shim forwards those calls to trackio for the live dashboard.
- **Minimum-norm oracle.** A full pseudoinverse of W_L amplifies near-null directions, so at fixed norm it steered nothing. It was truncated to the 2|K| key-frequency directions (W_L is effectively rank 2|K| per Nanda et al.).
- **TL4 initialization groks later than the TL3 demo.** A 12k-epoch smoke model had not grokked; the dev model groks around epoch 17–18k. The 40k-epoch budget from the paper gives ample margin.

## From the dev base model (seed 999)

- The reproduction checks pass on the user's HF checkpoint: key frequencies, cluster sizes and single-frequency neuron count match its model card exactly. This validates the neuron-FVE reimplementation and the TL3→TL4 weight conversion. It also shows the gap from Nanda et al.'s 433 single-frequency neurons is a property of that model, not of the metric.
- **The MLP input is exactly the A+B cone.** For every key frequency, a and b enter the same 2-D plane with identical coefficients (singular values 3 and 4 of the regression are about 0), the input is symmetric under a↔b, and the plane angle carries *half* the sum (`figures/resid_geometry_base_s999.png`: two colour cycles per turn).
- Zhong et al.'s Clock/Pizza diagnostics give intermediate values on our dev model. The user's HF checkpoint (TL3 init, different training length) scores Clock-like. The algorithm may vary across base models, which supports treating the base seed as the unit of inference.

## From the first dev SAEs (BatchTopK, k = 16)

- Reconstruction was near perfect at both sites, but 86–88% of latents were dead.
- **The live latents were not ring features.** They mixed frequency planes and came in identical pairs that fire on about half the inputs: ReLU(u·x) and ReLU(−u·x).
- Interpretation: resid_mid is about 8-dimensional (4 planes). With k ≥ 8 the SAE can reconstruct with a *dense linear ± code* in any rotated basis, so the sparsity limit never binds and nothing favours plane-aligned or localized features. This is the "global capture" mode of Bhalla et al.
- Consequence: the stage-1 constraint "k ≥ 2|K|", adopted from the design review to avoid cross-frequency lookup features, excludes the regime where tiling is forced. The k × m sweep on the dev seed tests this; see the dev-sweep experiment folder.
