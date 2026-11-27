# Experiment 02: dev-seed SAE exploration (sweeps, training health, feature types)

Author: Libor "Burny" Burian. Status: exploratory, dev seed 999 only. These results inform the stage-2 preregistration and are not confirmatory.

## Question
Which SAE configurations are healthy (few dead latents) and faithful (spliced accuracy, logit FVU)? Do their features align with the mechanism's frequency rings?

## Runs
All runs are listed in `results/tables/sae_dev.md`; the heatmaps are in `figures/sweep_base_s999.png`.
1. **Uncentred BatchTopK sweep**: k ∈ {2, 3, 4, 6, 8, 12, 16} × m ∈ {64, 128, 256, 512} at resid_mid and mlp_post. SAELens defaults: lr 3e-4, 3M samples, `expected_average_only_in` normalization.
2. **Training-health probes**: TopK, ReLU+L1 (Standard) with L1 ∈ {2, 8}, lr 1e-3 with 10M samples.
3. **Centring**: train on x − μ and fold μ into b_dec, equivalent to initializing the decoder bias at the data mean, as recommended by Bricken et al. 2023 and Gao et al. 2024.
4. **Centred BatchTopK sweep**: k ∈ {4, 6, 8} × m ∈ {64, 128, 256, 512} at both sites.

## Findings so far (exploratory)
- **Dense linear codes for k ≥ 2|K|.** resid_mid is about 8-dimensional (4 A+B planes). With k ≥ 8 the SAE reconstructs nearly perfectly with ± pairs of half-space features, in a basis that mixes the planes. Rotations of such a code are equally good under MSE, so the features are not identifiable.
- **Composed features.** With fewer active latents, or with many live latents (TopK at lr 1e-3; L1 SAEs with 254 live latents), features are conjunctions across several frequency rings. They fire on sparse, lattice-like sets of (a, b), not on single-frequency arcs (scratch galleries rendered on the dev seed). This matches the toy-model observation that SAEs learn composed rather than underlying features ([LessWrong](https://www.lesswrong.com/posts/a5wwqza2cY3W7L9cj/sparse-autoencoders-find-composed-features-in-small-toy)).
- **Dead latents are a training problem, not the inference-threshold conversion.** In a direct check, training-mode BatchTopK and the converted JumpReLU SAE have identical live counts.
  - Uncentred mlp_post SAEs collapse to 3–17 live latents at m ≥ 128.
  - Centring fixes the collapse (e.g. m = 256, k = 8: 59% live, 100% spliced accuracy), but the healthy SAE's features are again mostly composed (2% ring features).
  - Notably, *collapsed* SAEs keep mostly clean ring features (arc R² ≈ 0.9). The first features an SAE learns align with the mechanism; extra capacity goes into conjunctions.
- **Sign symmetry of ring features.** The few ring features at resid_mid steer sign-symmetrically: the frequency-k phase of the logit change is the same for ±c (H5 metric ≈ 0.9 on the dev seed). This is the half-angle (|cos|) prediction. It is a property of the MLP, so random in-plane directions should behave the same way; a control is included.

## Bugs found and fixed here
- **Transport rule.** The naive half-angle shift w_k δ/2 is only defined mod π. The exact shift is w_k·57·δ, since 57 = 2⁻¹ mod 113 and (a + 57δ, b + 57δ) multiplies z_k by e^{iw_k·57δ}.
- **Oracle baseline.** Rotating only the A+B planes ignores the Clock component. It was replaced by an exact shift of all fitted key-frequency Fourier components (A+B and sum lines), which reaches 83% on the dev seed versus 100% for the real shifted input.
