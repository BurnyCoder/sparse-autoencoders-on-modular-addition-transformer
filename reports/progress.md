# Progress

Author: Libor "Burny" Burian.

## 2026-11-27 16:15, dev-seed exploration (stage 1 committed; stage 2 not yet frozen)

**Done**
- Setup, literature review with an independent fact-check, and the stage-1 preregistration, all pushed before any results.
- TDD pipeline: 27 tests passing. They include the JS-vs-Python forward-pass equality check for the website and the TL4 hook-semantics test.
- Dev base model `base_s999`, 40k epochs: grokked (test accuracy 100%), uploaded to Hugging Face ([BurnyCoder/modular-addition-saes](https://huggingface.co/BurnyCoder/modular-addition-saes), with collection).
- Reproduction cross-check on the user's HF checkpoint: key frequencies, cluster sizes and single-frequency neuron count match its model card exactly.
- Mechanism of the dev model (tables: `results/tables/mechanism.md`; figure: `figures/resid_geometry_base_s999.png`):
  - **Positive:** the MLP input is an exact A+B cone in each key-frequency plane (rank-2 regression, a↔b symmetric, half-angle encoding of the sum).
  - **Positive:** a small but causally important Clock-like residual (sum-line components at k = 30 and 36). Removing it drops accuracy to 86%. Together these make the model a **Clock–Pizza hybrid**.
- SAE exploration on the dev seed (exploratory; the numbers are in `results/` and the generated tables):
  - **Negative:** BatchTopK SAEs with k ≥ 8 at resid_mid reconstruct perfectly, but learn dense ± (rotation-ambiguous) features that mix frequency planes.
  - **Negative:** with smaller k, or with more live latents (TopK at higher learning rate, L1 SAEs), the features are *conjunctions across rings* (composed features), not per-frequency ring features.
  - **Negative:** many latents are dead, caused by training rather than by SAELens' inference-threshold conversion (checked directly). This is worst at mlp_post, where 95–99% are dead at m ≥ 128.
  - **Positive (few features):** the ring features that do exist show the sign-symmetric (V-shaped) steering response the Pizza mechanism predicts.
  - Two bugs in the transport rule and the oracle were found and fixed on the dev seed. The naive half-angle rule ignored the mod-2π lift; the correct rule multiplies by 57 = 2⁻¹ mod 113. The oracle missed the Clock component. An exact Fourier-component-shift oracle now reaches 83%.

**Now**: an SAE training-health grid on the dev seed (architecture × learning rate, selected only on val FVU and dead fraction) and the remaining mlp_post sweep.

**Next**:
- Write the stage-2 preregistration: freeze the config with a health/faithfulness rule and document the deviations (k below 2|K|, the transport-rule correction, a possible architecture change if BatchTopK stays unhealthy).
- Run the confirmatory seeds 0–9 (about 1 hour of compute), then the controls, ablations, paper, site and final reviews.

**Overall**: about 35% of the plan done. Remaining compute is a few GPU-hours; the remaining wall-clock time goes mostly to analysis and writing.
