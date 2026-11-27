# State-of-the-art review: the methods this project builds on

Author: Libor "Burny" Burian. Date: 2026-11-27. Sources and quotes: [literature-review.md](literature-review.md).

This file answers one question per pipeline component: what is the current best practice, and what does this project use? Where we deviate, it says why.

## Base model: grokked 1-layer modular-addition transformer

| Item | State of the art | This project |
|---|---|---|
| Recipe | Nanda et al. 2023 ([2301.05217](https://arxiv.org/abs/2301.05217)): p = 113, 1L, d_model 128, 4 heads, d_mlp 512, ReLU, no LN, 30% train, AdamW lr 1e-3 wd 1, 40k full-batch epochs | Same. TransformerLens 4 demo seeds and splits (seed 999, data seed 598, betas (0.9, 0.98), biases frozen at 0), but 40k epochs instead of the demo's 25k |
| Mechanism readout | Fourier analysis of W_E, neuron FVE at one frequency, restricted/excluded loss (Nanda et al.); Clock vs Pizza diagnostics (Zhong et al. 2023, [2306.17844](https://arxiv.org/abs/2306.17844)) | All of them, reimplemented and checked on the user's HF checkpoint |
| Seeds | Representations vary across seeds (Chughtai et al. 2023, [2302.03025](https://arxiv.org/abs/2302.03025)) | One dev seed plus 10 confirmatory seeds; the base seed is the unit of inference |

## Sparse dictionary learning

| Item | State of the art | This project |
|---|---|---|
| Architecture | TopK ([2406.04093](https://arxiv.org/abs/2406.04093)), BatchTopK ([2412.06410](https://arxiv.org/abs/2412.06410)), JumpReLU ([2407.14435](https://arxiv.org/abs/2407.14435)) beat ReLU+L1 at fixed L0. Matryoshka SAEs lead on disentanglement in SAEBench ([2503.09532](https://arxiv.org/abs/2503.09532)) | BatchTopK primary (direct control of mean L0; the closest prior work, [2606.09940](https://arxiv.org/abs/2606.09940), uses BatchTopK crosscoders in its main experiments). TopK, JumpReLU and ReLU+L1 as ablations |
| Training library | SAELens 6.x (`SAETrainer` with custom data providers) | SAELens 6.56.4. Fallback: the BatchTopK reference code |
| Training recipe | Adam lr ≈ 3e-4, decoder-normalized, AuxK for dead latents, activation normalization | SAELens defaults (lr 3e-4 as in [2412.06410](https://arxiv.org/abs/2412.06410); unit-norm decoder rows at save; AuxK on dead latents) with `normalize_activations="expected_average_only_in"`. Batches drawn from about 10k in-memory activation vectors, so a train/test gap check is mandatory |
| Dictionary size, L0 | SAEBench trains L0 from 20 to 1000 on LLMs. For this toy model we reason (our argument, not a cited result) that L0 must cover the number of independent sub-manifolds | k ≥ 2 × number of key frequencies (≈ 10–16); m ∈ {128 … 1024} swept |
| Functional training | End-to-end KL SAEs ([2405.12241](https://arxiv.org/abs/2405.12241)) find more functionally relevant features | Optional ablation, because most mlp_post variance is predicted to be logit-null |
| Evaluation | NMSE/FVU, L0, dead fraction, spliced-in ΔCE/KL, loss recovered; downstream tasks (SAEBench) | All of these, plus **logit-space FVU** and the fraction of decoder norm in null(W_L). Feature interpretability is scored against ground-truth mechanism coordinates instead of LLM auto-interpretation |

## Interpreting features on manifolds

| Item | State of the art | This project |
|---|---|---|
| Discovering multi-dimensional features | Decoder-cosine clustering (Engels et al., [2405.14860](https://arxiv.org/abs/2405.14860)); Ising-coupling communities (Bhalla et al., [2604.28119](https://arxiv.org/abs/2604.28119)) | Spectral clustering of decoder cosines gives rings; a 2-D PCA of each ring gives feature angles. Ground truth (key frequency, ψ_k) is used only to validate |
| Expected geometry | SAEs tile circles with localized latents ([2509.02565](https://arxiv.org/abs/2509.02565)); "dilution" between global and local capture ([2604.28119](https://arxiv.org/abs/2604.28119)) | Arc model F ≈ r^γ·g(ψ) at each key frequency, with nulls at non-key frequencies and a locality-destroying rotation |
| Baseline dictionaries | Supervised dictionaries (Makelov et al., [2405.08366](https://arxiv.org/abs/2405.08366)); classical unsupervised factorizations (PCA, ICA, NMF, k-means/VQ; Michaud et al. [2509.02565](https://arxiv.org/abs/2509.02565) relate SAEs on manifolds to VQ) | PCA, ICA, NMF, k-means, a supervised mechanism dictionary, and a surrogate SAE trained on idealized-mechanism activations |

## Steering and interventions

| Item | State of the art | This project |
|---|---|---|
| SAE steering | Clamp to multiples of max activation (Templeton et al. 2024); add decoder vectors | Clamp/add at resid_mid and mlp_post, with dose–response over a ± grid instead of a single chosen α |
| Strong baselines | Prompting > finetuning > representation steering; among representation methods ReFT-r1 steers best, and DiffMean leads concept detection and beats SAEs at both (AxBench, [2501.17148](https://arxiv.org/abs/2501.17148)). Prompting has no analogue here because the model's only input is (a, b) | DiffMean / class-conditional means, a minimum-norm oracle (pseudoinverse of W_L), oracle Fourier-plane rotation, PCA-plane rotation, real counterfactual inputs, norm-matched random directions |
| Manifold steering | Moving along circles causally changes modular-arithmetic answers (Engels et al.); steering between centroids is smooth while single features are "often ineffective or brittle" (Bhalla et al.) | Ring transport (rotate active latents along the recovered ring) to make the model compute a+b+δ; antipodal swaps |
| Error term | Keep ε(x) = x − x̂ in interventions (Marks et al., [2403.19647](https://arxiv.org/abs/2403.19647)); sufficiency vs necessity (Makelov et al.) | Spliced (no error) as primary for transport, error-preserved as secondary, and an oracle-rotated-error diagnostic |
| Linearity caveat | Not discussed in the prior work we reviewed | Steering after the MLP is exactly linear in the logits here (no LN, zero biases), so mlp_post steering is only a sanity check against analytic predictions. The confirmatory steering runs at resid_mid, through the ReLU |

## Experimental methodology

Following the CLAUDE.md research guides (Lones 2021, [2108.02497](https://arxiv.org/abs/2108.02497); NeurIPS checklist; Lipton & Steinhardt 2018, [1807.03341](https://arxiv.org/abs/1807.03341)):
- preregistration before results, with a two-stage exploratory → confirmatory split;
- held-out inputs split by unordered pair to avoid (a,b)/(b,a) near-duplicates;
- a dev base seed for exploration and untouched confirmatory seeds;
- hierarchical bootstrap with Holm correction;
- every reported number generated by script from `results/`.
