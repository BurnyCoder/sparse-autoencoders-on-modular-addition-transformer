# Experiment 01: base models and mechanism readout

Author: Libor "Burny" Burian. Status: dev model done (seed 999); confirmatory seeds 0–9 pending stage 2.

## Question
Does the TransformerLens-4 reproduction of the Nanda et al. recipe grok and implement the known Fourier algorithm? What exactly does the MLP input look like (Clock vs Pizza)?

## Setup
- **Model**: 1 layer, 4 heads, d_model 128, d_head 32, d_mlp 512, ReLU, no LayerNorm, biases frozen at 0, untied embed/unembed. Input [a, b, 113], output logits over 113 residues (`src/modsae/model.py`).
- **Data**: all 12,769 pairs (a, b), 30% train split under data seed 598, as in the TransformerLens demo (`src/modsae/data.py`).
- **Training**: full-batch AdamW, lr 1e-3, weight decay 1, betas (0.9, 0.98), 40,000 epochs (the paper's count), float64 cross-entropy at the last position.
  - Why these hyperparameters: they are Nanda et al.'s, so the known mechanism is the ground truth we compare against. The only change is 40k epochs instead of the demo's 25k, because TL4 initialization groks later (~17–18k epochs on the dev seed).
- **Readout** (`src/modsae/mechanism.py`):
  - Key frequencies: argmax-FVE neurons, with ≥ 10 neurons at FVE ≥ 0.85.
  - Restricted / excluded logits (Nanda et al.).
  - Gradient symmetricity and distance irrelevance (Zhong et al.).
  - Regression of resid_mid on [cos/sin w_k a, cos/sin w_k b] per key frequency, with its singular values (A+B check).
  - a↔b asymmetry.
- **Cross-check**: the user's HF checkpoint (TL3, 25k epochs), loaded through TL4's own `convert_tl_checkpoint`.

## Results
Tables are generated into `results/tables/mechanism.md` by `python run.py report`.
- **Training curves**: `figures/training_base_s999.png` (memorization by ~epoch 1,500, grokking ~17–18k).
- **A+B cone**: `figures/resid_geometry_base_s999.png`. Each key-frequency plane of resid_mid is a disk whose angle carries half the sum.
- **HF cross-check**: key frequencies, cluster sizes and single-frequency neuron count match the checkpoint's model card exactly.

## Interpretation
- **The dev model implements a hybrid.**
  - Pizza part: about 94% of the MLP-input variance lies in the A+B planes.
  - Clock part: a small sum-line component cos/sin(w_k(a+b)) at two of the four key frequencies (written by attention). It is causally relevant: replacing the off-plane residual with its mean drops accuracy to 86%.
  - This fits the intermediate gradient symmetricity and distance irrelevance values.
- The HF checkpoint (different init and training length) scores Clock-like on Zhong's diagnostics. The algorithm mix may vary across base models, which motivates using the base seed as the unit of inference.

## Evidence type
Correlational (regressions, diagnostics) plus one ablation (residual mean-ablation).
