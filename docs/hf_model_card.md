---
license: mit
library_name: saelens
tags:
  - mechanistic-interpretability
  - sparse-autoencoder
  - sae
  - grokking
  - modular-addition
  - transformer-lens
datasets:
  - custom
---

# SAEs on a grokked modular-addition transformer

Base models and sparse autoencoders (SAEs) from the project
[sparse-autoencoders-on-modular-addition-transformer](https://github.com/BurnyCoder/sparse-autoencoders-on-modular-addition-transformer)
by Libor "Burny" Burian. Results, preregistration and paper are in the GitHub repository.

## Contents

- `base/<run>/model.pt`: the 1-layer modular-addition transformer of [Nanda et al. 2023](https://arxiv.org/abs/2301.05217). It uses the TransformerLens 4 demo recipe: p = 113, d_model 128, 4 heads, d_mlp 512, ReLU, no LayerNorm, biases frozen at 0, 30% train split (data seed 598), AdamW lr 1e-3, wd 1, 40k full-batch epochs. Next to it, `checkpoints.pt` holds a state dict every 100 epochs, and `training.json` / `mechanism.json` hold the training curves and the Fourier mechanism analysis.
- `saes/<run_id>/`: SAELens inference SAEs (`cfg.json` + `sae_weights.safetensors`; BatchTopK SAEs are saved as JumpReLU with a global threshold) trained on last-position (`=`) activations at `resid_mid` (the MLP input) or `mlp_post`. Each folder also has `metrics.json` (SAE-test FVU, logit FVU, L0, dead fraction, spliced accuracy/KL), and where computed `characterization.json` (features scored against the mechanism) and `steering.json`.

## Loading

```python
import torch
from sae_lens import SAE
from modsae import model  # from the GitHub repo
m, cfg, history = model.load_model("base/base_s999/model.pt", device="cpu")
sae = SAE.load_from_disk("saes/sae_base_s999_resid_mid_batchtopk_m256_k16_r0", device="cpu")
```

Note: in TransformerLens 4 the hook alias `blocks.0.hook_resid_mid` only feeds the MLP branch. To replace the whole residual stream, overwrite `blocks.0.hook_attn_out` (see `src/modsae/interventions.py`).
