"""Run the model on the full input grid and cache last-position activations.

Global context: the grid has only 12,769 inputs, so every analysis works on exact
activations for all inputs (no sampling). Rows are indexed idx = a*p + b (data.py).
"""

from __future__ import annotations

import torch
from transformer_lens.model_bridge import TransformerBridge

from modsae import data
from modsae.config import HOOKS


@torch.no_grad()
def run_grid(model: TransformerBridge, p: int, sites: tuple[str, ...] = ("resid_mid", "mlp_post", "resid_post"),
             device: str | None = None) -> dict[str, torch.Tensor]:
    """Return {'logits': (p*p, p), site: (p*p, d_site)} at the last ('=') position, float32 on CPU."""
    device = device or next(model.parameters()).device
    tokens, _ = data.make_dataset(p)
    names = {HOOKS[s]: s for s in sites}
    logits, cache = model.run_with_cache(tokens.to(device), names_filter=lambda n: n in names)
    out = {"logits": logits[:, -1].float().cpu()}
    for hook_name, site in names.items():
        out[site] = cache[hook_name][:, -1].float().cpu()
    return out


def weights(model: TransformerBridge) -> dict[str, torch.Tensor]:
    """Weights used by analytic computations (TransformerLens conventions, float32 CPU).

    W_in (d_model, d_mlp), W_out (d_mlp, d_model), W_U (d_model, p), W_E (p+1, d_model).
    W_L = W_out @ W_U maps MLP neuron activations to logits (Nanda et al.'s neuron-logit map).
    """
    w = {
        "W_E": model.W_E.detach().float().cpu(),
        "W_pos": model.W_pos.detach().float().cpu(),
        "W_in": model.W_in[0].detach().float().cpu(),
        "b_in": model.b_in[0].detach().float().cpu(),
        "W_out": model.W_out[0].detach().float().cpu(),
        "b_out": model.b_out[0].detach().float().cpu(),
        "W_U": model.W_U.detach().float().cpu(),
        "b_U": model.b_U.detach().float().cpu(),
    }
    w["W_L"] = w["W_out"] @ w["W_U"]
    return w
