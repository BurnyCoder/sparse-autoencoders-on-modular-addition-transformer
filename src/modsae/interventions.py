"""Interventions at resid_mid and mlp_post (preregistration §5).

Local: computes logits after replacing the last-position activation at a hook, either
with the hooked TransformerLens forward pass (source of truth) or with an exact analytic
path for the 1-layer, LayerNorm-free, zero-bias model (fast; equality is unit-tested).
Implements SAE code edits (ring transport, antipodal swap), signed single-feature steering,
spliced vs error-preserved writing, and the baselines (oracle plane rotation, DiffMean,
minimum-norm oracle, norm-matched random directions).

Global context: H3–H5 and the mlp_post sanity checks are computed here. The error-term
convention follows Marks et al. 2024 (https://arxiv.org/abs/2403.19647): x = x̂ + ε.
"""

from __future__ import annotations

import math

import torch

from modsae import data
from modsae.config import HOOKS


# ---------------------------------------------------------------- forward passes

@torch.no_grad()
def hooked_logits(model, p: int, idx: torch.Tensor, site: str, new_act: torch.Tensor) -> torch.Tensor:
    """Logits at '=' for inputs `idx` after overwriting the last-position activation at `site`.

    site 'resid_mid' replaces the *full residual stream* after attention (MLP input and the
    skip path), i.e. TransformerLens-3 `hook_resid_mid` semantics. In TransformerLens 4 the
    alias `blocks.0.hook_resid_mid` resolves to `blocks.0.ln2.hook_in`, which feeds only the
    MLP branch (verified on this model), so we overwrite the attention output instead:
    attn_out' = x − resid_pre gives resid_mid = resid_pre + attn_out' = x on both paths.
    site 'mlp_input' overwrites only the MLP branch input (the TL4 alias behaviour).
    """
    device = next(model.parameters()).device
    tokens, _ = data.make_dataset(p)
    new = new_act.to(device)
    store = {}

    def keep_pre(act, hook):
        store["pre"] = act[:, -1].clone()
        return act

    def overwrite(act, hook):
        act = act.clone()
        act[:, -1] = new.to(act.dtype)
        return act

    def overwrite_attn(act, hook):
        act = act.clone()
        act[:, -1] = (new - store["pre"]).to(act.dtype)
        return act

    if site == "resid_mid":
        hooks = [("blocks.0.hook_resid_pre", keep_pre), ("blocks.0.hook_attn_out", overwrite_attn)]
    elif site == "mlp_input":
        hooks = [(HOOKS["resid_mid"], overwrite)]
    else:
        hooks = [(HOOKS[site], overwrite)]
    out = model.run_with_hooks(tokens[idx].to(device), fwd_hooks=hooks)
    return out[:, -1].float().cpu()


def logits_from_resid_mid(x: torch.Tensor, w: dict, skip: torch.Tensor | None = None) -> torch.Tensor:
    """Exact logits from resid_mid for this architecture: skip + W_out ReLU(x W_in + b_in) + b_out, then W_U.

    skip defaults to x (full residual replacement). Passing the original resid_mid as `skip`
    gives the MLP-input-only intervention.
    """
    h = torch.relu(x @ w["W_in"] + w["b_in"])
    skip = x if skip is None else skip
    return (skip + h @ w["W_out"] + w["b_out"]) @ w["W_U"] + w["b_U"]


def logits_from_mlp_post(h: torch.Tensor, resid_mid: torch.Tensor, w: dict) -> torch.Tensor:
    """Exact logits from MLP post-activations (linear in h): (resid_mid + h W_out + b_out) W_U + b_U."""
    return (resid_mid + h @ w["W_out"] + w["b_out"]) @ w["W_U"] + w["b_U"]


# ---------------------------------------------------------------- SAE code edits

@torch.no_grad()
def sae_write(sae, x: torch.Tensor, z_new: torch.Tensor, mode: str) -> torch.Tensor:
    """Activation to write back: spliced = Dec(z'), error-preserved = Dec(z') + (x − Dec(Enc(x)))."""
    dev = sae.W_dec.device
    x_d, z_d = x.to(dev), z_new.to(dev)
    out = sae.decode(z_d)
    if mode == "error":
        out = out + (x_d - sae.decode(sae.encode(x_d)))
    elif mode != "spliced":
        raise ValueError(mode)
    return out.float().cpu()


@torch.no_grad()
def encode(sae, x: torch.Tensor) -> torch.Tensor:
    return sae.encode(x.to(sae.W_dec.device)).float().cpu()


def apply_transport(z: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """Move latent activations along rings: z' = z @ T (T from rings.transport_matrix)."""
    return z @ T


def signed_steer(x: torch.Tensor, direction: torch.Tensor, c: float, scale: float) -> torch.Tensor:
    """x + c · scale · d (d unit-norm decoder row; scale = mean activation norm), preregistration I3."""
    return x + c * scale * direction[None, :]


# ---------------------------------------------------------------- oracle / baseline edits

def rotate_in_planes(x: torch.Tensor, planes: dict[int, torch.Tensor], angles: dict[int, float], center: torch.Tensor) -> torch.Tensor:
    """Rotate the centered component of x inside each 2-D plane (rows orthonormal) by the given angle."""
    out = x.clone()
    xc = x - center
    for k, P in planes.items():
        coords = xc @ P.T  # (N, 2)
        c, s = math.cos(angles[k]), math.sin(angles[k])
        rot = torch.stack([c * coords[:, 0] - s * coords[:, 1], s * coords[:, 0] + c * coords[:, 1]], 1)
        out = out + (rot - coords) @ P
    return out


def diffmean_direction(acts: torch.Tensor, labels: torch.Tensor, target: int) -> torch.Tensor:
    """DiffMean (AxBench, https://arxiv.org/abs/2501.17148): mean(acts | label = target) − mean(acts)."""
    return acts[labels == target].mean(0) - acts.mean(0)


def min_norm_mlp_post(target_logit_change: torch.Tensor, W_L: torch.Tensor, rank: int | None = None) -> torch.Tensor:
    """Minimum-norm Δh with Δh W_L = t (pseudoinverse of the neuron-logit map W_L).

    With `rank`, the pseudoinverse is truncated to the top singular directions of W_L. Nanda et al.
    report W_L is effectively rank 10 (5 key frequencies x cos/sin), and the full pseudoinverse
    amplifies near-null directions, which makes it a poor efficiency oracle at fixed norm.
    """
    if rank is None:
        return target_logit_change.double() @ torch.linalg.pinv(W_L.double())
    U, S, Vh = torch.linalg.svd(W_L.double(), full_matrices=False)
    return (target_logit_change.double() @ Vh[:rank].T / S[:rank]) @ U[:, :rank].T


def random_unit_directions(n: int, d: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.nn.functional.normalize(torch.randn(n, d, generator=g), dim=1)
