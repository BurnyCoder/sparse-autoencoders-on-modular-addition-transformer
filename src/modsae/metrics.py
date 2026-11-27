"""Reconstruction and output metrics (preregistration §4.3).

Global context: shared by SAE evaluation, baseline dictionaries and interventions.
Definitions follow Gao et al. 2024 (normalized MSE = FVU, https://arxiv.org/abs/2406.04093)
and Karvonen et al. 2025 (loss recovered, https://arxiv.org/abs/2503.09532).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def fvu(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """Fraction of variance unexplained = sum ||x − x̂||² / sum ||x − mean(x)||²."""
    return ((x - x_hat).double().pow(2).sum() / (x - x.mean(0)).double().pow(2).sum().clamp_min(1e-30)).item()


def output_metrics(logits: torch.Tensor, labels: torch.Tensor, clean_logits: torch.Tensor | None = None) -> dict:
    """CE, accuracy and correct-answer margin; plus KL(clean || new) when clean logits are given."""
    lg = logits.double()
    out = {"ce": F.cross_entropy(lg, labels).item(), "acc": (lg.argmax(-1) == labels).double().mean().item()}
    correct = lg[torch.arange(len(labels)), labels]
    other = lg.clone()
    other[torch.arange(len(labels)), labels] = -torch.inf
    out["margin"] = (correct - other.max(-1).values).mean().item()
    if clean_logits is not None:
        logp_clean = F.log_softmax(clean_logits.double(), -1)
        out["kl"] = (logp_clean.exp() * (logp_clean - F.log_softmax(lg, -1))).sum(-1).mean().item()
    return out


def loss_recovered(ce_patched: float, ce_clean: float, ce_zero: float) -> float:
    """(CE_zero − CE_patched) / (CE_zero − CE_clean): 1 = as good as clean, 0 = as bad as zero-ablation."""
    return (ce_zero - ce_patched) / max(ce_zero - ce_clean, 1e-30)


def null_space_fraction(vectors: torch.Tensor, W_L: torch.Tensor, rtol: float = 1e-6) -> torch.Tensor:
    """Fraction of each row's squared norm lying in null(W_L) (rows are in neuron space; W_L: (d_mlp, p))."""
    U, S, _ = torch.linalg.svd(W_L.double(), full_matrices=False)
    row_space = U[:, S > rtol * S.max()]  # orthonormal basis of the directions W_L does not annihilate
    v = vectors.double()
    in_row = (v @ row_space).pow(2).sum(-1)
    return (1 - in_row / v.pow(2).sum(-1).clamp_min(1e-30)).float()
