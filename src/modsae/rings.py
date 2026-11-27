"""Unsupervised recovery of feature rings from SAE decoder geometry (preregistration §4.4).

Local: spectral clustering of |cosine similarity| between decoder rows (as in Engels et al.
2024, https://arxiv.org/abs/2405.14860, which clusters SAE decoder directions to find
circular features), with the number of clusters chosen by the eigengap of the normalized
graph Laplacian, then a centered 2-D PCA of each cluster's decoders gives each feature's
ring angle.

Global context: interventions (antipodal swap, ring transport) are built from these
unsupervised rings only; ground-truth frequencies are used only to validate them.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from sklearn.cluster import SpectralClustering


def eigengap_n_clusters(affinity: np.ndarray, max_clusters: int) -> int:
    """Number of clusters = position of the largest gap among the smallest Laplacian eigenvalues."""
    deg = affinity.sum(1)
    d_inv_sqrt = 1.0 / np.sqrt(np.clip(deg, 1e-12, None))
    lap = np.eye(len(affinity)) - d_inv_sqrt[:, None] * affinity * d_inv_sqrt[None, :]
    evals = np.sort(np.linalg.eigvalsh(lap))[: max_clusters + 1]
    gaps = np.diff(evals)
    return int(np.argmax(gaps[1:]) + 2) if len(gaps) > 1 else 1  # at least 2 clusters


def recover_rings(W_dec: torch.Tensor, members: torch.Tensor, max_clusters: int = 16, n_clusters: int | None = None,
                  seed: int = 0) -> dict:
    """Cluster decoder rows `W_dec[members]` into rings and assign ring angles.

    Returns {'labels': (n_feat,) ring id or -1, 'angle': (n_feat,) radians or nan, 'n_rings': int}.
    """
    idx = members.nonzero().flatten()
    D = torch.nn.functional.normalize(W_dec[idx].double(), dim=1)
    affinity = (D @ D.T).abs().numpy()
    np.fill_diagonal(affinity, 1.0)
    n = n_clusters or eigengap_n_clusters(affinity, min(max_clusters, len(idx) - 1))
    labels_sub = SpectralClustering(n_clusters=n, affinity="precomputed", random_state=seed).fit_predict(affinity)
    labels = torch.full((W_dec.shape[0],), -1, dtype=torch.long)
    angle = torch.full((W_dec.shape[0],), float("nan"))
    labels[idx] = torch.as_tensor(labels_sub, dtype=torch.long)
    for ring in range(n):
        sel = idx[torch.as_tensor(labels_sub == ring)]
        X = W_dec[sel].double()
        Xc = X - X.mean(0)
        if len(sel) >= 3:
            V = torch.linalg.svd(Xc, full_matrices=False).Vh[:2]
            coords = Xc @ V.T
            angle[sel] = torch.atan2(coords[:, 1], coords[:, 0]).float()
    return {"labels": labels, "angle": angle, "n_rings": n}


def circular_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Absolute angular difference wrapped to [0, π]."""
    d = torch.remainder(x - y + math.pi, 2 * math.pi) - math.pi
    return d.abs()


def transport_matrix(labels: torch.Tensor, angle: torch.Tensor, shift: dict[int, float] | float) -> torch.Tensor:
    """0/1 matrix T (n_feat, n_feat): row j sends feature j's activation to the ring member nearest angle_j + shift.

    Features outside rings (label -1) map to themselves. `shift` is one angle for all rings
    or a dict ring id -> angle. Applying z' = z @ T moves activations along rings.
    """
    n = labels.shape[0]
    T = torch.zeros(n, n)
    for j in range(n):
        ring = int(labels[j])
        if ring < 0 or torch.isnan(angle[j]):
            T[j, j] = 1.0
            continue
        s = shift[ring] if isinstance(shift, dict) else shift
        members = ((labels == ring) & ~torch.isnan(angle)).nonzero().flatten()
        target = angle[j] + s
        T[j, members[circular_distance(angle[members], target).argmin()]] = 1.0
    return T
