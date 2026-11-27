"""Dataset and splits for modular addition.

Local: builds the full input grid [a, b, '='] with labels (a+b) mod p, the base-model
train/test split, the SAE train/val/test split, and dev/test splits of shifts δ and
target sums.

Global context: the base-model split reproduces the TransformerLens Grokking demo
(https://github.com/TransformerLensOrg/TransformerLens/blob/main/demos/Grokking_Demo.ipynb):
idx = a*p + b, '=' token = p, torch.manual_seed(data_seed); torch.randperm(p*p).
The SAE split groups (a,b) with (b,a) so near-duplicate inputs never straddle splits
(preregistration §3).
"""

from __future__ import annotations

import torch


def index_to_ab(idx: torch.Tensor, p: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Map flat grid indices to operands (a, b) with idx = a*p + b (demo layout)."""
    return idx // p, idx % p


def make_dataset(p: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return tokens (p*p, 3) = [a, b, p] and labels (a+b) mod p, both int64.

    The demo builds a with einops.repeat(arange(p), "i -> (i j)") and b with
    "j -> (i j)", which is exactly idx // p and idx % p.
    """
    a, b = index_to_ab(torch.arange(p * p), p)
    equals = torch.full_like(a, p)  # the '=' token has id p (113 for p = 113)
    tokens = torch.stack([a, b, equals], dim=1)
    labels = (a + b) % p
    return tokens, labels


def base_split(p: int, frac_train: float, data_seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Base-model train/test indices, identical to the TransformerLens demo.

    Uses the global RNG like the demo does (torch.manual_seed then randperm), then
    restores the previous RNG state so callers are unaffected.
    """
    state = torch.random.get_rng_state()
    torch.manual_seed(data_seed)
    perm = torch.randperm(p * p)
    torch.random.set_rng_state(state)
    cutoff = int(p * p * frac_train)  # 3830 for p = 113, frac 0.3
    return perm[:cutoff], perm[cutoff:]


def sae_split(p: int, seed: int = 0, fracs: tuple[float, float, float] = (0.8, 0.1, 0.1)) -> dict[str, torch.Tensor]:
    """Split all p*p inputs into SAE train/val/test by unordered pair {a <= b}.

    Local: enumerate the p(p+1)/2 unordered pairs, shuffle them with a private
    generator, cut by `fracs`, then expand each pair to both orderings.
    Global context: prevents (a,b)/(b,a) near-duplicates from leaking between the
    data the SAE is trained or calibrated on and the data its metrics are reported on.
    """
    pairs = [(a, b) for a in range(p) for b in range(a, p)]
    gen = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(pairs), generator=gen).tolist()
    n_train = round(fracs[0] * len(pairs))
    n_val = round(fracs[1] * len(pairs))
    groups = {"train": order[:n_train], "val": order[n_train:n_train + n_val], "test": order[n_train + n_val:]}
    out = {}
    for name, pair_ids in groups.items():
        idx = set()
        for j in pair_ids:
            a, b = pairs[j]
            idx.add(a * p + b)  # ordering (a, b)
            idx.add(b * p + a)  # ordering (b, a); same index when a == b
        out[name] = torch.tensor(sorted(idx), dtype=torch.long)
    return out


def dev_test_split(values: torch.Tensor, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Shuffle `values` with a private generator; first half dev, second half test."""
    gen = torch.Generator().manual_seed(seed)
    perm = values[torch.randperm(len(values), generator=gen)]
    half = (len(values) + 1) // 2
    return perm[:half], perm[half:]
