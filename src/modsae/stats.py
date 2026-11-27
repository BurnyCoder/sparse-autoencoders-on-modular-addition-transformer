"""Statistics: hierarchical bootstrap, Holm correction, Wilson intervals (preregistration §7).

Global context: the base model is the unit of inference; SAE seeds are nested inside base
seeds and inputs inside SAE seeds. Confirmatory decisions use `hierarchical_bootstrap`
+ `holm`; Wilson intervals are descriptive only.
"""

from __future__ import annotations

import math

import numpy as np


def hierarchical_bootstrap(groups: list[list[np.ndarray]], n_boot: int = 10_000, seed: int = 0) -> np.ndarray:
    """Bootstrap distribution of the grand mean for nested data.

    groups[i][j] is the array of per-input values for base seed i and SAE seed j.
    Each resample draws base seeds with replacement, then SAE seeds within each, then
    inputs within each; the statistic is the mean of per-base-seed means.
    """
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        seed_means = []
        for i in rng.integers(0, len(groups), len(groups)):
            sae_means = []
            for j in rng.integers(0, len(groups[i]), len(groups[i])):
                vals = groups[i][j]
                sae_means.append(vals[rng.integers(0, len(vals), len(vals))].mean())
            seed_means.append(np.mean(sae_means))
        out[b] = np.mean(seed_means)
    return out


def percentile_ci(boot: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    alpha = (1 - level) / 2
    return float(np.quantile(boot, alpha)), float(np.quantile(boot, 1 - alpha))


def one_sided_p(boot: np.ndarray, threshold: float, direction: str = "greater") -> float:
    """Bootstrap p-value for H: mean > threshold (fraction of resamples on the wrong side; floor 1/n)."""
    wrong = (boot <= threshold) if direction == "greater" else (boot >= threshold)
    return max(float(wrong.mean()), 1.0 / len(boot))


def holm(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Holm–Bonferroni step-down: returns which hypotheses are rejected at family-wise level alpha."""
    order = sorted(pvals, key=pvals.get)
    m, rejected, still = len(order), {}, True
    for rank, name in enumerate(order):
        still = still and pvals[name] <= alpha / (m - rank)
        rejected[name] = still
    return rejected


def wilson(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (descriptive use only)."""
    if n == 0:
        return float("nan"), float("nan")
    phat = successes / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half
