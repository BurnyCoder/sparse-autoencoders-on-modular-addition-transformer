"""Run configuration: paths from .env and dataclass configs for each phase.

Local: `Paths` reads directories from environment variables (loaded from `.env` via
python-dotenv, https://pypi.org/project/python-dotenv/). `BaseModelConfig` and
`SAEConfig` hold every hyperparameter of a run and give it a stable run id.

Global context: run ids name the folders in results/ and checkpoints/, which is
how sweeps skip finished runs and how report.py finds results.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]  # repository root (src/modsae/config.py -> root)
load_dotenv(ROOT / ".env")  # no-op when .env is absent; values never override real env vars


@dataclass(frozen=True)
class Paths:
    """Output directories, resolved relative to the repository root."""

    checkpoints: Path = field(default_factory=lambda: ROOT / os.getenv("CHECKPOINT_DIR", "checkpoints"))
    cache: Path = field(default_factory=lambda: ROOT / os.getenv("CACHE_DIR", "cache"))
    results: Path = field(default_factory=lambda: ROOT / os.getenv("RESULTS_DIR", "results"))
    logs: Path = field(default_factory=lambda: ROOT / os.getenv("LOGS_DIR", "logs"))
    figures: Path = field(default_factory=lambda: ROOT / "figures")

    def make(self) -> "Paths":
        for p in (self.checkpoints, self.cache, self.results, self.logs, self.figures):
            p.mkdir(parents=True, exist_ok=True)
        return self


def device() -> str:
    """Device from DEVICE env var, falling back to CPU if CUDA is unavailable."""
    import torch

    want = os.getenv("DEVICE", "cuda")
    return want if (want != "cuda" or torch.cuda.is_available()) else "cpu"


@dataclass(frozen=True)
class BaseModelConfig:
    """Nanda et al. 2023 / TransformerLens demo recipe (preregistration §2)."""

    p: int = 113
    d_model: int = 128
    n_heads: int = 4
    d_head: int = 32
    d_mlp: int = 512
    seed: int = 999  # model init seed (demo: 999)
    data_seed: int = 598  # base train/test split seed (demo: 598)
    frac_train: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 1.0
    betas: tuple[float, float] = (0.9, 0.98)
    num_epochs: int = 40_000  # paper; the demo uses 25k
    checkpoint_every: int = 100
    log_every: int = 100
    constant_attention: bool = False  # control: W_Q = W_K = 0 frozen -> uniform attention

    @property
    def run_id(self) -> str:
        suffix = "_constattn" if self.constant_attention else ""
        return f"base_s{self.seed}{suffix}"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SAEConfig:
    """One SAE training run (preregistration §3, planned-runs.md phase B)."""

    base_run: str = "base_s999"  # which base model (or control) the activations come from
    site: str = "resid_mid"  # resid_mid | mlp_post | resid_post
    arch: str = "batchtopk"  # batchtopk | topk | jumprelu | standard
    d_sae: int = 256
    k: int = 16  # target L0 for (Batch)TopK
    l1_coefficient: float = 5.0  # standard (ReLU + L1) only
    l0_coefficient: float = 1.0  # jumprelu only
    seed: int = 0
    lr: float = 3e-4  # Bussmann et al. 2024 App. A.1
    batch_size: int = 512
    total_samples: int = 3_000_000
    variant: str = ""  # optional tag, e.g. "surrogate"

    @property
    def run_id(self) -> str:
        tag = f"_{self.variant}" if self.variant else ""
        return f"sae_{self.base_run}_{self.site}_{self.arch}_m{self.d_sae}_k{self.k}{tag}_r{self.seed}"

    def to_dict(self) -> dict:
        return asdict(self)


# Hook names (TransformerLens 4 keeps the HookedTransformer aliases).
HOOKS = {
    "resid_mid": "blocks.0.hook_resid_mid",
    "mlp_post": "blocks.0.mlp.hook_post",
    "resid_post": "blocks.0.hook_resid_post",
}
