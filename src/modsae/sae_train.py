"""Train SAEs on cached activations with SAELens `SAETrainer` and an in-memory data provider.

Local: builds a SAELens training SAE for the requested architecture (BatchTopK, TopK,
JumpReLU, Standard ReLU+L1), streams random mini-batches of SAE-train activations from
GPU memory, logs live metrics (train MSE/L0/dead latents, SAE-val FVU) to trackio, and
saves the inference SAE (cfg.json + sae_weights.safetensors; BatchTopK is saved as
JumpReLU with a learned threshold, per SAELens docs).

Global context: the SAE-train / SAE-val split comes from data.sae_split; SAE-test is
never seen here (preregistration §3). API sources: SAELens training docs
(https://decoderesearch.github.io/SAELens/dev/training_saes) and the installed
sae_lens.training.sae_trainer source (SAETrainer(cfg, sae, data_provider, evaluator)).
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from pathlib import Path

import torch
from sae_lens import (
    SAE,
    BatchTopKTrainingSAEConfig,
    JumpReLUTrainingSAEConfig,
    StandardTrainingSAEConfig,
    TopKTrainingSAEConfig,
)
from sae_lens.config import LoggingConfig, SAETrainerConfig
from sae_lens.saes.sae import TrainingSAE
from sae_lens.training import sae_trainer as _trainer_module
from sae_lens.training.sae_trainer import SAETrainer

from modsae.config import Paths, SAEConfig
from modsae.metrics import fvu


def training_sae_config(cfg: SAEConfig, d_in: int, device: str):
    """SAELens training config for `cfg.arch` (field names from the installed sae_lens configs)."""
    common = dict(d_in=d_in, d_sae=cfg.d_sae, device=device, normalize_activations="expected_average_only_in")
    if cfg.arch == "batchtopk":
        return BatchTopKTrainingSAEConfig(k=cfg.k, **common)
    if cfg.arch == "topk":
        return TopKTrainingSAEConfig(k=cfg.k, **common)
    if cfg.arch == "jumprelu":
        return JumpReLUTrainingSAEConfig(l0_coefficient=cfg.l0_coefficient, **common)
    if cfg.arch == "standard":
        return StandardTrainingSAEConfig(l1_coefficient=cfg.l1_coefficient, **common)
    raise ValueError(f"unknown SAE architecture {cfg.arch!r}")


def batch_provider(acts: torch.Tensor, batch_size: int, seed: int) -> Iterator[torch.Tensor]:
    """Infinite iterator of random mini-batches (sampling with replacement) from an in-memory tensor."""
    gen = torch.Generator(device=acts.device).manual_seed(seed)
    while True:
        yield acts[torch.randint(0, acts.shape[0], (batch_size,), device=acts.device, generator=gen)]


class _TrackioShim:
    """Stands in for the `wandb` module inside sae_lens' trainer and forwards scalar logs to trackio."""

    def __init__(self, trackio_module):
        self._tr = trackio_module

    @staticmethod
    def Histogram(values):  # histograms are dropped; scalars are what the dashboard shows
        return None

    def log(self, metrics: dict, step: int | None = None):
        scalars = {}
        for key, val in metrics.items():
            if isinstance(val, (int, float)):
                scalars[key] = float(val)
            elif torch.is_tensor(val) and val.numel() == 1:
                scalars[key] = float(val.item())
        if scalars:
            self._tr.log(scalars, step=step)


def _val_evaluator(acts_val: torch.Tensor):
    """SAETrainer evaluator: SAE-val FVU and L0 in the trainer's scaled activation space."""

    def evaluate(sae, data_provider, activation_scaler) -> dict:
        x = activation_scaler(acts_val)
        z = sae.encode(x)
        return {"val/fvu": fvu(x, sae.decode(z)), "val/l0": (z > 0).float().sum(-1).mean().item()}

    return evaluate


def train_sae(cfg: SAEConfig, acts_train: torch.Tensor, acts_val: torch.Tensor, device: str, out_dir: Path | None = None,
              logger=None, use_trackio: bool = True) -> tuple[SAE, dict]:
    """Train one SAE and save the inference model to `out_dir` (default checkpoints/<run_id>)."""
    out_dir = out_dir or (Paths().make().checkpoints / cfg.run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)  # SAE weight init uses the global RNG
    acts_train, acts_val = acts_train.to(device), acts_val.to(device)
    mu = acts_train.mean(0) if cfg.center else torch.zeros_like(acts_train[0])
    # Centering: SAELens initialises b_dec at 0; training on x − μ and folding μ into b_dec afterwards is exactly an SAE
    # whose decoder bias starts at the data mean (Bricken et al. 2023 and Gao et al. 2024 initialise it at the geometric median).
    acts_train, acts_val = acts_train - mu, acts_val - mu
    sae = TrainingSAE.from_dict(training_sae_config(cfg, acts_train.shape[1], device).to_dict())
    tracker = None
    if use_trackio:
        try:
            import trackio

            trackio.init(project=os.getenv("TRACKIO_PROJECT", "modular-addition-saes"), name=cfg.run_id, config=cfg.to_dict())
            _trainer_module.wandb = _TrackioShim(trackio)  # route sae_lens' wandb.log calls to trackio
            tracker = trackio
        except Exception:
            tracker = None
    tcfg = SAETrainerConfig(
        total_training_samples=cfg.total_samples, train_batch_size_samples=cfg.batch_size, device=device, lr=cfg.lr,
        lr_end=cfg.lr, lr_scheduler_name="constant", lr_warm_up_steps=0, lr_decay_steps=0, n_restart_cycles=1,
        adam_beta1=0.9, adam_beta2=0.999, autocast=False,
        dead_feature_window=1000, feature_sampling_window=2000, n_checkpoints=0, checkpoint_path=None, save_final_checkpoint=False,
        logger=LoggingConfig(log_to_wandb=tracker is not None, wandb_log_frequency=50, eval_every_n_wandb_logs=10),
    )
    t0 = time.time()
    trainer = SAETrainer(cfg=tcfg, sae=sae, data_provider=batch_provider(acts_train, cfg.batch_size, cfg.seed),
                         evaluator=_val_evaluator(acts_val))
    trained = trainer.fit()
    if tracker is not None:
        tracker.finish()
    trained.save_inference_model(str(out_dir))
    if cfg.center:
        _fold_mean_into_b_dec(out_dir, mu.detach().float().cpu())
        acts_train, acts_val = acts_train + mu, acts_val + mu  # evaluate the saved SAE on raw activations
    inference = load_sae(out_dir, device)
    with torch.no_grad():
        info = {"run_id": cfg.run_id, "config": cfg.to_dict(), "train_seconds": time.time() - t0,
                "train_fvu": fvu(acts_train, inference.decode(inference.encode(acts_train))),
                "val_fvu": fvu(acts_val, inference.decode(inference.encode(acts_val)))}
    (out_dir / "train_info.json").write_text(json.dumps(info, indent=2))
    if logger:
        logger.info("%s trained in %.0fs | train FVU %.4f | val FVU %.4f", cfg.run_id, info["train_seconds"], info["train_fvu"], info["val_fvu"])
    return inference, info


def _fold_mean_into_b_dec(out_dir: Path, mu: torch.Tensor) -> None:
    """With apply_b_dec_to_input, encode uses (x − b_dec) and decode adds b_dec, so b_dec' = b_dec + μ makes the SAE
    trained on x − μ act identically on raw x."""
    from safetensors.torch import load_file, save_file

    path = out_dir / "sae_weights.safetensors"
    w = load_file(str(path))
    w["b_dec"] = w["b_dec"] + mu.to(w["b_dec"].dtype)
    save_file(w, str(path))


def load_sae(path: Path, device: str) -> SAE:
    """Load an inference SAE saved by `save_inference_model` (SAELens SAE.load_from_disk)."""
    return SAE.load_from_disk(str(path), device=device)
