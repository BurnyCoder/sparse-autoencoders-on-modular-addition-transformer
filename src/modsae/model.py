"""Build, train, save and load the 1-layer modular-addition transformer.

Local: wraps TransformerLens 4 `TransformerBridge.boot_native` (the construction the
current TransformerLens Grokking demo uses) with the Nanda et al. 2023 hyperparameters.
Training runs on the bridge's underlying `NativeModel` (same parameters, about 7x
faster per step than going through the hooked bridge, measured on this machine);
analysis and interventions use the bridge's hooks.

Global context: this produces the base models that every SAE is trained on.
Sources: Nanda et al. 2023 (https://arxiv.org/abs/2301.05217); TransformerLens demo
(https://github.com/TransformerLensOrg/TransformerLens/blob/main/demos/Grokking_Demo.ipynb).
"""

from __future__ import annotations

import io
import pickle
import time
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.utilities.tl_checkpoint_conversion import convert_tl_checkpoint

from modsae import data
from modsae.config import BaseModelConfig, Paths


def bridge_config(cfg: BaseModelConfig, device: str) -> TransformerBridgeConfig:
    """TransformerBridgeConfig matching the demo's model (no LayerNorm, ReLU, 3 positions)."""
    return TransformerBridgeConfig(
        n_layers=1, n_heads=cfg.n_heads, d_model=cfg.d_model, d_head=cfg.d_head, d_mlp=cfg.d_mlp,
        act_fn="relu", normalization_type=None, d_vocab=cfg.p + 1, d_vocab_out=cfg.p, n_ctx=3,
        init_weights=True, device=device, seed=cfg.seed,
    )


def build_model(cfg: BaseModelConfig, device: str) -> TransformerBridge:
    """Fresh model with biases frozen at zero (demo) and optional constant attention."""
    model = TransformerBridge.boot_native(bridge_config(cfg, device), device=device)
    for name, param in model.named_parameters():
        if name.endswith("bias"):  # demo: biases frozen (they are initialised to 0)
            param.requires_grad = False
        if cfg.constant_attention and (".attn." in name) and ("._original_component.q." in name or "._original_component.k." in name):
            with torch.no_grad():
                param.zero_()  # q = k = 0 -> all attention scores 0 -> uniform causal attention
            param.requires_grad = False
    return model


def loss_fn(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Demo loss: float64 log-softmax cross-entropy at the last ('=') position."""
    if logits.dim() == 3:
        logits = logits[:, -1]
    return F.cross_entropy(logits.to(torch.float64), labels)


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    if logits.dim() == 3:
        logits = logits[:, -1]
    return (logits.argmax(-1) == labels).float().mean().item()


def train(cfg: BaseModelConfig, device: str, logger=None, use_trackio: bool = True) -> tuple[TransformerBridge, dict]:
    """Full-batch AdamW training (demo recipe); returns model and a history dict.

    The history holds train/test loss and accuracy every `log_every` epochs and the
    path of the checkpoint file (one state dict every `checkpoint_every` epochs).
    """
    paths = Paths().make()
    torch.manual_seed(cfg.seed)
    model = build_model(cfg, device)
    tokens, labels = data.make_dataset(cfg.p)
    train_idx, test_idx = data.base_split(cfg.p, cfg.frac_train, cfg.data_seed)
    tokens, labels = tokens.to(device), labels.to(device)
    x_tr, y_tr, x_te, y_te = tokens[train_idx], labels[train_idx], tokens[test_idx], labels[test_idx]
    native = model.original_model  # shares parameters with the bridge; no hook overhead
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas)
    hist = {"epoch": [], "train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}
    ckpts, ckpt_epochs = [], []
    tracker = _start_tracker(cfg, use_trackio)
    t0 = time.time()
    for epoch in range(cfg.num_epochs):
        loss = loss_fn(native(x_tr), y_tr)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (epoch + 1) % cfg.log_every == 0 or epoch == 0:
            with torch.no_grad():
                lte = native(x_te)
                ltr = native(x_tr)
                row = dict(epoch=epoch + 1, train_loss=loss_fn(ltr, y_tr).item(), test_loss=loss_fn(lte, y_te).item(),
                           train_acc=accuracy(ltr, y_tr), test_acc=accuracy(lte, y_te))
            for key, val in row.items():
                hist[key].append(val)
            if tracker is not None:
                tracker.log({k: v for k, v in row.items() if k != "epoch"}, step=row["epoch"])
            if logger and ((epoch + 1) % (cfg.log_every * 20) == 0 or epoch == 0):
                logger.info("%s epoch %d | train loss %.3e acc %.4f | test loss %.3e acc %.4f | %.0fs", cfg.run_id,
                            row["epoch"], row["train_loss"], row["train_acc"], row["test_loss"], row["test_acc"], time.time() - t0)
        if (epoch + 1) % cfg.checkpoint_every == 0:
            ckpts.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            ckpt_epochs.append(epoch + 1)
    if tracker is not None:
        tracker.finish()
    out_dir = paths.checkpoints / cfg.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"checkpoints": ckpts, "epochs": ckpt_epochs}, out_dir / "checkpoints.pt")
    save_model(model, cfg, out_dir / "model.pt", history=hist)
    hist["train_seconds"] = time.time() - t0
    return model, hist


def _start_tracker(cfg: BaseModelConfig, use_trackio: bool):
    """Start a trackio run (local dashboard: `uv run trackio show`); None if disabled/unavailable."""
    if not use_trackio:
        return None
    try:
        import os

        import trackio

        trackio.init(project=os.getenv("TRACKIO_PROJECT", "modular-addition-saes"), name=cfg.run_id, config=cfg.to_dict())
        return trackio
    except Exception:  # dashboard is a convenience; never fail training because of it
        return None


def save_model(model: TransformerBridge, cfg: BaseModelConfig, path: Path, history: dict | None = None) -> None:
    torch.save({"config": cfg.to_dict(), "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "history": history}, path)


def load_model(path: Path, device: str) -> tuple[TransformerBridge, BaseModelConfig, dict | None]:
    """Load a model saved by `save_model` (weights_only: our files hold only tensors/primitives)."""
    blob = torch.load(path, map_location="cpu", weights_only=True)
    raw = blob["config"]
    raw["betas"] = tuple(raw["betas"])
    cfg = BaseModelConfig(**raw)
    model = build_model(cfg, device)
    model.load_state_dict(blob["state_dict"])
    return model, cfg, blob.get("history")


def load_state(model: TransformerBridge, state_dict: dict) -> TransformerBridge:
    """Load a checkpoint state dict (e.g. the memorization checkpoint) into a built model."""
    model.load_state_dict(state_dict)
    return model


class _TLConfigStub:
    """Stands in for TransformerLens 3 HookedTransformerConfig when unpickling legacy files."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)


class _RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that only allows torch/numpy/collections/builtins and stubs TransformerLens classes."""

    _ALLOWED = ("torch", "numpy", "collections", "builtins", "_codecs", "copyreg")

    def find_class(self, module, name):
        if module.startswith("transformer_lens"):
            return _TLConfigStub
        if module.split(".")[0] in self._ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked global {module}.{name}")


_restricted_pickle = types.SimpleNamespace(Unpickler=_RestrictedUnpickler, load=lambda f, **kw: _RestrictedUnpickler(f, **kw).load(),
                                           loads=lambda b, **kw: _RestrictedUnpickler(io.BytesIO(b), **kw).load(), __name__="restricted_pickle")


def load_hf_legacy_checkpoint(repo_id: str, filename: str, device: str) -> tuple[TransformerBridge, dict]:
    """Load the user's TransformerLens-3 grokking checkpoint from the HF Hub into a TL4 bridge.

    Local: downloads `filename`, unpickles it with a restricted unpickler (the file
    stores a TL3 HookedTransformerConfig object), converts the HookedTransformer
    state dict with TransformerLens' own `convert_tl_checkpoint`, and returns the
    bridge plus the raw blob (losses, split indices).
    Global context: used only to cross-check our reproduction (preregistration §2).
    """
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(repo_id, filename, cache_dir=str(Paths().make().cache / "hf"))
    blob = torch.load(local, map_location="cpu", weights_only=False, pickle_module=_restricted_pickle)
    cfg = BaseModelConfig(num_epochs=len(blob.get("checkpoints", [])) * 100 or 25_000)
    bcfg = bridge_config(cfg, device)
    converted = convert_tl_checkpoint(blob["model"], bcfg)
    model = TransformerBridge.boot_native(bcfg, device=device)
    own = model.state_dict()
    for key, val in own.items():  # legacy files omit some all-zero biases; mirror TL's zero-fill
        converted.setdefault(key, torch.zeros_like(val))
    model.load_state_dict(converted, strict=True)
    return model, blob
