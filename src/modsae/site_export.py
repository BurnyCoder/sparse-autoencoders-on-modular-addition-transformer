"""Export model weights, SAE weights, feature metadata and result summaries for the GitHub Pages site.

Local: writes site/data/<name>.json. Float arrays are stored as base64-encoded little-endian
float32 with their shape, so the browser can rebuild Float32Arrays and run the exact model and
SAE forward pass client-side (site/js/model.js; equality with Python is checked by
tests/test_site_export.py through Node).

Global context: the interactive explorer (feature browser, steering playground) computes every
number it shows from these weights, so it cannot drift from the Python results.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import torch

from modsae.config import ROOT, Paths

SITE_DATA = ROOT / "site" / "data"


def pack(t: torch.Tensor) -> dict:
    arr = t.detach().float().cpu().contiguous().numpy().astype("<f4")
    return {"shape": list(arr.shape), "b64": base64.b64encode(arr.tobytes()).decode("ascii")}


def export_model(run: str, out_dir: Path = SITE_DATA) -> Path:
    """Raw nn.Linear-layout state dict of a base model (weights are (out, in))."""
    blob = torch.load(Paths().checkpoints / run / "model.pt", map_location="cpu", weights_only=True)
    sd = blob["state_dict"]
    keys = ["embed.weight", "pos_embed.weight", "blocks.0.attn.q.weight", "blocks.0.attn.q.bias", "blocks.0.attn.k.weight",
            "blocks.0.attn.k.bias", "blocks.0.attn.v.weight", "blocks.0.attn.v.bias", "blocks.0.attn.o.weight", "blocks.0.attn.o.bias",
            "blocks.0.mlp.in.weight", "blocks.0.mlp.in.bias", "blocks.0.mlp.out.weight", "blocks.0.mlp.out.bias",
            "unembed.weight", "unembed.bias"]
    out = {"run": run, "n_heads": blob["config"]["n_heads"], "d_head": blob["config"]["d_head"], "p": blob["config"]["p"],
           "weights": {k: pack(sd[k]) for k in keys}}
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"model_{run}.json"
    path.write_text(json.dumps(out))
    return path


def export_sae(run_id: str, out_dir: Path = SITE_DATA) -> Path:
    """Inference SAE (JumpReLU: z = ReLU(h)·[h > θ], h = (x − b_dec) W_enc + b_enc; x̂ = z W_dec + b_dec) + feature table."""
    from safetensors.torch import load_file

    ck = Paths().checkpoints / run_id
    w = load_file(str(ck / "sae_weights.safetensors"))
    res = Paths().results / run_id
    meta = {"run_id": run_id, "config": json.loads((res / "metrics.json").read_text())["config"],
            "weights": {k: pack(v) for k, v in w.items()}}
    fpath = res / "features.pt"
    if fpath.exists():
        feat = torch.load(fpath, weights_only=False)
        t = feat["table"]
        cols = [c for c in ("k_key", "r2_key_best", "gamma", "width", "preferred_angle", "odd_fraction", "torus_r2") if c in t]
        meta["features"] = {"alive_idx": feat["alive_idx"].tolist(), "ring_mask": feat["ring_mask"].tolist(),
                            "ring_label": feat["rings"]["labels"].tolist(), "ring_angle": [None if np.isnan(x) else x for x in feat["rings"]["angle"].tolist()],
                            **{c: [None if (isinstance(x, float) and np.isnan(x)) else x for x in t[c].tolist()] for c in cols}}
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"sae_{run_id}.json"
    path.write_text(json.dumps(meta))
    return path


def export_reference(run: str, n: int = 64, out_dir: Path = SITE_DATA) -> Path:
    """Reference resid_mid / logits for the first n grid inputs (used by the JS equality test)."""
    grid = torch.load(Paths().cache / run / "grid.pt", weights_only=True)
    out = {"idx": list(range(n)), "resid_mid": pack(grid["resid_mid"][:n]), "logits": pack(grid["logits"][:n])}
    path = out_dir / f"reference_{run}.json"
    path.write_text(json.dumps(out))
    return path


def _md_to_html(md: str) -> str:
    """Tiny markdown-table to HTML converter for the generated tables (pipes only)."""
    lines = [l for l in md.strip().splitlines() if l.strip()]
    if len(lines) < 2:
        return ""
    cells = lambda l: [c.strip() for c in l.strip().strip("|").split("|")]
    head = "".join(f"<th>{c}</th>" for c in cells(lines[0]))
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in cells(l)) + "</tr>" for l in lines[2:])
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def build_site(model_run: str, sae_runs: list[str], status_html: str = "") -> Path:
    """Export everything the explorer needs into site/data and copy figures into site/assets."""
    import datetime
    import shutil

    paths = Paths()
    SITE_DATA.mkdir(parents=True, exist_ok=True)
    assets = ROOT / "site" / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for fig in paths.figures.glob("*.png"):
        shutil.copy(fig, assets / fig.name)
    export_model(model_run)
    grid = torch.load(paths.cache / model_run / "grid.pt", weights_only=True)
    saes = []
    for run_id in sae_runs:
        export_sae(run_id)
        cfg = json.loads((paths.results / run_id / "metrics.json").read_text())["config"]
        entry = {"id": run_id, "file": f"sae_{run_id}.json", "site": cfg["site"], "resid_norm": grid["resid_mid"].norm(dim=-1).mean().item()}
        steer = paths.results / run_id / "steering.json"
        if steer.exists() and cfg["site"] == "resid_mid":
            entry["calibration"] = json.loads(steer.read_text()).get("ring_calibration", {})
        saes.append(entry)
    tables_dir = paths.results / "tables"
    tables_html = "".join(f"<h3>{p.stem}</h3>" + _md_to_html(p.read_text(encoding="utf-8")) for p in sorted(tables_dir.glob("*.md")))
    manifest = {"date": datetime.date.today().isoformat(), "model": f"model_{model_run}.json", "saes": saes,
                "tables_html": tables_html, "status_html": status_html}
    path = SITE_DATA / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path
