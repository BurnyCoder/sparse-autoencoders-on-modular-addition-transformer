"""The website's JavaScript forward pass must reproduce the Python model (run through Node)."""

import json
import shutil
import subprocess

import pytest
import torch

from modsae import activations, model, site_export
from modsae.config import BaseModelConfig, ROOT


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_js_forward_matches_python(tmp_path, monkeypatch):
    monkeypatch.setenv("CHECKPOINT_DIR", str(tmp_path / "ck"))
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    cfg = BaseModelConfig()
    m = model.build_model(cfg, "cpu")
    (tmp_path / "ck" / cfg.run_id).mkdir(parents=True)
    model.save_model(m, cfg, tmp_path / "ck" / cfg.run_id / "model.pt")
    grid = activations.run_grid(m, cfg.p)
    (tmp_path / "cache" / cfg.run_id).mkdir(parents=True)
    torch.save(grid, tmp_path / "cache" / cfg.run_id / "grid.pt")
    mp = site_export.export_model(cfg.run_id, tmp_path / "site")
    rp = site_export.export_reference(cfg.run_id, 32, tmp_path / "site")
    out = subprocess.run(["node", str(ROOT / "tests" / "check_site_model.mjs"), str(mp), str(rp)], capture_output=True, text=True, check=True)
    res = json.loads(out.stdout)
    assert res["maxResid"] < 1e-4 and res["maxLogit"] < 1e-4
