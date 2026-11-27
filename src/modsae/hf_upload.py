"""Upload base models and SAEs to one Hugging Face model repo and add it to a collection.

Local: uses huggingface_hub (create_repo, upload_folder, create_collection, add_collection_item;
https://huggingface.co/docs/huggingface_hub). Authentication comes from the local `hf auth login`
token; no secret is read from or written to the repository.

Global context: the CLAUDE.md asks for every trained model to be pushed to Hugging Face in one
collection for this research. Layout in the repo:
  base/<run>/model.pt (+ checkpoints.pt), saes/<run_id>/{cfg.json, sae_weights.safetensors, metrics.json, characterization.json}
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

from modsae.config import ROOT, Paths

COLLECTION_TITLE = "SAEs on a grokked modular-addition transformer"


def upload(base_runs: list[str], sae_runs: list[str], include_checkpoints: bool = True, message: str = "Upload models") -> str:
    repo_id = os.getenv("HF_REPO_ID", "BurnyCoder/modular-addition-saes")
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    paths = Paths()
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        card = ROOT / "docs" / "hf_model_card.md"
        if card.exists():
            shutil.copy(card, stage / "README.md")
        for run in base_runs:
            src = paths.checkpoints / run
            dst = stage / "base" / run
            dst.mkdir(parents=True)
            shutil.copy(src / "model.pt", dst / "model.pt")
            if include_checkpoints and (src / "checkpoints.pt").exists():
                shutil.copy(src / "checkpoints.pt", dst / "checkpoints.pt")
            for name in ("training.json", "mechanism.json"):
                if (paths.results / run / name).exists():
                    shutil.copy(paths.results / run / name, dst / name)
        for run_id in sae_runs:
            dst = stage / "saes" / run_id
            dst.mkdir(parents=True)
            for name in ("cfg.json", "sae_weights.safetensors"):
                shutil.copy(paths.checkpoints / run_id / name, dst / name)
            for name in ("metrics.json", "characterization.json", "steering.json"):
                if (paths.results / run_id / name).exists():
                    shutil.copy(paths.results / run_id / name, dst / name)
        api.upload_folder(repo_id=repo_id, folder_path=str(stage), repo_type="model", commit_message=message)
    _ensure_collection(api, repo_id)
    return f"https://huggingface.co/{repo_id}"


def _ensure_collection(api: HfApi, repo_id: str) -> None:
    owner = repo_id.split("/")[0]
    existing = [c for c in api.list_collections(owner=owner, limit=100) if c.title == COLLECTION_TITLE]
    collection = existing[0] if existing else api.create_collection(
        title=COLLECTION_TITLE, namespace=owner,
        description="Base models and sparse autoencoders for the grokked modular-addition transformer (Nanda et al. 2023).")
    for item, kind in ((repo_id, "model"), ("BurnyCoder/grokking-modular-addition-transformer", "model")):
        try:
            api.add_collection_item(collection.slug, item_id=item, item_type=kind, exists_ok=True)
        except Exception:
            pass
