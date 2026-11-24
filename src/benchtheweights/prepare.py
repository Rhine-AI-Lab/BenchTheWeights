"""Download the pinned checkpoints, build the dev and hidden splits, and check the clean scores.

    python -m benchtheweights.prepare [--models mlp-mnist ...] [--cache DIR]

Writes, under the cache (BTW_CACHE, default ./cache):
  checkpoints/<model_id>/            the weights and the loader's extra files (registry.HF_EXTRA_FILES)
  data/<dataset_id>/{dev,hidden}.pt  the splits, as dicts of tensors
  validation.json                    file hashes and the clean scores on both splits
A network whose clean score on either split is below MIN_CLEAN_SCORE is rejected.

A dataset is built by the preparer of its modality in PREPARERS. A preparer gets the DatasetSpec and the parquet
table, and returns the dev and hidden dicts, each with `labels`, `source_index` and the inputs that
models.model_inputs and models.targets read.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
from typing import Callable

import numpy as np
import torch

MIN_CLEAN_SCORE = {
    "cifar10": 0.90,
    "cifar100": 0.70,
    "sst2": 0.85,
    "mnist": 0.95,
    "cifar10-denoise": 0.85,
    "cifar10-recon": 0.80,
}


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_checkpoint(model_id: str) -> dict:
    from huggingface_hub import hf_hub_download

    from .models import checkpoint_dir
    from .registry import HF_EXTRA_FILES, MODELS

    spec = MODELS[model_id]
    out = checkpoint_dir(model_id)
    out.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for fname in [spec.weights_file, *HF_EXTRA_FILES[spec.loader]]:
        shutil.copyfile(hf_hub_download(spec.hf_repo, fname, revision=spec.revision), out / fname)
        hashes[fname] = sha256(out / fname)
    return hashes


def random_split(spec, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(spec.split_seed))
    return perm[: spec.n_dev], perm[spec.n_dev : spec.n_dev + spec.n_hidden]


def subsets(spec, labels: torch.Tensor, inputs: dict) -> dict[str, dict]:
    def subset(idx):
        d = {"labels": labels[idx], "source_index": idx}
        for k, v in inputs.items():
            d[k] = [v[i] for i in idx.tolist()] if isinstance(v, list) else v[idx].contiguous()
        return d

    dev, hidden = random_split(spec, labels.numel())
    return {"dev": subset(dev), "hidden": subset(hidden)}


def prepare_images(spec, table: dict) -> dict[str, dict]:
    """image, mnist, denoise and recon: uint8 `images` [N, H, W, 3] ([N, 28, 28] for mnist). denoise adds one
    fixed (t, eps) per example and the noisy `x_t`, so the backend and the verifier see the same inputs."""
    from PIL import Image

    labels = torch.tensor(table[spec.label_column], dtype=torch.long)
    mode = "L" if spec.modality == "mnist" else "RGB"
    imgs = np.stack([np.asarray(Image.open(io.BytesIO(d["bytes"])).convert(mode)) for d in table[spec.input_column]])
    inputs = {"images": torch.from_numpy(imgs)}
    if spec.modality == "denoise":
        e = spec.extra
        g = torch.Generator().manual_seed(e["noise_seed"])
        x0 = inputs["images"].permute(0, 3, 1, 2).float().div(255.0) * 2 - 1
        t = torch.randint(0, e["timesteps"], (labels.numel(),), generator=g)
        eps = torch.randn(x0.shape, generator=g)
        betas = torch.linspace(e["beta_start"], e["beta_end"], e["timesteps"], dtype=torch.float64)
        ab = torch.cumprod(1 - betas, 0)[t].float().view(-1, 1, 1, 1)
        inputs = {"x_t": ab.sqrt() * x0 + (1 - ab).sqrt() * eps, "t": t, "eps": eps, "images": inputs["images"]}
    return subsets(spec, labels, inputs)


def prepare_text(spec, table: dict) -> dict[str, dict]:
    """text: `input_ids` and `attention_mask` padded to the model's max_length, with the raw `text`."""
    from transformers import AutoTokenizer

    from .models import checkpoint_dir
    from .registry import MODELS

    labels = torch.tensor(table[spec.label_column], dtype=torch.long)
    model_id = next(m for m, s in MODELS.items() if s.dataset_id == spec.dataset_id)
    tok = AutoTokenizer.from_pretrained(checkpoint_dir(model_id))
    enc = tok(table[spec.input_column], padding="max_length", truncation=True,
              max_length=MODELS[model_id].extra["max_length"], return_tensors="pt")
    inputs = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "text": table[spec.input_column]}
    return subsets(spec, labels, inputs)


PREPARERS: dict[str, Callable[..., dict[str, dict]]] = {
    "image": prepare_images,
    "mnist": prepare_images,
    "denoise": prepare_images,
    "recon": prepare_images,
    "text": prepare_text,
}


def prepare_dataset(dataset_id: str) -> dict:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    from .models import data_path
    from .registry import DATASETS

    spec = DATASETS[dataset_id]
    if spec.modality not in PREPARERS:
        raise SystemExit(f"{dataset_id}: no preparer for modality {spec.modality!r} in prepare.PREPARERS")
    src = hf_hub_download(spec.hf_repo, spec.parquet_file, revision=spec.revision, repo_type="dataset")
    splits = PREPARERS[spec.modality](spec, pq.read_table(src).to_pydict())
    out = {}
    for split, d in splits.items():
        path = data_path(dataset_id, split)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(d, path)
        out[split] = {"n": int(d["labels"].numel()), "sha256": sha256(path)}
    return out


def main() -> None:
    from .registry import MODELS

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=list(MODELS), help="model ids (default: all registered)")
    ap.add_argument("--cache", help="cache folder (default: BTW_CACHE, or ./cache)")
    a = ap.parse_args()
    if a.cache:
        os.environ["BTW_CACHE"] = os.path.abspath(a.cache)
    from .models import build_model, cache_root, configure_torch, load_split, score

    configure_torch()
    report = {"models": {}, "datasets": {}}
    for model_id in a.models:
        report["models"][model_id] = {"files": download_checkpoint(model_id)}
    for dataset_id in sorted({MODELS[m].dataset_id for m in a.models}):
        report["datasets"][dataset_id] = prepare_dataset(dataset_id)
    prev = cache_root() / "validation.json"
    if prev.exists():  # keep the entries of the networks not prepared in this run
        old = json.loads(prev.read_text())
        for section in ("models", "datasets"):
            for k, v in old.get(section, {}).items():
                report[section].setdefault(k, v)

    for model_id in a.models:
        ds = MODELS[model_id].dataset_id
        model = build_model(model_id)
        scores = {split: score(model, model_id, load_split(ds, split)) for split in ("dev", "hidden")}
        floor = MIN_CLEAN_SCORE.get(ds)
        ok = floor is None or min(scores.values()) >= floor
        report["models"][model_id].update(clean_score=scores, objective=MODELS[model_id].objective, valid=ok)
        print(f"{model_id}: dev={scores['dev']:.4f} hidden={scores['hidden']:.4f}"
              + ("" if floor is None else f" (min {floor})"), flush=True)
        if not ok:
            raise SystemExit(f"{model_id}: clean score below {floor}")
    (cache_root() / "validation.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
