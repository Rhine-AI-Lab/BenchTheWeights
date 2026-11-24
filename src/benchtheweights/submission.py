"""The signed bundle of a solver's final edits, written by the backend and read by the verifier.

A bundle is a directory:
    delta.safetensors   the changed elements, relative to the starting weights (`<name>::idx`, `<name>::val`)
    submission.json     the accounting, with an HMAC-SHA256 over it and the delta's sha256
    tool_log.jsonl      every backend request, for the trial's artifacts
"""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def _canonical(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def sign(payload: dict, key: str) -> str:
    return hmac.new(key.encode(), _canonical(payload), hashlib.sha256).hexdigest()


def write_bundle(out_dir: Path, delta: dict[str, torch.Tensor], payload: dict, key: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "delta.safetensors.tmp"
    if delta:
        save_file(delta, str(tmp))
    else:
        save_file({"__empty__": torch.zeros(0)}, str(tmp))
    tmp.replace(out_dir / "delta.safetensors")
    payload = dict(payload, delta_sha256=hashlib.sha256((out_dir / "delta.safetensors").read_bytes()).hexdigest())
    payload["hmac"] = sign(payload, key)
    tmpj = out_dir / "submission.json.tmp"
    tmpj.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmpj.replace(out_dir / "submission.json")


def read_bundle(in_dir: Path, key: str) -> tuple[dict, dict[str, torch.Tensor]]:
    """(payload, delta). FileNotFoundError if there is no bundle, ValueError if it was altered."""
    sj, st = in_dir / "submission.json", in_dir / "delta.safetensors"
    if not sj.exists() or not st.exists():
        raise FileNotFoundError(f"no submission bundle in {in_dir}")
    payload = json.loads(sj.read_text())
    mac = payload.pop("hmac", "")
    if not hmac.compare_digest(mac, sign(payload, key)):
        raise ValueError("submission signature mismatch")
    if hashlib.sha256(st.read_bytes()).hexdigest() != payload.get("delta_sha256"):
        raise ValueError("delta checksum mismatch")
    delta = {k: v for k, v in load_file(str(st)).items() if k != "__empty__"}
    return payload, delta


def apply_delta(params: dict[str, torch.Tensor], delta: dict[str, torch.Tensor]) -> int:
    """Apply a delta in place. Returns the number of elements that changed."""
    changed = 0
    names = {k.rsplit("::", 1)[0] for k in delta}
    for name in names:
        if name not in params:
            raise ValueError(f"delta references unknown tensor {name}")
        idx, val = delta[f"{name}::idx"], delta[f"{name}::val"]
        t = params[name].data
        if idx.numel() and (idx.min() < 0 or idx.max() >= t.numel()):
            raise ValueError(f"delta indices out of range for {name}")
        flat = t.view(-1)
        before = flat[idx].clone()
        flat[idx] = val.to(t.dtype)
        changed += int((flat[idx] != before).sum())
    return changed
