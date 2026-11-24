"""The hidden verifier, in Harbor's separate verifier environment (no network).

It rebuilds the network from the pinned checkpoint and the problem's starting weights, checks the
backend's signature on the solver's final edits, applies them, counts the edited elements again and
scores the result on the hidden split.

    python -m benchtheweights.verifier --spec /tests/spec.json --bundle /srv/export --out /logs/verifier

Writes reward.json (Harbor's reward) and metrics.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from . import models as M
from .registry import MODELS
from .submission import apply_delta, read_bundle


def verify(spec: dict, bundle_dir: Path) -> dict:
    t0 = time.time()
    M.configure_torch()
    model_id = spec["model_id"]
    hidden = M.load_split(MODELS[model_id].dataset_id, "hidden")
    model, params = M.load_start(model_id, spec["start_weights"])
    total = sum(p.numel() for p in params.values())
    clean, start = spec["scores"]["hidden_clean"], spec["scores"]["hidden_corrupted"]
    start_now = M.score(model, model_id, hidden)

    status, payload, edited = "ok", {}, 0
    try:
        payload, delta = read_bundle(bundle_dir, spec["hmac_key"])
        if payload.get("task_id") != spec["task_id"]:
            raise ValueError("submission belongs to a different task")
        edited = apply_delta(params, delta)
    except FileNotFoundError:
        status = "missing_submission"  # scored as no repair
    except ValueError as exc:
        status = f"invalid_submission: {exc}"
    lim = spec["limits"]
    edit_fraction = edited / total
    over_budget = edited > int(lim["max_edit_fraction"] * total)
    if status.startswith("invalid") or over_budget:
        model, params = M.load_start(model_id, spec["start_weights"])  # scored as the starting weights
        if over_budget:
            status = "over_edit_budget"
    repaired = M.score(model, model_id, hidden)
    if repaired != repaired:  # NaN, which validated edits cannot produce
        repaired = start

    recovery = (repaired - start) / (clean - start)
    passes = int(payload.get("forward_passes", 0))
    metrics = {
        "reward": round(max(-1.0, min(1.0, recovery)), 6),
        "recovery": round(recovery, 6),
        "relative_to_clean": round(repaired / clean, 6) if clean else None,
        "repaired_score": round(repaired, 6),
        "clean_score": clean,
        "corrupted_score": start,
        "edit_fraction": round(edit_fraction, 8),
        "edited_params": edited,
        "forward_passes": passes,
        "valid_submission": int(status == "ok"),
        "agent_active_sec": float(payload.get("agent_active_sec", 0.0)),
        "verifier_time_sec": round(time.time() - t0, 3),
    }
    details = {
        **metrics,
        "status": status,
        "task_id": spec["task_id"],
        "model_id": model_id,
        "objective": MODELS[model_id].objective,
        "corrupted_score_recomputed": start_now,
        "determinism_ok": abs(start_now - start) < 1e-9,
        "rejected_requests": payload.get("rejected_requests"),
        "history": payload.get("history", []),
    }
    return {"reward": metrics, "details": details}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="/tests/spec.json")
    ap.add_argument("--bundle", default="/srv/export")
    ap.add_argument("--out", default="/logs/verifier")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = verify(json.loads(Path(args.spec).read_text()), Path(args.bundle))
    (out / "reward.json").write_text(json.dumps(res["reward"], indent=2))
    (out / "metrics.json").write_text(json.dumps(res["details"], indent=2))
    print(json.dumps(res["details"], indent=2))


if __name__ == "__main__":
    main()
