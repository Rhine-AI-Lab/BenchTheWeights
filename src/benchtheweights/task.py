"""Write a problem's Harbor task. Runs inside the problem's verifier image (`scripts/make_task.py` calls it).

It measures the clean and starting scores on both splits, draws the task's secrets (the HMAC key that
signs the solver's edits, and the oracle's token) and writes the task folder:

    task.toml                        limits, separate verifier, collect hook, artifacts
    instruction.md                   the solver's prompt (prompts/solver.md, filled in)
    environment/Dockerfile           the solver's image: the `py` kernel and the `forward` client only
    environment/docker-compose.yaml  the backend on an internal network, no internet
    environment/backend/             the backend's image + episode.json        (never visible to the solver)
    tests/                           the verifier's image + spec.json, test.sh (never visible to the solver)
    solution/solve.sh                the oracle: restores the clean weights through a token-gated command

The task folder holds secrets: keep it out of git and away from the solver.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import stat
import sys
from pathlib import Path

from . import models as M
from .instruction import fmt_score, render_instruction
from .registry import MODELS, SCORE_INFO

START = "/opt/btw/start/weights.safetensors"

# Wall time Harbor allows per forward pass on top of the time limit, since the clock pauses during passes.
FORWARD_ALLOWANCE_SEC = 120

TASK_TOML = """schema_version = "1.3"

[task]
name = "{task_id}"
description = "Improve a {arch} by editing its weights directly (problem {pid}); scored by the share of the {metric} gap closed on a hidden test set."
authors = [{{ name = "Bench The Weights - Repair" }}]
keywords = ["weight-space", "model-editing", "pytorch", "{family}", "separate-verifier", "no-internet"]

[metadata]
category = "machine-learning"
tags = ["benchtheweights", "{pid}", "{model_id}"]
btw_problem = "{pid}"
btw_model = "{model_id}"
btw_family = "{family}"
btw_objective = "{objective}"
btw_degradation = {degradation:.4f}
btw_image_tag = "{tag}"

[agent]
# The time limit ({minutes:g} min), which the backend enforces from the solver's first request, plus
# {passes} x {fwd_allow} s for forward passes, during which the backend's clock stops.
timeout_sec = {timeout:.1f}

[verifier]
timeout_sec = 1800.0
environment_mode = "separate"

# Freeze the backend's state into /srv/export once the solver has stopped.
[[verifier.collect]]
command = "python -m benchtheweights.backend.finalize /srv/episode.json"
service = "backend"
timeout_sec = 120.0

[verifier.environment]
network_mode = "no-network"
build_timeout_sec = 1200.0
cpus = 4
memory_mb = {verifier_mem}

[environment]
# `main` is attached only to the internal network it shares with the backend (environment/docker-compose.yaml),
# so it has no internet. The harness adapters in harnesses/ add an egress proxy to the model API.
network_mode = "public"
build_timeout_sec = 1200.0
cpus = 2
memory_mb = 8192
storage_mb = 10240

[environment.healthcheck]
command = "python3 -c \\"import os, urllib.request; urllib.request.urlopen(os.environ['WSP_BACKEND'] + '/health', timeout=5)\\""
interval_sec = 3.0
timeout_sec = 30.0
start_period_sec = 600.0
start_interval_sec = 3.0
retries = 3

# The signed bundle of the final edits and the backend's request log.
[[artifacts]]
source = "/srv/export"
service = "backend"
"""

COMPOSE = """services:
  main:
    environment:
      WSP_BACKEND: "http://backend:8000"
    depends_on:
      backend:
        condition: service_healthy
    # only the internal network: the solver reaches the backend and nothing else
    networks: [repair]
  backend:
    build:
      context: ./backend
    command: ["python", "-m", "benchtheweights.backend.server", "--episode", "/srv/episode.json", "--port", "8000"]
    environment:
      BTW_THREADS: "4"
    cpus: 4
    mem_limit: {mem}
    mem_reservation: 2g
    networks: [repair]
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]
      interval: 2s
      timeout: 5s
      retries: 300
      start_period: 5s

networks:
  repair:
    internal: true
"""

TEST_SH = """#!/bin/bash
# Score the backend-signed final edits on the hidden split.
set -u
mkdir -p /logs/verifier
python -m benchtheweights.verifier --spec /tests/spec.json --bundle /srv/export --out /logs/verifier \\
  > /logs/verifier/verify.log 2>&1
status=$?
cat /logs/verifier/verify.log
if [ ! -f /logs/verifier/reward.json ]; then
  echo "verifier failed (exit $status)" >&2
  echo '{"reward": 0.0, "recovery": 0.0, "valid_submission": 0}' > /logs/verifier/reward.json
fi
exit 0
"""

SOLVE_SH = """#!/bin/bash
# Oracle: restore the clean value of every parameter that differs from the starting weights, through a
# token-gated backend command. Harbor uploads this file only for its oracle agent.
set -e
python3 - <<'PY'
import wsp
print(wsp._json("oracle-restore", {{"token": "{token}"}}))
print(wsp._forward({{"type": "evaluate"}})["edited"]["score"])
PY
"""

README = """# {task_id}

Problem **{pid}**: a modified **{arch}** ({task}). The solver is not told that the network was modified;
see [instruction.md](instruction.md).

- **Hidden {metric}:** {clean} for the clean network, {start} for the starting weights.
- **Limits:** {passes} forward passes, {edit_pct:g}% of the parameters, {minutes:g} minutes.
- **Images:** `{prefix}-backend:{tag}` (dev split and starting weights), `{prefix}-verifier:{tag}` (both splits),
  solver `{agent_image}`.

```
harbor run -p <this dir> -a oracle    # expect reward 1.0
harbor run -p <this dir> -a nop       # expect reward 0.0
```
"""


def write_task(a) -> None:
    meta = json.loads(Path(a.meta).read_text())
    model_id = meta["model_id_in_registry"]
    spec = MODELS[model_id]
    limits = {"max_forward_passes": a.passes, "max_edit_fraction": a.edit, "time_limit_sec": round(60 * a.minutes, 3)}
    M.configure_torch()
    splits = {s: M.load_split(spec.dataset_id, s) for s in ("dev", "hidden")}
    clean_model = M.build_model(model_id)
    start_model, start = M.load_start(model_id, START)
    clean_p = M.editable_parameters(clean_model)
    changed = sum(int((start[n] != clean_p[n]).sum()) for n in start)
    total = sum(p.numel() for p in start.values())
    scores = {f"{s}_clean": M.score(clean_model, model_id, d) for s, d in splits.items()}
    scores.update({f"{s}_corrupted": M.score(start_model, model_id, d) for s, d in splits.items()})
    floor = M.floor_score(model_id, splits["hidden"])
    scores["hidden_chance"] = floor
    scores["degradation"] = (scores["hidden_clean"] - scores["hidden_corrupted"]) / max(scores["hidden_clean"] - floor, 1e-9)
    for key, s in (("clean_score", "hidden_clean"), ("corrupted_score", "hidden_corrupted")):
        ref = (meta.get(key) or {}).get("hidden")
        if ref is not None and abs(ref - scores[s]) > 1e-3:
            print(f"warning: {s} = {scores[s]:.5f} in the image, {ref:.5f} in meta.json", file=sys.stderr)
    if scores["hidden_clean"] - scores["hidden_corrupted"] <= 0:
        raise SystemExit("the starting weights do not score below the clean network on the hidden split")
    if changed / total > a.edit:
        print(f"warning: the starting weights differ from the clean ones in {changed / total:.4%} of the parameters, "
              f"more than the edit budget ({a.edit:.2%}): an exact repair would be rejected", file=sys.stderr)

    hmac_key, token = secrets.token_hex(32), secrets.token_hex(16)
    pid, task_id = a.problem_id, f"benchtheweights/{a.problem_id}"
    metric, metric_desc = SCORE_INFO[spec.objective]
    episode = {"task_id": task_id, "model_id": model_id, "start_weights": START, "limits": limits,
               "hmac_key": hmac_key, "oracle_token_sha256": hashlib.sha256(token.encode()).hexdigest(),
               "public_scores": {"metric": metric, "metric_definition": metric_desc,
                                 "reference_network_test_score": round(scores["hidden_clean"], 4),
                                 "this_network_test_score": round(scores["hidden_corrupted"], 4),
                                 "split": "hidden test set (scoring only); forward passes use a separate dev split"}}
    vspec = {"task_id": task_id, "model_id": model_id, "start_weights": START, "limits": limits, "hmac_key": hmac_key,
             "scores": scores, "changed_params": changed, "changed_fraction": changed / total,
             "problem": pid, "image_tag": a.tag}
    d = Path(a.out)
    for sub in ("environment/backend", "tests", "solution"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    fmt = dict(task_id=task_id, pid=pid, arch=spec.architecture, task=spec.task, metric=metric, family=spec.family,
               model_id=model_id, objective=spec.objective, degradation=scores["degradation"], tag=a.tag,
               minutes=a.minutes, timeout=60 * a.minutes + a.passes * FORWARD_ALLOWANCE_SEC,
               fwd_allow=FORWARD_ALLOWANCE_SEC, verifier_mem=a.verifier_mem,
               clean=fmt_score(spec.objective, scores["hidden_clean"]),
               start=fmt_score(spec.objective, scores["hidden_corrupted"]), passes=a.passes,
               edit_pct=100 * a.edit, prefix=a.image_prefix, agent_image=a.agent_image)
    files = {
        "task.toml": TASK_TOML.format(**fmt),
        "README.md": README.format(**fmt),
        "instruction.md": render_instruction(model_id, scores["hidden_clean"], scores["hidden_corrupted"], limits),
        "environment/Dockerfile": "# The solver's workspace: the `py` kernel and the `forward` client. No network weights, "
                                  f"data, torch or internet.\nFROM {a.agent_image}\nWORKDIR /work\n",
        "environment/docker-compose.yaml": COMPOSE.format(mem=a.backend_mem),
        "environment/backend/Dockerfile": "# The backend: the problem's network, dev split and starting weights, and this episode.\n"
                                          f"FROM {a.image_prefix}-backend:{a.tag}\nCOPY episode.json /srv/episode.json\n",
        "environment/backend/episode.json": json.dumps(episode, indent=2),
        "tests/Dockerfile": "# The verifier: the problem's network, both splits and starting weights, and this task's spec.\n"
                            f"FROM {a.image_prefix}-verifier:{a.tag}\nCOPY test.sh spec.json /tests/\nRUN chmod +x /tests/test.sh\n",
        "tests/test.sh": TEST_SH,
        "tests/spec.json": json.dumps(vspec, indent=2),
        "solution/solve.sh": SOLVE_SH.format(token=token),
    }
    for name, text in files.items():
        (d / name).write_text(text)
    for x in ("tests/test.sh", "solution/solve.sh"):
        p = d / x
        p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(json.dumps({"task": str(d), "hidden_clean": scores["hidden_clean"], "hidden_start": scores["hidden_corrupted"],
                      "dev_clean": scores["dev_clean"], "dev_start": scores["dev_corrupted"],
                      "changed_fraction": round(changed / total, 6), "limits": limits}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--problem-id", required=True)
    ap.add_argument("--image-prefix", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--agent-image", required=True)
    ap.add_argument("--passes", type=int, required=True)
    ap.add_argument("--edit", type=float, required=True)
    ap.add_argument("--minutes", type=float, required=True)
    ap.add_argument("--backend-mem", default="6g")
    ap.add_argument("--verifier-mem", type=int, default=8192, help="MB")
    write_task(ap.parse_args())


if __name__ == "__main__":
    main()
