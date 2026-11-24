#!/usr/bin/env python3
"""Package a problem as a Harbor task.

    scripts/make_task.py PROBLEM_DIR --out tasks
    harbor run -p tasks/<problem> -a oracle      # expect reward 1.0
    harbor run -p tasks/<problem> -a nop         # expect reward 0.0

PROBLEM_DIR holds `meta.json` (with `model_id_in_registry`) and the starting weights,
`corrupted.safetensors`. The network's checkpoint and data come from the cache
(default `cache/` in this repo): `checkpoints/<model_id>/` and `data/<dataset_id>/{dev,hidden}.pt`.

It builds three images on top of `btw-runtime` (scripts/build_images.sh):

    btw-<problem>-runtime:<tag>    this package, the checkpoint and the starting weights
    btw-<problem>-backend:<tag>    + the dev split, which the solver's forward passes run on
    btw-<problem>-verifier:<tag>   + the dev and hidden splits, for scoring only

then measures the scores inside the verifier image and writes the task folder (src/benchtheweights/task.py).
The solver's image is the shared `btw-agent`. Limits: the flags, else meta.json's recommendations, else
50 forward passes, 3% of the parameters and 25 minutes.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sh(cmd: list[str], log: Path) -> None:
    with log.open("a") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.flush()
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        sys.exit(f"failed: {' '.join(cmd[:4])} ... (see {log}):\n" + "".join(log.read_text().splitlines(True)[-25:]))


def image_exists(name: str) -> bool:
    return subprocess.run(["docker", "image", "inspect", name], capture_output=True).returncode == 0


def registry_entry(model_id: str) -> tuple[str, str]:
    """(dataset_id, version), read without importing torch."""
    sys.path.insert(0, str(ROOT / "src"))
    from benchtheweights import __version__
    from benchtheweights.registry import MODELS

    if model_id not in MODELS:
        sys.exit(f"{model_id} is not registered in src/benchtheweights/registry.py")
    return MODELS[model_id].dataset_id, __version__


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("problem_dir", type=Path)
    ap.add_argument("--id", help="problem id (default: the folder's name)")
    ap.add_argument("--weights", type=Path, help="starting weights (default PROBLEM_DIR/corrupted.safetensors)")
    ap.add_argument("--cache", type=Path, default=Path(os.environ.get("BTW_CACHE", ROOT / "cache")))
    ap.add_argument("--out", type=Path, default=Path("tasks"), help="parent folder of the task (default %(default)s)")
    ap.add_argument("--passes", type=int, help="forward-pass budget")
    ap.add_argument("--edit", type=float, help="edit budget, as a fraction of the parameters")
    ap.add_argument("--minutes", type=float, help="time limit")
    ap.add_argument("--backend-mem", default="6g", help="the backend's memory limit (default %(default)s)")
    ap.add_argument("--verifier-mem", type=int, default=8192, help="the verifier's memory in MB (default %(default)s)")
    ap.add_argument("--tag", help="tag of the problem's images (default: the package version)")
    args = ap.parse_args()

    prob = args.problem_dir.expanduser().resolve()
    pid = args.id or prob.name
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", pid):
        sys.exit("the problem id must be lowercase letters, digits, '.', '_' or '-' (it names Docker images)")
    weights = (args.weights or prob / "corrupted.safetensors").resolve()
    meta = json.loads((prob / "meta.json").read_text())
    model_id = meta["model_id_in_registry"]
    dataset_id, version = registry_entry(model_id)
    cache = args.cache.expanduser().resolve()
    for f in (weights, cache / "checkpoints" / model_id, cache / "data" / dataset_id / "dev.pt",
              cache / "data" / dataset_id / "hidden.pt"):
        if not f.exists():
            sys.exit(f"missing {f}")

    tag = args.tag or version
    prefix = f"btw-{pid}"
    base, agent = f"btw-runtime:{version}", f"btw-agent:{version}"
    for img in (base, agent):
        if not image_exists(img):
            sys.exit(f"missing image {img}: run scripts/build_images.sh")
    passes = args.passes or meta.get("recommended_forward_passes") or 50
    edit = args.edit or meta.get("recommended_edit_fraction") or 0.03
    minutes = args.minutes or meta.get("time_limit_min") or 25

    out_parent = args.out.expanduser().resolve()
    out_parent.mkdir(parents=True, exist_ok=True)
    task_dir = out_parent / pid
    if task_dir.exists():
        sys.exit(f"{task_dir} exists; remove it first")
    log = out_parent / f"{pid}.build.log"
    log.write_text("")

    # the build context holds only this problem's checkpoint, splits and starting weights
    stage = Path(tempfile.mkdtemp(prefix=f"btw-{pid}-"))
    try:
        shutil.copy2(ROOT / "pyproject.toml", stage / "pyproject.toml")
        shutil.copytree(ROOT / "src", stage / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"))
        shutil.copytree(cache / "checkpoints" / model_id, stage / "checkpoints" / model_id)
        shutil.copytree(cache / "data" / dataset_id, stage / "data" / dataset_id)
        (stage / "data-dev" / dataset_id).mkdir(parents=True)
        shutil.copy2(cache / "data" / dataset_id / "dev.pt", stage / "data-dev" / dataset_id / "dev.pt")
        (stage / "start").mkdir()
        shutil.copy2(weights, stage / "start" / "weights.safetensors")
        (stage / "runtime.Dockerfile").write_text(f"""FROM {base}
COPY checkpoints/ /opt/btw/cache/checkpoints/
COPY pyproject.toml /opt/btw/package/pyproject.toml
COPY src /opt/btw/package/src
RUN pip install --no-deps /opt/btw/package && rm -rf /opt/btw/package
COPY start/weights.safetensors /opt/btw/start/weights.safetensors
""")
        (stage / "backend.Dockerfile").write_text(f"""FROM {prefix}-runtime:{tag}
COPY data-dev/ /opt/btw/cache/data/
RUN mkdir -p /srv/export
EXPOSE 8000
""")
        (stage / "verifier.Dockerfile").write_text(f"""FROM {prefix}-runtime:{tag}
COPY data/ /opt/btw/cache/data/
RUN mkdir -p /tests /logs/verifier
""")
        for kind in ("runtime", "backend", "verifier"):
            print(f"building {prefix}-{kind}:{tag} ...", flush=True)
            sh(["docker", "build", "-f", str(stage / f"{kind}.Dockerfile"), "-t", f"{prefix}-{kind}:{tag}", str(stage)], log)
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    print("measuring the scores in the verifier image and writing the task ...", flush=True)
    sh(["docker", "run", "--rm", "--network", "none", "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp",
        "-v", f"{prob / 'meta.json'}:/meta.json:ro", "-v", f"{out_parent}:/out", f"{prefix}-verifier:{tag}",
        "python", "-m", "benchtheweights.task", "--meta", "/meta.json", "--out", f"/out/{pid}",
        "--problem-id", pid, "--image-prefix", prefix, "--tag", tag, "--agent-image", agent,
        "--passes", str(passes), "--edit", str(edit), "--minutes", str(minutes),
        "--backend-mem", args.backend_mem, "--verifier-mem", str(args.verifier_mem)], log)
    print("".join(log.read_text().splitlines(True)[-14:]))
    print(f"task: {task_dir}\n  harbor run -p {task_dir} -a oracle    # expect reward 1.0\n"
          f"  then convert it for a harness: harnesses/{{claude_code,codex,antigravity}}/make_task.py")


if __name__ == "__main__":
    main()
