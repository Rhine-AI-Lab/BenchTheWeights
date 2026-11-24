#!/usr/bin/env python3
"""Convert base tasks (scripts/make_task.py) into Antigravity (agy) tasks, for Gemini models.

    harnesses/antigravity/make_task.py tasks/hb-004 ... --out tasks-agy
    harnesses/antigravity/login.sh                      # once: Google sign-in
    harnesses/antigravity/run.sh tasks-agy/hb-004 gemini-4-neon xhigh

agy runs inside the solver's container with our Harbor agent (agent.py). The conversion
  - builds the solver's image: the task's agent image plus agy. Google's release server only serves the latest
    release; pass --agy-url and --agy-sha512 to pin another one. The image tag carries agy's version, which the
    agent checks before every trial;
  - adds the egress proxy, relaying HTTPS only to the Google hosts agy needs. With a Google account sign-in
    (--auth account, the default): the Code Assist API (cloudcode-pa, daily-cloudcode-pa), the OAuth token refresh
    (oauth2) and the sign-in eligibility check (www, aicode), all under googleapis.com; without the last two the
    sign-in fails with "Eligibility check failed". With an API key (--auth apikey): generativelanguage.googleapis.com.
    Telemetry (play.googleapis.com) and the self-updater are refused.
Backend, verifier, instruction, limits and oracle are the base task's.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common  # noqa: E402

ALLOW = {
    "account": ("cloudcode-pa.googleapis.com:443,daily-cloudcode-pa.googleapis.com:443,oauth2.googleapis.com:443,"
                "www.googleapis.com:443,aicode.googleapis.com:443"),
    "apikey": "generativelanguage.googleapis.com:443",
}
MANIFEST = "https://antigravity-cli-auto-updater-974169037036.us-central1.run.app/manifests/linux_amd64.json"


def latest_release() -> tuple[str, str]:
    with urllib.request.urlopen(MANIFEST, timeout=30) as r:
        m = json.load(r)
    return m["url"], m["sha512"]


def agy_image(base: str, url: str, sha512: str, rebuild: bool) -> str:
    """Build once per base image and release; the tag names agy's version, read from the built binary."""
    prefix = f"btw-agent-agy:{base.split(':', 1)[-1]}"
    by_release = common.build(HERE / "agent.Dockerfile", f"{prefix}-{sha512[:12]}", HERE, rebuild,
                              AGENT_IMAGE=base, AGY_URL=url, AGY_SHA512=sha512)
    out = subprocess.run(["docker", "run", "--rm", "--network", "none", by_release, "agy", "--version"],
                         capture_output=True, text=True, check=True).stdout
    tag = f"{prefix}-agy{out.strip().splitlines()[-1]}"
    subprocess.run(["docker", "tag", by_release, tag], check=True)
    return tag


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tasks", nargs="+", type=Path, help="base task folders")
    ap.add_argument("--out", required=True, type=Path, help="parent folder of the converted tasks")
    ap.add_argument("--auth", choices=sorted(ALLOW), default="account")
    ap.add_argument("--allow", help="comma-separated host:port the proxy relays (default: those of --auth)")
    ap.add_argument("--agy-url", help="agy release tarball (default: the latest release)")
    ap.add_argument("--agy-sha512", help="sha512 of --agy-url")
    ap.add_argument("--rebuild", action="store_true", help="rebuild the images even if they exist")
    a = ap.parse_args()
    if bool(a.agy_url) != bool(a.agy_sha512):
        sys.exit("--agy-url and --agy-sha512 go together")
    url, sha512 = (a.agy_url, a.agy_sha512) if a.agy_url else latest_release()
    allow = a.allow or ALLOW[a.auth]
    a.out.mkdir(parents=True, exist_ok=True)
    egress = common.egress_image(a.rebuild)
    images: dict[str, str] = {}
    for src in a.tasks:
        dst = common.copy_task(src, a.out)
        base = common.base_agent_image(dst)
        if base not in images:
            images[base] = agy_image(base, url, sha512, a.rebuild)
        image = images[base]
        version = image.rsplit("-agy", 1)[-1]
        common.set_solver_image(dst, image, "# The solver's workspace: the `py` kernel, the `forward` client and agy.\n"
                                             "# No network weights, data, torch or internet; the only way out is the egress proxy.")
        compose = common.add_egress(dst, allow, egress, f"Antigravity task ({a.auth}): `main` stays on the internal network; `egress`\n"
                                                        f"relays HTTPS CONNECT to {allow} only (harnesses/antigravity/make_task.py).")
        common.write_compose(dst, compose)
        (dst / ".agy_version").write_text(version + "\n")
        common.append_readme(dst, f"## Antigravity ({a.auth})\n\nSolver image `{image}` (agy {version}); egress allows "
                                  f"only `{allow}`. Run with `harnesses/antigravity/run.sh`.")
        print(f"{dst}: {image}")


if __name__ == "__main__":
    main()
