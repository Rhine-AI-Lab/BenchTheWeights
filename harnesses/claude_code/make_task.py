#!/usr/bin/env python3
"""Convert base tasks (scripts/make_task.py) into Claude Code tasks.

    harnesses/claude_code/make_task.py tasks/hb-012 ... --out tasks-claude-code
    harnesses/claude_code/run.sh tasks-claude-code/hb-012 claude-opus-5-5

Claude Code runs inside the solver's container, so the conversion
  - builds the solver's image: the task's agent image plus Claude Code, installed at build time (Harbor's
    install step needs the internet and is skipped because `claude` is already on the PATH), with WebSearch
    and WebFetch denied in /etc/claude-code/managed-settings.json;
  - adds the egress proxy, relaying HTTPS to api.anthropic.com:443 only, with HTTPS_PROXY set for the solver;
  - turns off Claude Code's background tasks and raises its Bash timeouts to the agent timeout. The backend's
    clock pauses during forward passes, which is fair only if the agent waits on them.
Backend, verifier, instruction, limits and oracle are the base task's.
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common  # noqa: E402

ALLOW = "api.anthropic.com:443"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tasks", nargs="+", type=Path, help="base task folders")
    ap.add_argument("--out", required=True, type=Path, help="parent folder of the converted tasks")
    ap.add_argument("--cc-version", default="2.1.318", help="Claude Code version baked into the image (default %(default)s)")
    ap.add_argument("--rebuild", action="store_true", help="rebuild the images even if they exist")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    egress = common.egress_image(a.rebuild)
    for src in a.tasks:
        dst = common.copy_task(src, a.out)
        base = common.base_agent_image(dst)
        tag = f"btw-agent-claude-code:{base.split(':', 1)[-1]}-cc{a.cc_version}"
        image = common.build(HERE / "agent.Dockerfile", tag, HERE, a.rebuild, AGENT_IMAGE=base, CC_VERSION=a.cc_version)
        common.set_solver_image(dst, image, "# The solver's workspace: the `py` kernel, the `forward` client and Claude Code.\n"
                                             "# No network weights, data, torch or internet; the only way out is the egress proxy.")
        compose = common.add_egress(dst, ALLOW, egress, f"Claude Code task: `main` stays on the internal network; `egress` relays\n"
                                                        f"HTTPS CONNECT to {ALLOW} only (harnesses/claude_code/make_task.py).")
        agent_ms = str(int(1000 * tomllib.loads((dst / "task.toml").read_text())["agent"]["timeout_sec"]))
        compose["services"]["main"]["environment"].update(
            {"CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1", "BASH_DEFAULT_TIMEOUT_MS": agent_ms, "BASH_MAX_TIMEOUT_MS": agent_ms})
        common.write_compose(dst, compose)
        common.append_readme(dst, f"## Claude Code\n\nSolver image `{image}` (Claude Code {a.cc_version}); egress allows "
                                  f"only `{ALLOW}`. Run with `harnesses/claude_code/run.sh`.")
        print(f"{dst}: {image}")


if __name__ == "__main__":
    main()
