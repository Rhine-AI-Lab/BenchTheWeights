#!/usr/bin/env python3
"""Convert base tasks (scripts/make_task.py) into Codex tasks.

    harnesses/codex/make_task.py tasks/hb-004 ... --out tasks-codex                                   # OpenAI
    harnesses/codex/make_task.py tasks/hb-004 ... --out tasks-codex-openrouter --provider openrouter   # Claude, Gemini
    harnesses/codex/make_task.py tasks/hb-004 ... --out tasks-codex-alibaba --provider alibaba         # Qwen, GLM, DeepSeek
    harnesses/codex/run.sh <provider> tasks-codex/hb-004 <model>

Codex runs inside the solver's container with Harbor's `codex` agent. The conversion
  - builds the solver's image: the task's agent image plus Node and Codex, installed at build time (Harbor's install
    step needs the internet and is skipped when `--ak version` matches), and the activity watcher
    (activity_watch.py), which `forward` starts before every pass. Codex can work while a pass computes; the
    watcher reports those turns to the backend, which takes them out of the clock pause;
  - adds the egress proxy, relaying HTTPS to the provider's API only;
  - for openrouter and alibaba, pins Codex's `apps` and `multi_agent` features off in /etc/codex/requirements.toml.
    Codex sends their tools with type "namespace", which Alibaba rejects; the pins give every non-OpenAI model
    the same tool list;
  - for openrouter, adds the prompt-cache relay (cache_relay.py) between Codex and the proxy.
Backend, verifier, instruction, limits and oracle are the base task's.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common  # noqa: E402

ALLOW = {
    "openai": "api.openai.com:443,chatgpt.com:443,auth.openai.com:443",  # API key, ChatGPT login and its refresh
    "openrouter": "openrouter.ai:443",
    "alibaba": "token-plan.ap-southeast-1.maas.aliyuncs.com:443",
}
REQUIREMENTS = "[features]\napps = false\nmulti_agent = false\n"


def relay_service(image: str) -> dict:
    return {
        "image": image,
        "environment": {"RELAY_UPSTREAM": "openrouter.ai", "RELAY_PROXY": f"egress:{common.PROXY_PORT}",
                        "RELAY_PORT": "8080", "RELAY_CACHE_MODELS": "anthropic/"},
        "depends_on": {"egress": {"condition": "service_healthy"}},
        "cpus": 1,
        "mem_limit": "256m",
        "networks": ["repair"],
        "healthcheck": {"test": ["CMD", "python", "-c", "import urllib.request; "
                                 "urllib.request.urlopen('http://127.0.0.1:8080/relay-health', timeout=3)"],
                        "interval": "2s", "timeout": "5s", "retries": 30, "start_period": "2s"},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tasks", nargs="+", type=Path, help="base task folders")
    ap.add_argument("--out", required=True, type=Path, help="parent folder of the converted tasks")
    ap.add_argument("--provider", choices=sorted(ALLOW), default="openai")
    ap.add_argument("--allow", help="comma-separated host:port the proxy relays (default: the provider's)")
    ap.add_argument("--codex-version", default="0.167.0", help="Codex version baked into the image (default %(default)s)")
    ap.add_argument("--rebuild", action="store_true", help="rebuild the images even if they exist")
    a = ap.parse_args()
    allow = a.allow or ALLOW[a.provider]
    a.out.mkdir(parents=True, exist_ok=True)
    egress = common.egress_image(a.rebuild)
    relay = None
    if a.provider == "openrouter":
        relay = common.build(HERE / "relay.Dockerfile", f"btw-codex-relay:{common.version()}", HERE, a.rebuild,
                             EGRESS_IMAGE=egress)
    for src in a.tasks:
        dst = common.copy_task(src, a.out)
        base = common.base_agent_image(dst)
        tag = f"btw-agent-codex:{base.split(':', 1)[-1]}-codex{a.codex_version}"
        image = common.build(HERE / "agent.Dockerfile", tag, HERE, a.rebuild, AGENT_IMAGE=base, CODEX_VERSION=a.codex_version)
        extra = ""
        if a.provider != "openai":
            (dst / "environment/codex_requirements.toml").write_text(REQUIREMENTS)
            extra = ("COPY codex_requirements.toml /etc/codex/requirements.toml\n"
                     "RUN chmod 0644 /etc/codex/requirements.toml\n")
        common.set_solver_image(dst, image, "# The solver's workspace: the `py` kernel, the `forward` client and Codex.\n"
                                             "# No network weights, data, torch or internet; the only way out is the egress proxy.",
                                extra)
        compose = common.add_egress(dst, allow, egress, f"Codex task ({a.provider}): `main` stays on the internal network; `egress` relays\n"
                                                        f"HTTPS CONNECT to {allow} only (harnesses/codex/make_task.py).")
        if relay:
            svc = compose["services"]
            svc["relay"] = relay_service(relay)
            svc["main"]["depends_on"]["relay"] = {"condition": "service_healthy"}
            for k in ("NO_PROXY", "no_proxy"):
                svc["main"]["environment"][k] += ",relay"
            common.add_artifact(dst, "/var/log/relay", "relay", "The cache relay's log: one line per model request, no headers or bodies.")
        common.write_compose(dst, compose)
        common.append_readme(dst, f"## Codex ({a.provider})\n\nSolver image `{image}` (Codex {a.codex_version}); egress allows "
                                  f"only `{allow}`. Run with `harnesses/codex/run.sh {a.provider}`.")
        print(f"{dst}: {image}")


if __name__ == "__main__":
    main()
