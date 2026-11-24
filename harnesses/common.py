"""Shared steps of the harness adapters: copy a task, swap the solver's image, add the egress proxy.

A base task (scripts/make_task.py) gives the solver's container no route out at all. An agent that runs inside
the container, such as Claude Code or Codex, needs its model API, so each adapter adds an egress proxy that
relays HTTPS to that API only. Backend, verifier, instruction, limits and oracle stay the base task's.
"""

from __future__ import annotations

import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PROXY_PORT = 3128
NO_PROXY = "backend,localhost,127.0.0.1"


def version() -> str:
    text = (ROOT / "src/benchtheweights/__init__.py").read_text()
    return re.search(r'__version__ = "(.+)"', text).group(1)


def image_exists(name: str) -> bool:
    return subprocess.run(["docker", "image", "inspect", name], capture_output=True).returncode == 0


def build(dockerfile: Path, tag: str, context: Path, rebuild: bool = False, **build_args: str) -> str:
    if rebuild or not image_exists(tag):
        cmd = ["docker", "build", "-f", str(dockerfile), "-t", tag]
        for k, v in build_args.items():
            cmd += ["--build-arg", f"{k}={v}"]
        print("+", " ".join(cmd + [str(context)]), flush=True)
        subprocess.run(cmd + [str(context)], check=True)
    return tag


def egress_image(rebuild: bool = False) -> str:
    d = ROOT / "harnesses/egress"
    return build(d / "Dockerfile", f"btw-egress:{version()}", d, rebuild)


def base_agent_image(task: Path) -> str:
    froms = [l.split()[1] for l in (task / "environment/Dockerfile").read_text().splitlines() if l.startswith("FROM ")]
    if len(froms) != 1:
        sys.exit(f"{task}: expected exactly one FROM line in environment/Dockerfile")
    return froms[0]


def copy_task(src: Path, out: Path) -> Path:
    """Copy a base task to out/<name>, writable even if the source is a read-only frozen set."""
    src = src.expanduser().resolve()
    dst = out.expanduser().resolve() / src.name
    if dst.exists():
        sys.exit(f"{dst} exists; remove it first")
    compose = yaml.safe_load((src / "environment/docker-compose.yaml").read_text())
    if "egress" in compose["services"] or compose["services"]["main"].get("networks") != ["repair"] \
            or not compose["networks"]["repair"].get("internal"):
        sys.exit(f"{src}: not a base task (scripts/make_task.py)")
    shutil.copytree(src, dst)
    for p in [dst, *dst.rglob("*")]:
        p.chmod(p.stat().st_mode | stat.S_IWUSR)
    return dst


def set_solver_image(task: Path, image: str, comment: str, extra: str = "") -> None:
    (task / "environment/Dockerfile").write_text(f"{comment}\nFROM {image}\nWORKDIR /work\n{extra}")


def add_egress(task: Path, allow: str, egress: str, header: str) -> dict:
    """Add the egress proxy, relaying HTTPS CONNECT to `allow` (comma-separated host:port) only.
    Returns the compose file as a dict; write it back with write_compose()."""
    compose = yaml.safe_load((task / "environment/docker-compose.yaml").read_text())
    services = compose["services"]
    proxy = f"http://egress:{PROXY_PORT}"
    services["main"].setdefault("environment", {}).update(
        {"HTTPS_PROXY": proxy, "https_proxy": proxy, "NO_PROXY": NO_PROXY, "no_proxy": NO_PROXY})
    services["main"].setdefault("depends_on", {})["egress"] = {"condition": "service_healthy"}
    services["egress"] = {
        "image": egress,
        "environment": {"EGRESS_ALLOW": allow, "EGRESS_PORT": str(PROXY_PORT)},
        "cpus": 1,
        "mem_limit": "256m",
        "networks": ["repair", "egress"],
        "healthcheck": {
            "test": ["CMD", "python", "-c", f"import socket; socket.create_connection(('127.0.0.1', {PROXY_PORT}), 3)"],
            "interval": "2s", "timeout": "5s", "retries": 30, "start_period": "2s",
        },
    }
    compose["networks"]["egress"] = {"driver": "bridge"}
    compose["_header"] = header
    add_artifact(task, "/var/log/egress", "egress", "The egress proxy's log: every request the solver's container made to leave the task network.")
    return compose


def write_compose(task: Path, compose: dict) -> None:
    header = compose.pop("_header")
    text = "".join(f"# {line}\n" for line in header.splitlines())
    (task / "environment/docker-compose.yaml").write_text(text + yaml.safe_dump(compose, sort_keys=False))


def add_artifact(task: Path, source: str, service: str, comment: str) -> None:
    toml = task / "task.toml"
    toml.write_text(toml.read_text().rstrip() + f'\n\n# {comment}\n[[artifacts]]\nsource = "{source}"\nservice = "{service}"\n')


def append_readme(task: Path, text: str) -> None:
    readme = task / "README.md"
    if readme.exists():
        readme.write_text(readme.read_text().rstrip() + "\n\n" + text.strip() + "\n")
