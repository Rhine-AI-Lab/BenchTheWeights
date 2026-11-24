#!/usr/bin/env python3
"""Report to the backend when Codex works, so a forward pass's clock pause credits only waiting time.

Claude Code is blocked while `forward` runs. Codex can get control back while a pass keeps computing (its exec
tool returns after a yield time and the command goes on in the background), so it could think or run other
commands while the clock is paused. This watcher reads the live Codex session log
($CODEX_HOME/sessions/**/rollout-*.jsonl) and posts to the backend's /agent/active, as [start, end] in epoch
seconds, every model turn in which the agent did something other than wait: from the moment a tool result came
back to the next tool call, if the turn has reasoning or a message or ends in a call that is not a pure wait.
Pure waits are polling a running command without input (write_stdin with chars "") and Codex's wait and sleep
tools. Tool execution itself is not reported, as with Claude Code. Turns outside every pass change nothing.

    activity_watch.py --ensure    # start the watcher once, detached; `forward` runs this before every pass
"""
from __future__ import annotations

import datetime
import glob
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

HOME = os.environ.get("CODEX_HOME", "/tmp/codex-home")
BACKEND = os.environ.get("WSP_BACKEND", "http://backend:8000")
PIDFILE = "/tmp/wsp-active-watch.pid"
LOG = "/tmp/wsp-active-watch.log"
CALLS = ("custom_tool_call", "function_call", "local_shell_call")


def ts(s: str) -> float:
    return datetime.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def is_wait(name: str | None, inp: str) -> bool:
    """True if this tool call only waits for something already running."""
    if name in ("wait", "sleep", "wait_agent"):
        return True
    calls = re.findall(r"tools\.(\w+)\s*\(", inp)
    if name == "write_stdin":
        calls = ["write_stdin"]
    elif not calls:
        return False
    for c in calls:
        if c in ("wait", "sleep"):
            continue
        if c == "write_stdin" and re.search(r"chars\W*:\s*\\?\"\\?\"", inp):
            continue
        return False
    return True


def report(start: float, end: float) -> None:
    body = json.dumps({"start": start, "end": end}).encode()
    req = urllib.request.Request(f"{BACKEND}/agent/active", data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                r.read()
            return
        except Exception as e:  # the backend is busy or restarting: retry, then drop this interval
            err = e
            time.sleep(0.5)
    with open(LOG, "a") as f:
        f.write(f"report failed {start:.3f}-{end:.3f}: {err}\n")


def watch() -> None:
    pos: dict[str, int] = {}
    turn = None  # start of the current model turn (a tool result came back), or None while a tool runs
    work = False  # the current turn has reasoning or a message
    while True:
        for f in sorted(glob.glob(os.path.join(HOME, "sessions", "**", "rollout-*.jsonl"), recursive=True)):
            try:
                with open(f) as fh:
                    fh.seek(pos.get(f, 0))
                    chunk = fh.read()
                    # keep a partial last line for the next read
                    cut = chunk.rfind("\n") + 1
                    pos[f] = pos.get(f, 0) + len(chunk[:cut].encode())
            except OSError:
                continue
            for line in chunk[:cut].splitlines():
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if d.get("type") != "response_item":
                    if d.get("type") == "event_msg" and (d.get("payload") or {}).get("type") == "task_started":
                        turn, work = ts(d["timestamp"]), False
                    continue
                p = d.get("payload") or {}
                ty, t = p.get("type") or "", ts(d["timestamp"])
                if ty in ("reasoning", "message"):
                    if p.get("role") == "user":
                        turn, work = t, False
                        continue
                    work = True
                    if turn is not None:
                        report(turn, t)
                elif ty in CALLS:
                    inp = p.get("input") if p.get("input") is not None else p.get("arguments")
                    inp = inp if isinstance(inp, str) else json.dumps(inp)
                    wait = is_wait(p.get("name"), inp or "")
                    if turn is not None and (work or not wait):
                        report(turn, t)
                    turn, work = None, False
                elif ty.endswith("_output"):
                    turn, work = t, False
        time.sleep(0.25)


def ensure() -> None:
    try:
        pid = int(open(PIDFILE).read())
        os.kill(pid, 0)
        return
    except (OSError, ValueError):
        pass
    with open(os.devnull) as dn, open(LOG, "a") as lg:
        p = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--run"], stdin=dn, stdout=lg, stderr=lg,
                             start_new_session=True, close_fds=True)
    with open(PIDFILE, "w") as f:
        f.write(str(p.pid))


if __name__ == "__main__":
    if sys.argv[1:] == ["--ensure"]:
        ensure()
    elif sys.argv[1:] == ["--run"]:
        watch()
    else:
        sys.exit(__doc__)
