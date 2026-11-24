#!/usr/bin/env python3
"""List the commands, and the forward passes among them, that agy ran in the background, with their durations.

    harnesses/antigravity/background_audit.py jobs/<job>/<trial> ...

agy's stream-json does not say when a command moves to the background, and it emits events in step order, not
time order. Two signals do show it:
  - timestamps: a run_command result carries "Created At" and "Completed At". A command that completed after a
    later command was created was still running while the model went on;
  - polling: manage_task with Action status, kill or wait on task-N means command N was running in the background.
    A bare mention of task-N, such as reading its log, proves nothing.
While a backgrounded forward pass computes, the backend's clock is paused and the model may keep working, so its
duration is an upper bound on that free time.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

FORWARD = re.compile(r"\bforward\b|fwd\(|forward\(")


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def audit_log(path: Path) -> dict:
    cmds: dict[int, dict] = {}
    polled: set[int] = set()
    raw = path.read_text(errors="replace")
    for line in raw.splitlines():
        if not line.startswith("{"):
            continue
        try:
            s = json.loads(line).get("step_update")
        except json.JSONDecodeError:
            continue
        if not isinstance(s, dict) or s.get("step_type") != "tool":
            continue
        info = s.get("tool_info") if isinstance(s.get("tool_info"), dict) else {}
        par = info.get("parameters") if isinstance(info.get("parameters"), dict) else {}
        name = s.get("tool_name") or info.get("name")
        if name == "manage_task" and par.get("Action") in ("status", "kill", "wait") and par.get("TaskId"):
            m = re.search(r"task-(\d+)", str(par["TaskId"]))
            if m:
                polled.add(int(m.group(1)))
        if name == "run_command" and s.get("state") == "DONE":
            out = info.get("output") or ""
            c = re.search(r"Created At: (\S+)", out)
            d = re.search(r"Completed At: (\S+)", out)
            cl = par.get("CommandLine", "")
            cmds[s["step_index"]] = {"created": _ts(c.group(1)) if c else None,
                                     "completed": _ts(d.group(1)) if d else None,
                                     "forward": bool(FORWARD.search(cl)),
                                     "duration_sec": float(s.get("duration_seconds") or 0.0),
                                     "command": cl[:120]}
    steps = sorted(cmds)
    background: dict[int, str] = {}
    for i in steps:
        done = cmds[i]["completed"]
        if done is None:
            continue
        for j in steps:
            if j > i and cmds[j]["created"] is not None and cmds[j]["created"] < done:
                background[i] = f"step {j} started before it finished"
                break
    for i in polled:
        if i in cmds:
            background.setdefault(i, "polled with manage_task")
    bf = {i: r for i, r in background.items() if cmds[i]["forward"]}
    return {
        "run_commands": len(cmds),
        "timestamped": sum(1 for c in cmds.values() if c["created"] is not None),
        "background_commands": len(background),
        "background_forward_passes": [{"step": i, "why": r, "duration_sec": round(cmds[i]["duration_sec"], 1),
                                       "command": cmds[i]["command"]} for i, r in sorted(bf.items())],
        "background_forward_sec": round(sum(cmds[i]["duration_sec"] for i in bf), 1),
    }


def main() -> None:
    print(f"{'trial':58} {'cmds':>5} {'bg cmds':>7} {'bg fwd':>6} {'bg fwd s':>8} {'recovery':>9}")
    for arg in sys.argv[1:]:
        t = Path(arg)
        log = t / "agent" / "antigravity-cli.jsonl"
        if not log.exists():
            continue
        a = audit_log(log)
        m = t / "verifier" / "metrics.json"
        rec = f"{json.loads(m.read_text())['recovery']:.3f}" if m.exists() else "-"
        name = f"{t.parent.name}"[:58]
        print(f"{name:58} {a['run_commands']:>5} {a['background_commands']:>7} "
              f"{len(a['background_forward_passes']):>6} {a['background_forward_sec']:>8.0f} {rec:>9}")


if __name__ == "__main__":
    main()
