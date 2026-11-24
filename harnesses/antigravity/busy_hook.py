#!/usr/bin/env python3
"""agy hook that keeps the solver waiting while a command runs (PreToolUse on all tools, PostToolUse on
run_command, Stop).

Claude Code runs with background tasks off and Codex with the activity watcher, so neither can work while a
forward pass computes and the backend's clock is paused. agy has no such setting: it hands the model a turn about
10 s into any long command. This hook:
  - marks a run_command as running at PreToolUse and clears the mark at PostToolUse. While a command runs, every
    tool except waiting on it (manage_task, schedule, wait) is refused;
  - clears the mark of a command the model kills with manage_task, since agy fires no PostToolUse for it. agy
    names a command's task task-<step>, so marks are keyed by step index;
  - refuses subagents, whose turns may not pass through these hooks;
  - at Stop, refuses to end the session while a command runs. Without it, models concluded the tools were broken
    and gave up mid-command.
Every decision is appended to busy.log next to this file, which the agent copies into the trial's logs.
"""

import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, "running")
LOG = os.path.join(HERE, "busy.log")
STALE_SEC = 660  # agy moves a command to the background after at most 10 min
WAITING_TOOLS = ("manage_task", "schedule", "wait", "wait_5_seconds")
SUBAGENT_TOOLS = ("invoke_subagent", "define_subagent", "manage_subagents", "browser_subagent")


def log(event, name, decision, note=""):
    try:
        with open(LOG, "a") as f:
            f.write(json.dumps({"t": round(time.time(), 2), "event": event, "tool": name, "decision": decision,
                                "note": note}) + "\n")
    except OSError:
        pass


def running():
    """[(mark, seconds since the command started)] for the commands still running."""
    try:
        now = time.time()
        ages = ((f, now - os.path.getmtime(os.path.join(STATE, f))) for f in os.listdir(STATE))
        return [(f, age) for f, age in ages if age < STALE_SEC]
    except OSError:
        return []


def busy_msg(busy):
    age = int(max(a for _, a in busy))
    return (f"A command you started is still running ({age} s so far). This is normal: long analyses and forward passes "
            "take a while, and the tools are working. In this benchmark you cannot do anything else while a command "
            "runs, and ending your session now would forfeit your remaining time: wait for it (manage_task with "
            "Action status, or schedule a short wait), then continue your work with its result.")


def main():
    try:
        ev = json.load(sys.stdin)
    except Exception:
        ev = {}
    event = ev.get("hookEventName") or ("PostToolUse" if "error" in ev else "PreToolUse")
    call = ev.get("toolCall") or {}
    name = call.get("name") or ""
    step = ev.get("stepIdx")
    args = call.get("args") or {}
    mark = f"s{int(step)}" if isinstance(step, (int, float)) or str(step).isdigit() else \
        "c" + ("".join(c for c in str(call.get("id") or "") if c.isalnum() or c in "-_") or "unknown")

    if event == "Stop":
        busy = running()
        if busy:
            log(event, "-", "continue", f"busy={len(busy)}")
            print(json.dumps({"decision": "continue", "reason": busy_msg(busy)}))
        else:
            log(event, "-", "allow-stop", "busy=0")
            print("{}")
        return

    if event == "PostToolUse":
        if name == "run_command":
            try:
                os.remove(os.path.join(STATE, mark))
            except OSError:
                pass
            log(event, name, "done", f"{mark} id={call.get('id')}")
        print("{}")
        return

    if name == "manage_task" and str(args.get("Action", "")).lower() == "kill":
        m = re.search(r"task-(\d+)", str(args.get("TaskId", "")))
        if m:
            try:
                os.remove(os.path.join(STATE, f"s{m.group(1)}"))
                log(event, name, "cleared", f"s{m.group(1)} (killed)")
            except OSError:
                log(event, name, "kill-unmatched", f"task-{m.group(1)}")
    out = {"decision": "allow"}
    busy = running()
    if name in SUBAGENT_TOOLS:
        out = {"decision": "deny", "reason": "Subagents are disabled in this benchmark: work in this session only."}
    elif busy and name not in WAITING_TOOLS:
        out = {"decision": "deny", "reason": busy_msg(busy)}
    elif name == "run_command":
        os.makedirs(STATE, exist_ok=True)
        open(os.path.join(STATE, mark), "w").close()
    log(event, name, out["decision"], f"busy={len(busy)} step={step}")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
