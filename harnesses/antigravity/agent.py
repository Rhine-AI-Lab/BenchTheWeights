"""Harbor agent for Google's Antigravity CLI (agy) 1.5.

    PYTHONPATH=harnesses/antigravity harbor run -p <task> --agent-import-path agent:Antigravity -m google/<model> ...

Harbor 0.18.0's `antigravity-cli` agent was written for an older, Gemini CLI-style agy: it installs agy from the
internet at run time, writes settings to ~/.agy (agy 1.5 reads ~/.gemini/antigravity-cli/settings.json), caps
reasoning at "high" and parses Gemini CLI session files. This agent keeps Harbor's plumbing and replaces those parts:

  install   nothing: agy is in the solver image. The trial stops if its version is not the task's (AGY_VERSION).
  auth      the sign-in saved by login.sh is copied into the container for the run and deleted before Harbor
            collects the logs. With GEMINI_API_KEY set on the host, agy uses the Gemini API instead.
  settings  the counterparts of the Claude Code settings: URL reads and subagents denied; no permission prompts
            (the container is the sandbox); reasoning effort from `--ak reasoning_effort` (default xhigh).
  hooks     busy_hook.py: no tools but waiting, and no stopping, while a command runs; no subagents.
  run       `agy --prompt` with stream-json output, saved to agent/antigravity-cli.jsonl. A watchdog stops agy after
            10 min without output: agy 1.5.11 can hang after an internal error once the model is done, and it
            otherwise writes an event at least every ~10 s, even during a long command.
  check     agent/protocol_check.json: the models that answered, tool calls, web, asynchronous and subagent calls,
            the forward passes that ran in the background (background_audit.py) and the hook's decisions. A trial
            is invalid if it used a web tool, another model answered, no tool call was parsed or the hook files
            changed during the run. search_web has no permission rule and may run on Google's side, so only this
            check catches it.
"""

from __future__ import annotations

import json
import os
import re
import shlex
from pathlib import Path
from typing import Any

from harbor.agents.installed.antigravity_cli import AntigravityCli
from harbor.agents.installed.base import with_prompt_template
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from background_audit import audit_log

AGY = "/root/.local/bin/agy"
SETTINGS_DIR = "/root/.gemini/antigravity-cli"
HOOK_DIR = "/root/.gemini/config"
HOOK_NAME = "busy_hook.py"
HOOK_SRC = Path(__file__).with_name(HOOK_NAME).read_text()
_HOOK_CMD = {"type": "command", "command": f"python3 {HOOK_DIR}/hooks/{HOOK_NAME}", "timeout": 10}
HOOKS_JSON = {"btw-busy": {"PreToolUse": [{"matcher": "*", "hooks": [_HOOK_CMD]}],
                           "PostToolUse": [{"matcher": "run_command", "hooks": [_HOOK_CMD]}],
                           "Stop": [_HOOK_CMD]}}
WATCHDOG_SEC = 600

EFFORTS = ("low", "medium", "high", "xhigh", "max")
WEB_TOOLS = ("search_web", "read_url_content", "read_url", "browser_subagent", "web_search", "google_search",
             "url_context")
BROWSER_TOOLS = ("open_browser_url", "read_browser_page", "execute_browser_javascript", "click_browser_pixel",
                 "list_browser_pages", "capture_browser_screenshot", "capture_browser_console_logs")
ASYNC_TOOLS = ("command_status", "send_command_input", "read_terminal", "manage_task", "schedule")
SUBAGENT_TOOLS = ("invoke_subagent", "browser_subagent", "define_subagent", "manage_subagents")
# everything the sign-in may leave under /root, removed before the logs are collected
AUTH_REMOVE = "/root/.gemini /root/.config /root/.local/share/keyrings /root/.cache/antigravity /root/.antigravity"


class Antigravity(AntigravityCli):
    def __init__(self, *args, reasoning_effort: str | None = "xhigh", **kwargs):
        if reasoning_effort not in EFFORTS:
            raise ValueError(f"reasoning_effort must be one of {EFFORTS}, got {reasoning_effort!r}")
        super().__init__(*args, reasoning_effort=None, **kwargs)  # the parent accepts up to "high" only
        self._reasoning_effort = reasoning_effort

    def get_version_command(self) -> str | None:
        return f"{AGY} --version"

    async def install(self, environment: BaseEnvironment) -> None:
        want = os.environ.get("AGY_VERSION", "")
        r = await environment.exec(command=f"{AGY} --version", user="root")
        if r.return_code != 0:
            raise RuntimeError(f"agy is not in the solver image ({AGY}); convert the task with make_task.py")
        got = ((r.stdout or "").strip().splitlines() or [""])[-1]
        if want and got != want:
            raise RuntimeError(f"agy {got} in the image, task built for {want}")

    @staticmethod
    def _settings(model: str, api_key: bool) -> dict[str, Any]:
        s: dict[str, Any] = {
            "model": model,
            "agentMode": "",  # default mode: "auto" needs agy's terminal sandbox
            "toolPermission": "always-proceed",
            "enableTerminalSandbox": False,
            "trustedWorkspaces": ["/work"],
            "allowNonWorkspaceAccess": True,
            "permissions": {"deny": ["read_url(*)", "execute_url(*)", "invoke_subagent(*)", "define_subagent(*)",
                                     "manage_subagents(*)", "browser_subagent(*)"]},
            "showTips": False,
            "showFeedbackSurvey": False,
            "notifications": False,
        }
        if api_key:
            s["modelProvider"] = "gemini"  # agy ignores GEMINI_API_KEY without it
        return s

    @staticmethod
    async def _upload_auth(environment: BaseEnvironment) -> bool:
        auth = Path(os.environ.get("AGY_AUTH_DIR", Path.home() / ".config/agy-auth")).expanduser() / "root"
        if not auth.is_dir():
            return False
        await environment.upload_dir(auth, "/root")
        return True

    @with_prompt_template
    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        if not self.model_name or "/" not in self.model_name:
            raise ValueError("the model must look like google/<model>")
        model = self.model_name.split("/", 1)[1]
        env = {"GEMINI_CLI_TRUST_WORKSPACE": "true"}
        api_key = "GEMINI_API_KEY" in os.environ
        if api_key:
            env["GEMINI_API_KEY"] = os.environ["GEMINI_API_KEY"]
        elif not await self._upload_auth(environment):
            raise RuntimeError("no Google sign-in (harnesses/antigravity/login.sh) and no GEMINI_API_KEY")
        settings = shlex.quote(json.dumps(self._settings(model, api_key), indent=2))
        hooks = shlex.quote(json.dumps(HOOKS_JSON, indent=2))
        await self.exec_as_root(environment, f"mkdir -p {SETTINGS_DIR} {HOOK_DIR}/hooks "
                                             f"&& printf %s {settings} > {SETTINGS_DIR}/settings.json "
                                             f"&& printf %s {hooks} > {HOOK_DIR}/hooks.json "
                                             f"&& printf %s {shlex.quote(HOOK_SRC)} > {HOOK_DIR}/hooks/{HOOK_NAME} "
                                             f"&& chmod -R go-rwx /root "
                                             f"&& cp {HOOK_DIR}/hooks.json /logs/agent/hooks.json "
                                             f"&& cp {HOOK_DIR}/hooks/{HOOK_NAME} /logs/agent/{HOOK_NAME}")
        # --prompt takes the prompt as its value; a bare --print would take the next flag as the prompt
        agy = (f"cd /work && {AGY} --dangerously-skip-permissions --disable-slash-commands "
               f"--model {shlex.quote(model)} --effort {self._reasoning_effort} --output-format stream-json "
               f"--prompt {shlex.quote(instruction)} 2>/logs/agent/antigravity-cli.stderr </dev/null "
               f"| stdbuf -oL tee /logs/agent/antigravity-cli.jsonl")
        cmd = (f"({agy}) & P=$!; while kill -0 $P 2>/dev/null; do sleep 30; "
               f"a=$(( $(date +%s) - $(stat -c %Y /logs/agent/antigravity-cli.jsonl 2>/dev/null || date +%s) )); "
               f"if [ $a -gt {WATCHDOG_SEC} ]; then echo 'watchdog: no agent output for {WATCHDOG_SEC} s, stopping agy' "
               f">> /logs/agent/antigravity-cli.stderr; pkill -f {AGY}; fi; done; wait $P")
        try:
            await self.exec_as_agent(environment, command=cmd, env=env)
        finally:
            await self.exec_as_root(environment, f"cp {SETTINGS_DIR}/settings.json /logs/agent/settings.json 2>/dev/null; "
                                                 f"cp {HOOK_DIR}/hooks.json /logs/agent/hooks_end.json 2>/dev/null; "
                                                 f"cp {HOOK_DIR}/hooks/{HOOK_NAME} /logs/agent/busy_hook_end.py 2>/dev/null; "
                                                 f"cp {HOOK_DIR}/hooks/busy.log /logs/agent/busy_hook.log 2>/dev/null; "
                                                 f"rm -rf {AUTH_REMOVE}")

    @staticmethod
    def _walk(o: Any):
        if isinstance(o, dict):
            yield o
            for v in o.values():
                yield from Antigravity._walk(v)
        elif isinstance(o, list):
            for v in o:
                yield from Antigravity._walk(v)

    def populate_context_post_run(self, context: AgentContext) -> None:
        log = self.logs_dir / "antigravity-cli.jsonl"
        if not log.exists():
            return
        events = []
        for line in log.read_text(errors="replace").splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        models, tools, waits = {}, {}, []
        tin = tout = tcache = 0
        seen_steps: set = set()

        def add_tool(name: str, a: Any) -> None:
            tools[name] = tools.get(name, 0) + 1
            if isinstance(a, str):
                try:
                    a = json.loads(a)
                except json.JSONDecodeError:
                    a = {}
            if name == "run_command" and isinstance(a, dict):
                w = a.get("WaitMsBeforeAsync", a.get("wait_ms_before_async"))
                if w is not None:
                    waits.append(w)

        def add_usage(u: dict) -> None:
            nonlocal tin, tout, tcache
            tin += int(u.get("input_tokens") or u.get("promptTokenCount") or u.get("inputTokens") or 0)
            tout += int(u.get("output_tokens") or u.get("candidatesTokenCount") or u.get("outputTokens") or 0)
            tout += int(u.get("thoughtsTokenCount") or u.get("thinking_tokens") or 0)
            tcache += int(u.get("cache_read_tokens") or u.get("cached_tokens") or u.get("cachedContentTokenCount")
                          or u.get("cachedTokens") or 0)

        # agy 1.5.11 stream-json: {"event": "init", "init": {"model"}}; then {"event": "step_update", "step_update":
        # {"step_index", "state": ACTIVE|DONE, "step_type": tool|agent_response|..., "tool_name", "tool_info":
        # {"name", "parameters"}, "usage"}}, each tool step once ACTIVE and once DONE; and a final
        # {"event": "result", "result": {"usage"}} with the session's totals.
        steps = [e["step_update"] for e in events if isinstance(e.get("step_update"), dict)]
        result_usage = next((e["result"]["usage"] for e in events if isinstance(e.get("result"), dict)
                             and isinstance(e["result"].get("usage"), dict)), None)
        for e in events:
            m = e["init"].get("model") if isinstance(e.get("init"), dict) else None
            if isinstance(m, str) and m:
                models[m] = models.get(m, 0) + 1
        for s in steps:
            for k in ("model", "modelName", "model_name"):
                if isinstance(s.get(k), str) and s[k]:
                    models[s[k]] = models.get(s[k], 0) + 1
            if s.get("step_type") == "tool":
                key = (s.get("conversation_id"), s.get("step_index"))
                if key in seen_steps:
                    continue
                seen_steps.add(key)
                info = s.get("tool_info") if isinstance(s.get("tool_info"), dict) else {}
                name = s.get("tool_name") or info.get("name")
                if isinstance(name, str) and name:
                    add_tool(name, info.get("parameters") or info.get("args") or {})
            elif result_usage is None and isinstance(s.get("usage"), dict) and s.get("state") == "DONE":
                add_usage(s["usage"])
        if result_usage is not None:
            add_usage(result_usage)
        if not steps:
            # another agy version's format: walk every object, so the check is not silently blind
            for d in (x for e in events for x in self._walk(e)):
                for k in ("model", "modelName", "model_name", "modelVersion"):
                    if isinstance(d.get(k), str) and d[k]:
                        models[d[k]] = models.get(d[k], 0) + 1
                name = d.get("name") or d.get("tool_name") or d.get("toolName")
                if isinstance(name, str) and any(k in d for k in ("args", "input", "arguments", "parameters")):
                    add_tool(name, d.get("args") or d.get("input") or d.get("arguments") or d.get("parameters") or {})
                if isinstance(d.get("usage"), dict):
                    add_usage(d["usage"])

        bg = audit_log(log)
        tampered = []
        for start, end in (("hooks.json", "hooks_end.json"), (HOOK_NAME, "busy_hook_end.py")):
            fa, fb = self.logs_dir / start, self.logs_dir / end
            if not fa.exists() or not fb.exists() or fa.read_bytes() != fb.read_bytes():
                tampered.append(f"{start} missing or changed by the end of the run")
        want = self.model_name.split("/", 1)[1] if self.model_name else ""
        web = {t: n for t, n in tools.items() if t in WEB_TOOLS or t in BROWSER_TOOLS or "browser" in t}
        other_models = sorted(m for m in models if want and want not in m)
        no_tools = bool(events) and not tools
        check = {
            "requested_model": want,
            "models_seen": models,
            "tool_calls": tools,
            "web_tool_calls": web,
            "async_tool_calls": {t: n for t, n in tools.items() if t in ASYNC_TOOLS},
            "run_command_wait_ms": waits,
            "subagent_calls": {t: n for t, n in tools.items() if t in SUBAGENT_TOOLS},
            "parsed_steps": len(steps),
            "background_commands": bg["background_commands"],
            "background_forward_passes": bg["background_forward_passes"],
            "background_forward_sec": bg["background_forward_sec"],
            "watchdog_stopped": _watchdog(self.logs_dir),
            "hook_tampering": tampered,
            "busy_hook": _busy_summary(self.logs_dir / "busy_hook.log"),
            "invalid": bool(web or other_models or no_tools or tampered),
            "reasons": ([f"web tools used: {web}"] if web else [])
                       + ([f"other models answered: {other_models}"] if other_models else [])
                       + (["no tool calls parsed from the agent log"] if no_tools else [])
                       + tampered,
        }
        (self.logs_dir / "protocol_check.json").write_text(json.dumps(check, indent=2))
        if tin or tout:
            context.n_input_tokens, context.n_output_tokens, context.n_cache_tokens = tin, tout, tcache


def _busy_summary(path: Path) -> dict:
    """Counts from the hook's log: completed commands, and tools refused while a command ran or as subagents."""
    if not path.exists():
        return {}
    busy = sub = other = cmds = 0
    for line in path.read_text(errors="replace").splitlines():
        try:
            x = json.loads(line)
        except json.JSONDecodeError:
            continue
        if x.get("event") == "PostToolUse":
            cmds += 1
        elif x.get("decision") == "deny":
            if x.get("tool") in SUBAGENT_TOOLS:
                sub += 1
            elif int((re.search(r"busy=(\d+)", str(x.get("note", ""))) or [0, "0"])[1]) > 0:
                busy += 1
            else:
                other += 1
    return {"commands_completed": cmds, "refused_while_busy": busy, "refused_subagent": sub, "refused_other": other}


def _watchdog(logs: Path) -> bool:
    e = logs / "antigravity-cli.stderr"
    return e.exists() and "watchdog:" in e.read_text(errors="replace")
