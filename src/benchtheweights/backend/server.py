"""The trusted backend, in its own container next to the solver's.

It serves the weight-space API to the solver's kernel over the task's internal network. The solver's
container never sees this filesystem: the episode, the checkpoint, the starting weights and the dev split
exist only here. Requests are serialized through one lock and logged.

The clock stops while a forward pass computes, so slow passes and a loaded machine do not eat the solver's
working time. Any other request that arrives during a pass shows that the solver is not only waiting, and
the clock runs again from that moment. Agents that can work while a command runs (Codex) also report their
working intervals to /agent/active, and those intervals are taken out of the pause.

API (POST /v1/<command>):
    JSON body -> JSON response: info, usage, edited, drop_edit, oracle-restore
    tensor        JSON {"name", "which"}           -> .npy bytes
    data          {}                               -> .npz bytes (the dev split)
    save_edit     ?name=<tensor>, .npy body        -> JSON
    forward       JSON request, or an .npz body with the request under "__request__" and custom
                  inputs under "input:<field>"     -> .npz bytes, summary JSON under "__summary__"
Errors are JSON {"ok": false, "error": ...} with a 4xx or 5xx status.

    python -m benchtheweights.backend.server --episode /srv/episode.json --port 8000
"""

from __future__ import annotations

import argparse
import io
import json
import os
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from ..engine import ToolError, WeightSpaceEngine
from ..submission import write_bundle

MAX_BODY = 512 * 2**20
# the signed bundle is written again after these, so it always holds the current edits
EXPORT_AFTER = {"save_edit", "drop_edit", "forward", "oracle-restore"}
COMMANDS = {"info", "usage", "edited", "tensor", "data", "save_edit", "drop_edit", "forward", "oracle-restore"}


def npy_bytes(a: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, a, allow_pickle=False)
    return buf.getvalue()


def npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def json_array(obj) -> np.ndarray:
    return np.frombuffer(json.dumps(obj).encode(), dtype=np.uint8)


class Backend:
    def __init__(self, episode_path: Path, export_dir: Path):
        self.episode = json.loads(episode_path.read_text())
        self.export_dir = export_dir
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = export_dir / "tool_log.jsonl"
        self.lock = threading.Lock()
        self.fwd_started: float | None = None  # a forward pass has been computing since then
        self.fwd_disturbed: float | None = None  # another request arrived during it at this time
        self.passes: list = []  # [start, end of the paused part, credited sec] per finished pass
        self.active: list = []  # [start, end] intervals in which the agent reported working
        self.plock = threading.Lock()
        self.engine = WeightSpaceEngine(self.episode)
        self.export(final=False)

    def _credit(self, start: float, end: float) -> float:
        """The paused seconds of [start, end], minus the reported agent activity in it."""
        iv = sorted((max(a, start), min(b, end)) for a, b in self.active if b > start and a < end)
        busy, cur = 0.0, None
        for a, b in iv:
            if cur and a <= cur[1]:
                cur[1] = max(cur[1], b)
            else:
                if cur:
                    busy += cur[1] - cur[0]
                cur = [a, b]
        if cur:
            busy += cur[1] - cur[0]
        return max(0.0, end - start - busy)

    def agent_active(self, start: float, end: float) -> dict:
        """Record an interval of agent work and take it out of the pause of every pass it overlaps."""
        now = time.time()
        end = min(float(end), now)
        start = float(start)
        if not end > start:
            return {"ok": True, "removed_sec": 0.0}
        removed = 0.0
        with self.plock:
            self.active.append([start, end])
            for p in self.passes:
                if end > p[0] and start < p[1]:
                    new = self._credit(p[0], p[1])
                    removed += p[2] - new
                    self.engine.paused_sec -= p[2] - new
                    p[2] = new
            fs = self.fwd_started  # a running pass: _credit takes this interval out when it ends
        if removed or (fs is not None and end > fs):
            self.log({"command": "agent_active", "epoch": round(now, 3), "start": round(start, 3),
                      "end": round(end, 3), "removed_sec": round(removed, 3), "during_pass": fs is not None})
        return {"ok": True, "removed_sec": round(removed, 3)}

    def log(self, rec: dict) -> None:
        with self.log_path.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    def handle(self, command: str, query: dict, body: bytes, ctype: str) -> tuple[int, str, bytes]:
        """Returns (status, content type, body)."""
        if self.fwd_started is not None and self.fwd_disturbed is None:
            self.fwd_disturbed = time.time()  # the solver is doing something else: the pause ends here
        with self.lock:
            e = self.engine
            t0 = time.time()
            status, out_type, out = 200, "application/json", b""
            logged: dict = {}
            try:
                if command not in COMMANDS:
                    raise ToolError(f"unknown command {command!r}")
                if command != "oracle-restore":
                    e.touch()
                req: dict = {}
                if ctype.startswith("application/json") and body:
                    try:
                        req = json.loads(body)
                    except json.JSONDecodeError as exc:
                        raise ToolError(f"invalid JSON: {exc}") from None
                    if not isinstance(req, dict):
                        raise ToolError("the body must be a JSON object")
                if command == "tensor":
                    name, which = req.get("name"), req.get("which", "current")
                    logged = {"name": name, "which": which}
                    out_type, out = "application/x-npy", npy_bytes(e.get_tensor(str(name), str(which)))
                elif command == "data":
                    out_type, out = "application/x-npz", npz_bytes(e.dev_arrays())
                elif command == "save_edit":
                    name = (query.get("name") or [""])[0]
                    logged = {"name": name}
                    try:
                        arr = np.load(io.BytesIO(body), allow_pickle=False)
                    except Exception:
                        raise ToolError("save_edit needs the array as .npy bytes") from None
                    if not isinstance(arr, np.ndarray):
                        raise ToolError("save_edit needs a single array (.npy), not an archive")
                    res = e.save_edit(name, arr)
                    logged["changed"] = res["changed_elements"]
                    out = json.dumps({"ok": True, "result": res}).encode()
                elif command == "forward":
                    inputs = None
                    if ctype.startswith("application/x-npz"):
                        try:
                            z = np.load(io.BytesIO(body), allow_pickle=False)
                            req = json.loads(bytes(z["__request__"]).decode())
                            inputs = {k[len("input:"):]: z[k] for k in z.files if k.startswith("input:")}
                        except Exception:
                            raise ToolError("could not read the forward request archive") from None
                    logged = {"request": req, "custom_inputs": sorted(inputs) if inputs else None}
                    self.fwd_started, self.fwd_disturbed = time.time(), None
                    try:
                        summary, arrays = e.forward(req, inputs or None)
                    finally:
                        p_end = self.fwd_disturbed or time.time()
                        with self.plock:
                            paused = self._credit(self.fwd_started, p_end)
                            self.passes.append([self.fwd_started, p_end, paused])
                        self.fwd_started = None
                        e.paused_sec += paused
                        logged["clock_paused_sec"] = round(paused, 3)
                    if "usage" in summary:  # report the clock with this pass's pause credited
                        summary["usage"] = e.usage()
                    logged["summary"] = {k: v for k, v in summary.items() if k != "arrays"}
                    out_type = "application/x-npz"
                    out = npz_bytes({"__summary__": json_array(summary), **arrays})
                else:
                    logged = {k: v for k, v in req.items() if k != "token"}
                    if command == "info":
                        res = e.info()
                    elif command == "usage":
                        res = e.usage()
                    elif command == "edited":
                        res = e.edited()
                    elif command == "drop_edit":
                        res = e.drop_edit(str(req.get("name")))
                    else:  # oracle-restore
                        res = e.oracle_restore(req)
                    out = json.dumps({"ok": True, "result": res}).encode()
            except ToolError as exc:
                e.rejected_requests += 1
                status, out_type = 400, "application/json"
                out = json.dumps({"ok": False, "error": str(exc), "usage": e.usage()}).encode()
            except Exception as exc:  # the solver never sees internals
                traceback.print_exc()
                status, out_type = 500, "application/json"
                out = json.dumps({"ok": False, "error": f"internal error ({type(exc).__name__})"}).encode()
            if command in EXPORT_AFTER and status == 200:
                self.export(final=False)
            self.log({
                "t": round(t0 - e.started_at, 3),
                "duration_sec": round(time.time() - t0, 3),
                "command": command,
                "request": logged,
                "ok": status == 200,
                "error": json.loads(out).get("error") if status != 200 else None,
                "usage": e.usage(),
            })
            return status, out_type, out

    def export(self, final: bool) -> None:
        """Write the signed bundle of the current edits. There is no separate submit step."""
        e = self.engine
        delta = e.delta()
        payload = {
            "task_id": self.episode["task_id"],
            "model_id": e.model_id,
            "final": final,
            "forward_passes": e.forward_passes,
            "rejected_requests": e.rejected_requests,
            "edited_params": sum(int(v.numel()) for k, v in delta.items() if k.endswith("::idx")),
            "total_params": e.total_params,
            "max_edit_fraction": e.max_edit_fraction,
            "max_forward_passes": e.max_forward_passes,
            "time_limit_sec": e.time_limit_sec,
            "agent_active_sec": round(e.time_used(), 3),
            "clock_paused_sec": round(e.paused_sec, 3),
            "history": e.history,
        }
        write_bundle(self.export_dir, delta, payload, self.episode["hmac_key"])


def make_handler(backend: Backend):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, ctype: str, data: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _json(self, status: int, body: dict) -> None:
            self._send(status, "application/json", json.dumps(body).encode())

        def do_GET(self):
            if self.path == "/health":
                return self._json(200, {"ok": True})
            self._json(404, {"ok": False, "error": "not found"})

        def do_POST(self):
            url = urlparse(self.path)
            try:
                n = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return self._json(400, {"ok": False, "error": "bad Content-Length"})
            if n > MAX_BODY:
                return self._json(413, {"ok": False, "error": f"request larger than {MAX_BODY // 2**20} MB"})
            body = self.rfile.read(n) if n else b""
            if url.path == "/internal/finalize":  # the collect hook, from inside this container
                try:
                    key = json.loads(body or b"{}").get("key")
                except (json.JSONDecodeError, AttributeError):
                    key = None
                if key != backend.episode["hmac_key"]:
                    return self._json(404, {"ok": False, "error": "not found"})
                with backend.lock:
                    backend.export(final=True)
                return self._json(200, {"ok": True})
            if url.path == "/agent/active":
                # outside /v1/: not a tool call, so it neither touches the clock nor ends a pause
                try:
                    q = json.loads(body or b"{}")
                    return self._json(200, backend.agent_active(q["start"], q["end"]))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    return self._json(400, {"ok": False, "error": "needs {start, end}"})
            if not url.path.startswith("/v1/"):
                return self._json(404, {"ok": False, "error": "not found"})
            ctype = self.headers.get("Content-Type", "application/json")
            status, out_type, data = backend.handle(url.path[4:], parse_qs(url.query), body, ctype)
            self._send(status, out_type, data)

        def log_message(self, fmt, *args):
            pass

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", default="/srv/episode.json")
    ap.add_argument("--export", default=os.environ.get("BTW_EXPORT", "/srv/export"))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    backend = Backend(Path(args.episode), Path(args.export))
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(backend))
    print(f"backend ready on {args.host}:{args.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
