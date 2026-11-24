#!/usr/bin/env python3
"""`py`: a persistent Python kernel over the network's weights (variables survive between runs).

    py <<'EOF'            run the code on stdin
    ...
    EOF
    py -c 'CODE'          run CODE
    py FILE.py            run a file
    py --restart          restart the kernel (all variables are lost)

The kernel starts on first use with the prelude `import numpy as np; from wsp import *` (see wsp.py).
If the last statement of a run is an expression, its value is printed, as in a notebook. Each run may
take up to --timeout seconds (default 300): a run that takes longer is stopped and the kernel keeps
its variables. Output is cut at 10,000 characters. Every run is logged.

Needs only the standard library (and numpy, for the prelude).
"""

from __future__ import annotations

import argparse
import ast
import fcntl
import hashlib
import io
import json
import os
import signal
import socket
import struct
import subprocess
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
SOCK = os.environ.get("WSP_KERNEL_SOCK", "/tmp/wsp-kernel.sock")
if len(SOCK.encode()) > 100:  # AF_UNIX paths are limited to ~108 bytes: use a short, stable alias
    SOCK = f"/tmp/wsp-kernel-{hashlib.sha256(SOCK.encode()).hexdigest()[:16]}.sock"
WORK = os.environ.get("WSP_WORK", "/work")
MAX_OUT = int(os.environ.get("WSP_MAX_OUTPUT", "10000"))
DEFAULT_TIMEOUT = float(os.environ.get("WSP_RUN_TIMEOUT", "300"))
PRELUDE = "import numpy as np\nnp.set_printoptions(precision=4, suppress=True, linewidth=160)\nfrom wsp import *\n"


def _log_path() -> str:
    p = os.environ.get("WSP_KERNEL_LOG")
    if p:
        return p
    if os.path.isdir("/logs/agent") and os.access("/logs/agent", os.W_OK):
        return "/logs/agent/kernel_log.jsonl"
    return os.path.join(WORK, ".kernel_log.jsonl")


# ------------------------------------------------------------------------------ wire format

def _send(conn: socket.socket, obj: dict) -> None:
    data = json.dumps(obj).encode()
    conn.sendall(struct.pack(">Q", len(data)) + data)


def _recv(conn: socket.socket) -> dict:
    def exactly(n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = conn.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("kernel closed the connection")
            buf += chunk
        return buf

    (n,) = struct.unpack(">Q", exactly(8))
    return json.loads(exactly(n))


# ------------------------------------------------------------------------------ server (the kernel)

class _RunTimeout(BaseException):
    """Raised inside a run when its time is up (BaseException: `except Exception` in user code won't eat it)."""


def _on_alarm(signum, frame):
    raise _RunTimeout()


def _cut(text: str) -> str:
    if len(text) > MAX_OUT:
        return text[:MAX_OUT] + f"\n[output cut at {MAX_OUT:,} characters]"
    return text


def _run(code: str, ns: dict, timeout: float) -> tuple[str, str]:
    buf = io.StringIO()
    status = "ok"
    signal.signal(signal.SIGALRM, _on_alarm)
    with redirect_stdout(buf), redirect_stderr(buf):
        try:
            signal.setitimer(signal.ITIMER_REAL, timeout)
            tree = ast.parse(code, "<py>", "exec")
            last = None
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last = ast.Expression(tree.body.pop().value)
            exec(compile(tree, "<py>", "exec"), ns)
            if last is not None:
                value = eval(compile(last, "<py>", "eval"), ns)
                if value is not None:
                    print(repr(value))
        except _RunTimeout:
            status = "timeout"
            print(f"[run stopped after {timeout:g} s (the per-run time limit); the kernel keeps its variables]")
        except SystemExit:
            print("[SystemExit ignored: the kernel keeps running]")
        except BaseException:  # noqa: BLE001 - report everything to the user, keep the kernel alive
            status = "error"
            etype, value, tb = sys.exc_info()
            # keep the user's frames, drop the kernel's and the client's (they would also reveal its install path)
            frames = [f for f in traceback.extract_tb(tb) if not f.filename.startswith(HERE + os.sep)]
            print("Traceback (most recent call last):\n" + "".join(traceback.format_list(frames))
                  + "".join(traceback.format_exception_only(etype, value)), end="")
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
    return buf.getvalue(), status


def serve() -> None:
    sys.path.insert(0, HERE)
    os.makedirs(WORK, exist_ok=True)
    os.chdir(WORK)
    ns: dict = {"__name__": "__main__", "__builtins__": __builtins__}
    prelude_out, prelude_status = _run(PRELUDE, ns, 600)
    if os.path.exists(SOCK):
        os.unlink(SOCK)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SOCK)
    srv.listen(8)
    with open(SOCK + ".pid", "w") as f:
        f.write(str(os.getpid()))
    log = _log_path()
    pending_note = "" if prelude_status == "ok" else "[the prelude failed; `from wsp import *` is not loaded]\n" + prelude_out
    while True:
        conn, _ = srv.accept()
        with conn:
            try:
                req = _recv(conn)
            except (ConnectionError, ValueError):
                continue
            t0 = time.time()
            out, status = _run(req["code"], ns, float(req.get("timeout", DEFAULT_TIMEOUT)))
            out = pending_note + out
            pending_note = ""
            sec = round(time.time() - t0, 2)
            try:
                with open(log, "a") as f:
                    f.write(json.dumps({"t": round(t0, 3), "seconds": sec, "status": status, "code": req["code"],
                                        "output": out[:6000]}) + "\n")
            except OSError:
                pass
            try:
                _send(conn, {"output": _cut(out), "status": status, "seconds": sec})
            except OSError:
                pass


# ------------------------------------------------------------------------------ client

def _connect(timeout: float | None) -> socket.socket | None:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(SOCK)
        return s
    except OSError:
        s.close()
        return None


def _kernel_pid() -> int | None:
    try:
        return int(open(SOCK + ".pid").read().strip())
    except (OSError, ValueError):
        return None


def _kill() -> None:
    pid = _kernel_pid()
    if pid:
        try:
            os.killpg(pid, signal.SIGKILL)
        except OSError:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
    for p in (SOCK, SOCK + ".pid"):
        try:
            os.unlink(p)
        except OSError:
            pass


def _ensure_kernel() -> str:
    """Start the kernel if it is not running. Returns a note for the user ('' if it was running)."""
    s = _connect(5)
    if s:
        s.close()
        return ""
    with open(SOCK + ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        s = _connect(5)
        if s:
            s.close()
            return ""
        _kill()
        os.makedirs(WORK, exist_ok=True)
        err_path = os.path.join(WORK, ".kernel_stderr.log")
        with open(err_path, "ab") as err:
            proc = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--serve"], stdin=subprocess.DEVNULL,
                                    stdout=err, stderr=err, start_new_session=True, env=os.environ.copy())
        deadline = time.time() + 600
        while time.time() < deadline:
            s = _connect(5)
            if s:
                s.close()
                return "[kernel started]\n"
            if proc.poll() is not None:
                break
            time.sleep(0.3)
    try:
        tail = open(err_path, errors="replace").read()[-2000:]
    except OSError:
        tail = ""
    raise SystemExit("the kernel did not start:\n" + tail)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", nargs="?", help="a .py file to run, or - for stdin (default: stdin)")
    ap.add_argument("-c", dest="code", help="code to run")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="seconds per run (default %(default)s)")
    ap.add_argument("--restart", action="store_true", help="restart the kernel; all variables are lost")
    ap.add_argument("--serve", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.serve:
        serve()
        return 0
    if args.restart:
        _kill()
        print("[kernel stopped; it restarts with a fresh namespace on the next run]")
        if args.code is None and args.file is None:
            return 0
    if args.code is not None:
        code = args.code
    elif args.file and args.file != "-":
        with open(args.file) as f:
            code = f.read()
    else:
        if sys.stdin.isatty():
            ap.print_help()
            return 2
        code = sys.stdin.read()
    note = _ensure_kernel()
    s = _connect(None)
    if s is None:
        print(note + "[could not reach the kernel]")
        return 1
    s.settimeout(args.timeout + 120)  # the run stops itself at --timeout; this catches a stuck C call
    try:
        _send(s, {"code": code, "timeout": args.timeout})
        res = _recv(s)
    except (TimeoutError, socket.timeout, ConnectionError, OSError):
        _kill()
        print(note + f"[the kernel did not answer within {args.timeout + 120:g} s and was restarted: "
                     "all variables are lost]")
        return 1
    finally:
        s.close()
    out = note + res["output"]
    print(out, end="" if out.endswith("\n") or not out else "\n")
    _print_clock()
    return 0 if res["status"] == "ok" else 1


def _print_clock() -> None:
    """The session clock after every run, on stderr (see wsp._clock_line)."""
    try:
        sys.stdout.flush()
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import wsp

        line = wsp._clock_line()
        if line:
            print(line, file=sys.stderr, flush=True)
    except Exception:
        pass


if __name__ == "__main__":
    sys.exit(main())
