"""The network's weights as a Python API: the prelude of the `py` kernel, importable from any python3.

    NAMES, SHAPES, BUFFERS, MODULES, INFO      what the network is made of
    load(name), load_original(name)            float32 arrays (your edit if any / the weights as given)
    save_edit(name, array), drop_edit(name)    replace a whole tensor / undo that
    edited(), usage()                          your edits; budgets and time
    load_dev()                                 the dev examples (inputs and targets)
    load_pass(k)                               the arrays saved by forward pass k

Forward passes are not part of this API: they run only through the `forward` shell command, one
deliberate request at a time (as the forward-pass blocks of the Experiment-8 harness).

It talks to the trusted backend over HTTP and needs only the standard library and numpy.
"""

from __future__ import annotations

import io
import json
import os
import socket
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

BACKEND = os.environ.get("WSP_BACKEND", "http://backend:8000")
WORK = os.environ.get("WSP_WORK", "/work")

__all__ = ["np", "NAMES", "SHAPES", "BUFFERS", "MODULES", "INFO", "load", "load_original", "save_edit", "drop_edit",
           "edited", "usage", "load_dev", "load_pass", "BackendError"]


class BackendError(RuntimeError):
    """A request the backend refused (the message says why), or an unreachable backend."""


def _unreachable(exc: Exception) -> bool:
    """Connection refused or name not resolved yet: safe to retry, nothing reached the backend."""
    reason = getattr(exc, "reason", exc)
    return isinstance(reason, (ConnectionRefusedError, FileNotFoundError, socket.gaierror))


def _request(command: str, body: bytes = b"", ctype: str = "application/json", query: dict | None = None,
             timeout: float = 3600) -> tuple[str, bytes]:
    url = f"{BACKEND}/v1/{command}" + ("?" + urllib.parse.urlencode(query) if query else "")
    for attempt in range(60):
        req = urllib.request.Request(url, data=body, headers={"Content-Type": ctype}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.headers.get("Content-Type", ""), r.read()
        except urllib.error.HTTPError as e:
            raw = e.read()
            try:
                msg = json.loads(raw).get("error") or raw[:300].decode(errors="replace")
            except ValueError:
                msg = raw[:300].decode(errors="replace")
            raise BackendError(msg) from None
        except (urllib.error.URLError, OSError) as exc:
            if _unreachable(exc) and attempt < 59:
                time.sleep(2)
                continue
            raise BackendError(f"backend unreachable at {BACKEND}: {exc}") from None
    raise BackendError(f"backend unreachable at {BACKEND}")


def _json(command: str, payload: dict | None = None) -> dict:
    _, raw = _request(command, json.dumps(payload or {}).encode())
    res = json.loads(raw)
    if not res.get("ok"):
        raise BackendError(res.get("error", "request failed"))
    return res["result"]


# ----------------------------------------------------------------------------- what the network is

_INFO: dict | None = None


def _info() -> dict:
    global _INFO
    if _INFO is None:
        _INFO = _json("info")
    return _INFO


def __getattr__(name: str):
    if name == "INFO":
        return _info()
    if name == "NAMES":
        return [t["name"] for t in _info()["tensors"]]
    if name == "SHAPES":
        return {t["name"]: tuple(t["shape"]) for t in _info()["tensors"]} | {
            b["name"]: tuple(b["shape"]) for b in _info()["buffers"]}
    if name == "BUFFERS":
        return [b["name"] for b in _info()["buffers"]]
    if name == "MODULES":
        return {m["name"]: m["type"] for m in _info()["modules"]}
    raise AttributeError(name)


# ----------------------------------------------------------------------------- weights

_ORIGINAL: dict[str, np.ndarray] = {}


def _tensor(name: str, which: str) -> np.ndarray:
    _, raw = _request("tensor", json.dumps({"name": name, "which": which}).encode())
    return np.load(io.BytesIO(raw), allow_pickle=False)


def load(name: str) -> np.ndarray:
    """The tensor as it is now: your saved edit if there is one, else the weights as given (float32 copy).
    Buffers (read-only) load the same way."""
    return _tensor(name, "current")


def load_original(name: str) -> np.ndarray:
    """The tensor as you were given it, before any edit of yours (float32 copy)."""
    if name not in _ORIGINAL:
        _ORIGINAL[name] = _tensor(name, "original")
    return _ORIGINAL[name].copy()


def save_edit(name: str, array) -> dict:
    """Replace the whole tensor `name` with `array` (same shape). Returns the number of elements that now
    differ from the weights as given, and your budgets."""
    a = np.asarray(array)
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(a, dtype=np.float32) if a.dtype != np.float32 else a, allow_pickle=False)
    _, raw = _request("save_edit", buf.getvalue(), "application/x-npy", {"name": name})
    return json.loads(raw)["result"]


def drop_edit(name: str) -> dict:
    """Undo your edit of `name` (back to the weights as given); its edit budget is returned."""
    return _json("drop_edit", {"name": name})


def edited() -> dict:
    """{tensor name: number of elements that differ from the weights as given} for every edited tensor."""
    return _json("edited")["edited"]


def usage() -> dict:
    """Forward passes, edit budget and time: used and remaining."""
    return _json("usage")


def _clock_line(u: dict | None = None) -> str:
    """One line with the session clock and the forward-pass count, for the end of every `py` run and `forward`
    command (on stderr, so a grep or head on the output does not hide it). Empty if the backend cannot be reached."""
    try:
        if u is None:  # one short attempt: the footer must never hold up a run (usage() retries for minutes)
            req = urllib.request.Request(f"{BACKEND}/v1/usage", data=b"{}", method="POST",
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=5) as r:
                res = json.loads(r.read())
            if not res.get("ok"):
                return ""
            u = res["result"]
        u = u.get("usage", u)
        parts = []
        limit = float(u.get("time_limit_sec") or 0)
        if limit:
            used = min(float(u.get("time_used_sec") or 0), limit)
            mmss = lambda s: f"{int(s) // 60}:{int(s) % 60:02d}"
            parts.append(f"elapsed {mmss(used)} of {mmss(limit)}")
        if "forward_passes_used" in u:
            parts.append(f"forward passes {u['forward_passes_used']}/{u['forward_passes_max']}")
        return "[" + " | ".join(parts) + "]" if parts else ""
    except Exception:
        return ""


# ----------------------------------------------------------------------------- data

def load_dev() -> dict[str, np.ndarray]:
    """The dev examples: the inputs the network reads and the targets (see INFO["input"], INFO["target"])."""
    path = os.path.join(WORK, "dev.npz")
    if not os.path.exists(path):
        _, raw = _request("data", b"{}")
        os.makedirs(WORK, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(raw)
        os.replace(tmp, path)
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# ----------------------------------------------------------------------------- forward passes

def _fname(key: str) -> str:
    return key.replace("/", "_") + ".npy"


def _forward(request: dict, inputs: dict | None = None) -> dict:
    """One forward pass, for the `forward` command (not exported to the kernel). The summary is returned; the
    arrays are saved to WORK/fwd_<k>/ (one .npy per array) and can be loaded with load_pass(k)."""
    if inputs:
        arrays = {"__request__": np.frombuffer(json.dumps(request).encode(), dtype=np.uint8)}
        arrays.update({f"input:{k}": np.asarray(v) for k, v in inputs.items()})
        buf = io.BytesIO()
        np.savez(buf, **arrays)
        _, raw = _request("forward", buf.getvalue(), "application/x-npz")
    else:
        _, raw = _request("forward", json.dumps(request).encode())
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        summary = json.loads(bytes(z["__summary__"]).decode())
        out_dir = os.path.join(WORK, f"fwd_{summary['pass']}")
        os.makedirs(out_dir, exist_ok=True)
        listing = {}
        for k in z.files:
            if k != "__summary__":
                a = z[k]
                np.save(os.path.join(out_dir, _fname(k)), a, allow_pickle=False)
                listing[k] = {"shape": list(a.shape), "dtype": str(a.dtype)}
    summary["arrays"] = listing
    summary["dir"] = out_dir
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    return summary


def load_pass(k: int) -> dict[str, np.ndarray]:
    """The arrays of forward pass k, by name (as listed in its summary)."""
    d = os.path.join(WORK, f"fwd_{k}")
    with open(os.path.join(d, "summary.json")) as f:
        names = list(json.load(f)["arrays"])
    return {n: np.load(os.path.join(d, _fname(n)), allow_pickle=False) for n in names}
