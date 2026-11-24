#!/usr/bin/env python3
"""`forward`: run one forward pass of the network (counts 1 against the forward-pass budget).

    forward '{"type": "evaluate", "model": "both"}'
    forward request.json
    forward - < request.json
    forward '{"type": "read", "modules": ["layer1.0"]}' --inputs my_inputs.npz

Prints the pass's summary; its arrays are saved to $WSP_WORK/fwd_<k>/ (one .npy per array, plus
summary.json). In the kernel, `load_pass(k)` loads the arrays. Forward passes run only through this command.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import wsp  # noqa: E402


def dump(obj, ind: int = 0) -> str:
    """JSON with one key per line but lists of numbers kept on one line."""
    pad = " " * (ind + 1)
    if isinstance(obj, dict) and obj:
        items = [f"{pad}{json.dumps(k)}: {dump(v, ind + 1)}" for k, v in obj.items()]
        return "{\n" + ",\n".join(items) + "\n" + " " * ind + "}"
    if isinstance(obj, list) and any(isinstance(v, (dict, list)) for v in obj):
        return "[\n" + ",\n".join(pad + dump(v, ind + 1) for v in obj) + "\n" + " " * ind + "]"
    return json.dumps(obj)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="forward", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("request", help="the request as JSON, a .json file, or - for stdin")
    ap.add_argument("--inputs", help="an .npz file of custom inputs (one array per input field) for a `read` pass")
    args = ap.parse_args(argv)
    raw = args.request
    if raw == "-":
        raw = sys.stdin.read()
    elif not raw.lstrip().startswith("{") and os.path.exists(raw):
        with open(raw) as f:
            raw = f.read()
    try:
        req = json.loads(raw)
    except json.JSONDecodeError as e:
        print(json.dumps({"ok": False, "error": f"the request is not valid JSON: {e}"}))
        return 2
    inputs = None
    if args.inputs:
        with np.load(args.inputs, allow_pickle=False) as z:
            inputs = {k: z[k] for k in z.files}
    try:
        summary = wsp._forward(req, inputs)
    except wsp.BackendError as e:
        print(json.dumps({"ok": False, "error": str(e)}))
        _print_clock(None)
        return 1
    print(dump({"ok": True, **summary}))
    _print_clock(summary.get("usage"))
    return 0


def _print_clock(u: dict | None) -> None:
    """The session clock after every pass, on stderr, so a grep or head on the summary does not hide it."""
    sys.stdout.flush()
    line = wsp._clock_line(u)
    if line:
        print(line, file=sys.stderr, flush=True)


if __name__ == "__main__":
    sys.exit(main())
