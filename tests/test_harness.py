"""End-to-end tests of the harness on a small synthetic MLP: no downloads, no Docker.

The clean network labels its own random inputs, so it scores 100%; the starting weights swap two rows of
the output layer, and restoring them is an exact repair.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
CLIENT = ROOT / "src/benchtheweights/client"
MODEL_ID, DATASET_ID = "tiny-mlp", "tiny-digits"
ROWS = (2, 7)


@pytest.fixture(scope="session")
def problem(tmp_path_factory):
    """A registered tiny MLP with its checkpoint, splits and starting weights in a temporary cache."""
    cache = tmp_path_factory.mktemp("cache")
    os.environ["BTW_CACHE"] = str(cache)
    from benchtheweights import models as M
    from benchtheweights.registry import DATASETS, MODELS, DatasetSpec, ModelSpec

    DATASETS[DATASET_ID] = DatasetSpec(DATASET_ID, "none", "none", "none", "mnist", 10, n_dev=200, n_hidden=400)
    MODELS[MODEL_ID] = ModelSpec(MODEL_ID, "mlp", "3-layer MLP (784-32-32-10, ReLU)", "Digit classification (synthetic)",
                                 "none", "none", "model.safetensors", DATASET_ID, "mnist_mlp", extra={"hidden_size": 32})
    torch.manual_seed(0)
    net = M.MnistMLP(32)
    ckpt = cache / "checkpoints" / MODEL_ID
    ckpt.mkdir(parents=True)
    save_file({k: v.contiguous() for k, v in net.state_dict().items()}, str(ckpt / "model.safetensors"))
    model = M.build_model(MODEL_ID)
    for split, n in (("dev", 200), ("hidden", 400)):
        images = torch.randint(0, 256, (n, 28, 28), dtype=torch.uint8)
        data = {"images": images, "labels": torch.zeros(n, dtype=torch.long)}
        data["labels"] = M.predict(model, MODEL_ID, data).argmax(-1)
        (cache / "data" / DATASET_ID).mkdir(parents=True, exist_ok=True)
        torch.save(data, cache / "data" / DATASET_ID / f"{split}.pt")
    start = {k: v.clone().contiguous() for k, v in M.editable_parameters(model).items()}
    w = start["output_layer.weight"]
    w[list(ROWS)] = w[list(reversed(ROWS))].clone()
    start_path = cache / "start.safetensors"
    save_file(start, str(start_path))
    return {"cache": cache, "start": start_path}


def episode(problem, tmp_path, minutes: float = 0, passes: int = 10) -> Path:
    import hashlib

    ep = {"task_id": "benchtheweights/test", "model_id": MODEL_ID, "start_weights": str(problem["start"]),
          "limits": {"max_forward_passes": passes, "max_edit_fraction": 0.05, "time_limit_sec": 60 * minutes},
          "hmac_key": "k" * 64, "oracle_token_sha256": hashlib.sha256(b"token").hexdigest(), "public_scores": {}}
    p = tmp_path / "episode.json"
    p.write_text(json.dumps(ep))
    return p


@pytest.fixture
def backend(problem, tmp_path):
    from benchtheweights.backend.server import Backend

    return Backend(episode(problem, tmp_path), tmp_path / "export")


def call(b, command, req=None):
    status, _, out = b.handle(command, {}, json.dumps(req or {}).encode(), "application/json")
    return status, json.loads(out)


def forward(b, req):
    import io

    status, ctype, out = b.handle("forward", {}, json.dumps(req).encode(), "application/json")
    if ctype != "application/x-npz":
        return status, json.loads(out), None
    z = np.load(io.BytesIO(out))
    return status, json.loads(bytes(z["__summary__"]).decode()), {k: z[k] for k in z.files}


def save(b, name, arr):
    import io

    buf = io.BytesIO()
    np.save(buf, arr.astype(np.float32))
    status, _, out = b.handle("save_edit", {"name": [name]}, buf.getvalue(), "application/x-npy")
    return status, json.loads(out)


def test_start_weights_and_info(backend):
    _, res = call(backend, "info")
    info = res["result"]
    assert info["total_params"] == backend.engine.total_params
    assert [t["name"] for t in info["tensors"]][-1] == "output_layer.bias"
    _, s, _ = forward(backend, {"type": "evaluate"})
    assert s["edited"]["score"] < 1.0


def test_edit_accounting_budget_and_drop(backend):
    e = backend.engine
    w = e.get_tensor("output_layer.weight", "original")
    w2 = w.copy()
    w2[0, :5] += 1
    status, res = save(backend, "output_layer.weight", w2)
    assert status == 200 and res["result"]["changed_elements"] == 5
    big = e.get_tensor("input_layer.weight", "original") + 1
    status, res = save(backend, "input_layer.weight", big)
    assert status == 400 and "budget" in res["error"]
    status, res = save(backend, "output_layer.weight", np.full_like(w, np.nan))
    assert status == 400
    call(backend, "drop_edit", {"name": "output_layer.weight"})
    assert e.edited_numel == 0


def test_forward_passes_reads_and_budget(problem, tmp_path):
    from benchtheweights.backend.server import Backend

    b = Backend(episode(problem, tmp_path, passes=3), tmp_path / "export")
    _, s, arrays = forward(b, {"type": "read", "modules": ["mid_layer"], "module_inputs": True, "examples": [0, 1, 2]})
    assert arrays["mid_layer:out"].shape == (3, 32) and arrays["mid_layer:in"].shape == (3, 32)
    _, s, arrays = forward(b, {"type": "read", "modules": ["mid_layer"], "reduce": "channel_stats"})
    assert arrays["mid_layer:out:mean"].shape == (32,)
    status, s, _ = forward(b, {"type": "bogus"})
    assert status == 400 and b.engine.forward_passes == 2
    forward(b, {"type": "evaluate", "model": "both"})
    status, s, _ = forward(b, {"type": "evaluate"})
    assert status == 400 and "budget" in s["error"]


def test_clock_pause_and_agent_activity(backend):
    from benchtheweights import models as M

    slow = M.predict

    def predict(*a, **k):
        time.sleep(0.3)
        return slow(*a, **k)

    M.predict = predict
    try:
        forward(backend, {"type": "evaluate"})
    finally:
        M.predict = slow
    paused = backend.engine.paused_sec
    assert paused >= 0.3
    start, end = backend.passes[-1][:2]
    backend.agent_active(start, start + 0.1)
    assert backend.engine.paused_sec == pytest.approx(paused - 0.1, abs=1e-6)


def test_time_limit(problem, tmp_path):
    from benchtheweights.backend.server import Backend

    b = Backend(episode(problem, tmp_path, minutes=0.001), tmp_path / "export")
    call(b, "usage")
    time.sleep(0.1)
    status, s, _ = forward(b, {"type": "evaluate"})
    assert status == 400 and "time limit" in s["error"]


def spec_for(problem, backend, hidden_clean=1.0):
    from benchtheweights import models as M

    model, _ = M.load_start(MODEL_ID, problem["start"])
    hidden = M.load_split(DATASET_ID, "hidden")
    ep = backend.episode
    return {"task_id": ep["task_id"], "model_id": MODEL_ID, "start_weights": str(problem["start"]),
            "limits": ep["limits"], "hmac_key": ep["hmac_key"],
            "scores": {"hidden_clean": hidden_clean, "hidden_corrupted": M.score(model, MODEL_ID, hidden)}}


def test_oracle_and_verifier(problem, backend, tmp_path):
    from benchtheweights.verifier import verify

    spec = spec_for(problem, backend)
    assert verify(spec, tmp_path / "none")["details"]["status"] == "missing_submission"
    status, res = call(backend, "oracle-restore", {"token": "wrong"})
    assert status == 400
    status, res = call(backend, "oracle-restore", {"token": "token"})
    assert status == 200 and res["result"]["restored"] == ["output_layer.weight"]
    backend.export(final=True)
    out = verify(spec, backend.export_dir)
    assert out["reward"]["recovery"] == pytest.approx(1.0) and out["details"]["determinism_ok"]
    sub = backend.export_dir / "submission.json"
    payload = json.loads(sub.read_text())
    payload["forward_passes"] += 1
    sub.write_text(json.dumps(payload))
    out = verify(spec, backend.export_dir)
    assert out["details"]["status"].startswith("invalid_submission") and out["reward"]["recovery"] == 0


@pytest.fixture
def live(backend, tmp_path):
    from benchtheweights.backend.server import make_handler

    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(backend))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    env = dict(os.environ, WSP_BACKEND=f"http://127.0.0.1:{srv.server_address[1]}", WSP_WORK=str(tmp_path / "work"),
               WSP_KERNEL_SOCK=str(tmp_path / "k.sock"), WSP_KERNEL_LOG=str(tmp_path / "kernel.jsonl"), PYTHONPATH=str(CLIENT))
    yield env
    subprocess.run([sys.executable, str(CLIENT / "kernel.py"), "--restart"], env=env, capture_output=True)
    srv.shutdown()


def test_kernel_and_forward_cli(live):
    py = [sys.executable, str(CLIENT / "kernel.py")]
    r = subprocess.run(py + ["-c", "W = load(NAMES[-2]); W.shape"], env=live, capture_output=True, text=True, timeout=120)
    assert "(10, 32)" in r.stdout
    r = subprocess.run(py + ["-c", "W[2], W[7] = W[7].copy(), W[2].copy(); save_edit(NAMES[-2], W)['changed_elements']"],
                       env=live, capture_output=True, text=True, timeout=60)
    assert r.stdout.strip() == "64"
    r = subprocess.run([sys.executable, str(CLIENT / "forward_cli.py"), '{"type": "evaluate", "model": "both"}'],
                       env=live, capture_output=True, text=True, timeout=60)
    summary = json.loads(r.stdout)
    assert summary["edited"]["score"] == 1.0 and "forward passes 1/10" in r.stderr
    r = subprocess.run(py + ["-c", "sorted(load_pass(1))"], env=live, capture_output=True, text=True, timeout=60)
    assert "edited_outputs" in r.stdout
    r = subprocess.run(py + ["--timeout", "1", "-c", "import time; time.sleep(5)"], env=live, capture_output=True, text=True, timeout=60)
    assert "run stopped" in r.stdout


def test_write_task(problem, tmp_path, monkeypatch):
    import argparse

    from benchtheweights import task

    monkeypatch.setattr(task, "START", str(problem["start"]))
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"model_id_in_registry": MODEL_ID}))
    out = tmp_path / "hb-test"
    task.write_task(argparse.Namespace(meta=str(meta), out=str(out), problem_id="hb-test", image_prefix="btw-hb-test",
                                       tag="t", agent_image="btw-agent:t", passes=50, edit=0.03, minutes=25,
                                       backend_mem="6g", verifier_mem=8192))
    text = (out / "instruction.md").read_text()
    assert "{{" not in text and "corrupt" not in text.lower()
    assert "**25 minutes**" in text and "50 passes" in text
    toml = (out / "task.toml").read_text()
    assert "timeout_sec = 7500.0" in toml
    spec = json.loads((out / "tests/spec.json").read_text())
    assert spec["changed_params"] == 64
    assert spec["hmac_key"] not in text and spec["hmac_key"] not in (out / "environment/Dockerfile").read_text()


@pytest.mark.parametrize("harness", ["claude_code", "codex"])
def test_harness_conversion(problem, tmp_path, monkeypatch, harness):
    import importlib.util

    import yaml

    test_write_task(problem, tmp_path, monkeypatch)
    sys.path.insert(0, str(ROOT / "harnesses"))
    import common

    monkeypatch.setattr(common, "build", lambda dockerfile, tag, *a, **k: tag)
    spec = importlib.util.spec_from_file_location(f"make_{harness}", ROOT / "harnesses" / harness / "make_task.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    providers = ["openai", "openrouter"] if harness == "codex" else [None]
    for provider in providers:
        out = tmp_path / f"out-{provider}"
        argv = ["make_task.py", str(tmp_path / "hb-test"), "--out", str(out)] + (["--provider", provider] if provider else [])
        monkeypatch.setattr(sys, "argv", argv)
        mod.main()
        compose = yaml.safe_load((out / "hb-test/environment/docker-compose.yaml").read_text())
        svc = compose["services"]
        assert svc["main"]["networks"] == ["repair"] and compose["networks"]["repair"]["internal"]
        assert svc["main"]["environment"]["HTTPS_PROXY"] == "http://egress:3128"
        allow = svc["egress"]["environment"]["EGRESS_ALLOW"]
        if harness == "claude_code":
            assert allow == "api.anthropic.com:443"
            assert svc["main"]["environment"]["CLAUDE_CODE_DISABLE_BACKGROUND_TASKS"] == "1"
        elif provider == "openrouter":
            assert allow == "openrouter.ai:443" and "relay" in svc
            assert "requirements.toml" in (out / "hb-test/environment/Dockerfile").read_text()
        else:
            assert "chatgpt.com:443" in allow and "relay" not in svc
        assert '"/var/log/egress"' in (out / "hb-test/task.toml").read_text()
        assert (out / "hb-test/instruction.md").read_text() == (tmp_path / "hb-test/instruction.md").read_text()
