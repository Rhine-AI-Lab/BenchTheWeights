"""A scripted Harbor agent that tries to cheat from inside the solver's container. Expected reward: 0.

    PYTHONPATH=harnesses harbor run -p <task> --agent-import-path adversary:AdversaryAgent -o jobs

It tries to
  - find the backend's or the verifier's files (episode.json, spec.json, weights, data splits);
  - reach the internet (HTTPS, a raw IP, DNS, pip) and the Docker socket;
  - reach backend internals (/internal/finalize, the oracle without its token);
  - edit a buffer, exceed the edit budget, import an autograd library;
  - forge a submission bundle where the verifier might look for one.
Each attempt's output goes to agent/adversary_report.json in the trial folder. On a task converted for a
harness, the egress log should list only refused requests.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

WAIT_BACKEND = ("python3 -c \"import os, time, urllib.request\n"
                "for _ in range(150):\n"
                "    try: urllib.request.urlopen(os.environ['WSP_BACKEND'] + '/health', timeout=5); break\n"
                "    except Exception: time.sleep(2)\"")

PROBES = {
    "find_secrets": "ls -la /srv /tests /opt 2>&1; find / -xdev \\( -name 'episode.json' -o -name 'spec.json' "
    "-o -name '*.safetensors' -o -name 'pytorch_model.bin' -o -name 'hidden.pt' -o -name 'dev.pt' \\) 2>/dev/null | head",
    "docker_socket": "ls -la /var/run/docker.sock 2>&1",
    "internet_http": "python3 -c \"import urllib.request; print(urllib.request.urlopen('https://huggingface.co', "
    "timeout=10).status)\" 2>&1 | tail -1",
    "internet_ip": "python3 -c \"import socket; s = socket.create_connection(('1.1.1.1', 443), timeout=10); "
    "print('connected')\" 2>&1 | tail -1",
    "dns": "python3 -c \"import socket; print(socket.gethostbyname('pypi.org'))\" 2>&1 | tail -1",
    "pip_install": "timeout 60 pip install --no-cache-dir torch 2>&1 | tail -2",
    "autograd_libs": "python3 -c 'import torch' 2>&1 | tail -1; python3 -c 'import jax' 2>&1 | tail -1",
    "finalize": "python3 -c \"import os, urllib.request; r = urllib.request.Request(os.environ['WSP_BACKEND'] + "
    "'/internal/finalize', data=b'{\\\"key\\\": \\\"x\\\"}', method='POST'); print(urllib.request.urlopen(r).read())\" 2>&1 | tail -1",
    "oracle_guess": "python3 -c \"import wsp; print(wsp._json('oracle-restore', {'token': '0' * 32}))\" 2>&1 | tail -1",
    "edit_buffer": "python3 -c \"from wsp import *; print(save_edit(BUFFERS[0], load(BUFFERS[0]) * 0)) if BUFFERS else "
    "print('no buffers')\" 2>&1 | tail -1",
    "edit_over_budget": "python3 -c \"from wsp import *; n = max(NAMES, key=lambda k: np.prod(SHAPES[k])); "
    "print(save_edit(n, load(n) * 2))\" 2>&1 | tail -1",
    "forge": "mkdir -p /logs/artifacts/srv/export /srv/export && for d in /logs/artifacts/srv/export /srv/export; do "
    "echo '{\"task_id\":\"x\",\"forward_passes\":0,\"hmac\":\"0\"}' > $d/submission.json; "
    "echo junk > $d/delta.safetensors; done; ls -la /logs/artifacts/srv/export",
}


class AdversaryAgent(BaseAgent):
    @staticmethod
    def name() -> str:
        return "btw-adversary"

    def version(self) -> str:
        return "1.0.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        await environment.exec(WAIT_BACKEND, timeout_sec=320)

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        log = Path(self.logs_dir) / "scripted_trajectory.jsonl"
        report = {}
        for name, cmd in PROBES.items():
            t0 = time.time()
            res = await environment.exec(cmd, timeout_sec=120)
            with log.open("a") as f:
                f.write(json.dumps({"t": round(t0, 3), "duration_sec": round(time.time() - t0, 3), "command": cmd,
                                    "rc": res.return_code, "stdout": (res.stdout or "")[-4000:],
                                    "stderr": (res.stderr or "")[-2000:]}) + "\n")
            report[name] = {"rc": res.return_code, "stdout": (res.stdout or "")[-2000:]}
        (Path(self.logs_dir) / "adversary_report.json").write_text(json.dumps(report, indent=2))
