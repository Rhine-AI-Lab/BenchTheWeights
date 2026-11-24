"""The weight-space engine: it holds the network, the solver's edits and the budgets.

The solver works in its own container, in a Python kernel over the weights (`client/wsp.py`): it reads
tensors, analyses them and saves whole-tensor edits. It runs the network only through forward passes on
the dev split, which run here. The solver's code never touches the network or the data, and nothing here
computes gradients.

Accounting is exact: the number of parameter elements that differ from the weights the solver was given,
the number of forward passes, and the working time since the solver's first request.
"""

from __future__ import annotations

import hashlib
import math
import time
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn

from safetensors.torch import load_file

from . import models as M
from .registry import DATASETS, MODELS, SCORE_INFO

# The fields the network reads, per loader, as the solver sees them in `load_dev()` and in custom inputs.
INPUT_FIELDS = {
    "timm_cifar_resnet": ["images"],
    "hf_vit": ["images"],
    "mnist_mlp": ["images"],
    "hf_sequence_classifier": ["input_ids", "attention_mask"],
    "diffusers_unet": ["x_t", "t", "labels"],
    "simulusion_dit": ["x_t", "t", "labels"],
    "diffusers_taesd": ["images"],
    "hf_causal_lm": ["input_ids"],
}
INPUT_DESC = {
    "timm_cifar_resnet": "`images`: uint8 [N, 32, 32, 3] RGB; the network sees them scaled to [0, 1] and "
                         "normalized with the CIFAR mean (0.4914, 0.4822, 0.4465) and std (0.2023, 0.1994, 0.2010)",
    "hf_vit": "`images`: uint8 [N, 32, 32, 3] RGB; the network sees them scaled to [0, 1], bilinearly resized to "
              "224x224 and normalized to [-1, 1] (mean 0.5, std 0.5), in 16x16 patches",
    "mnist_mlp": "`images`: uint8 [N, 28, 28] grayscale digits; the network sees them scaled to [0, 1], normalized "
                 "with mean 0.1307 and std 0.3081, and flattened to 784 values",
    "hf_sequence_classifier": "`input_ids`, `attention_mask`: int64 [N, 66], uncased WordPiece tokens of English "
                              "sentences, padded (the sentences themselves are in `text`)",
    "diffusers_unet": "`x_t`: float32 [N, 3, 32, 32] noisy images and `t`: int64 [N] timesteps in [0, 999] "
                      "(linear beta schedule 1e-4..0.02 over 1000 steps, x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps "
                      "with x_0 in [-1, 1]); `labels` (the image class) is not read by this network",
    "simulusion_dit": "`x_t`: float32 [N, 3, 32, 32] noisy images, `t`: int64 [N] timesteps in [0, 999] "
                      "(linear beta schedule 1e-4..0.02 over 1000 steps, x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps "
                      "with x_0 in [-1, 1]) and `labels`: int64 [N] class labels (the conditioning)",
    "diffusers_taesd": "`images`: uint8 [N, 32, 32, 3] RGB; the network sees them scaled to [-1, 1] and bilinearly "
                       "resized to 64x64 (encoder -> 4x8x8 latent -> decoder)",
    "hf_causal_lm": "`input_ids`: int64 [N, 256], consecutive 256-token windows of English Wikipedia articles "
                    "(WikiText-2 raw), tokenized with the model's own BPE tokenizer (vocabulary of `vocab_size` "
                    "tokens); no padding, every position is a real token",
}
TARGET_DESC = {
    "classification": "`labels`: int64 [N], the true class",
    "denoising": "`eps`: float32 [N, 3, 32, 32], the noise that was added (the network must predict it)",
    "reconstruction": "the input itself: the network must reproduce `images` (resized to 64x64, in [-1, 1])",
    "lm": "the next token: at position i the target is `input_ids[:, i + 1]`. A forward pass reports the "
          "network's outputs as the per-token negative log-likelihood of those targets, float32 [N, 255]; the "
          "full next-token logits are the output of the module `embed_out`",
}

MAX_VALUE_ABS = 1e4
MAX_ITEMS = 1024  # examples or custom inputs per forward pass
MAX_READ_BYTES = 512 * 2**20  # arrays returned by one `read` pass


class ToolError(Exception):
    """An invalid request. The solver gets the message; the backend keeps running."""


def _r(x, nd: int = 6) -> float | None:
    x = float(x)
    return round(x, nd) if math.isfinite(x) else None


def _first_tensor(x):
    """The first tensor in a module's output or input (tuples, dicts and HF ModelOutputs)."""
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, dict):
        x = list(x.values())
    if isinstance(x, (list, tuple)):
        for v in x:
            t = _first_tensor(v)
            if t is not None:
                return t
    return None


def _channel_dim(t: torch.Tensor) -> int:
    """Channels: dim 1 of an NCHW feature map, else the last dim ([N, C] or [N, T, C])."""
    return 1 if t.ndim == 4 else t.ndim - 1


class WeightSpaceEngine:
    def __init__(self, episode: dict):
        M.configure_torch()
        self.episode = episode
        self.model_id = episode["model_id"]
        self.spec = MODELS[self.model_id]
        self.dataset_id = self.spec.dataset_id
        self.objective = self.spec.objective
        self.model: nn.Module = M.build_model(self.model_id)
        self.params = M.editable_parameters(self.model)
        self.set_start(load_file(episode["start_weights"]))
        self.total_params = sum(p.numel() for p in self.params.values())
        lim = episode["limits"]
        self.max_forward_passes = int(lim["max_forward_passes"])
        self.max_edit_fraction = float(lim["max_edit_fraction"])
        self.max_edit_numel = int(self.max_edit_fraction * self.total_params)
        self.time_limit_sec = float(lim.get("time_limit_sec") or 0)  # 0: no limit
        self.forward_passes = 0
        self.rejected_requests = 0
        self.history: list[dict] = []
        self.started_at = time.time()
        self.first_request_at: float | None = None
        self.paused_sec = 0.0  # forward-pass compute that the clock does not count (backend/server.py)
        self.dev = M.load_split(self.dataset_id, "dev")
        self.n_dev = M.num_examples(self.dev)
        self._modules = dict(self.model.named_modules())
        self.buffers = {n: b for n, b in self.model.named_buffers() if b.is_floating_point()}
        self.input_fields = INPUT_FIELDS[self.spec.loader]

    def set_start(self, state: dict[str, torch.Tensor]) -> None:
        """Load the weights the solver starts from, and clear all edits."""
        missing = set(self.params) - set(state)
        if missing:
            raise ValueError(f"starting weights lack {len(missing)} tensors, e.g. {sorted(missing)[:3]}")
        with torch.no_grad():
            for n, p in self.params.items():
                if tuple(state[n].shape) != tuple(p.shape):
                    raise ValueError(f"{n}: shape {tuple(state[n].shape)} != {tuple(p.shape)}")
                p.data.copy_(state[n])
        self.initial = {k: v.detach().clone() for k, v in self.params.items()}
        self.changed: dict[str, int] = {}

    # ------------------------------------------------------------ accounting

    @property
    def edited_numel(self) -> int:
        return sum(self.changed.values())

    def time_used(self) -> float:
        """Working time: since the first request, minus the forward-pass compute the solver waited on."""
        if not self.first_request_at:
            return 0.0
        return max(0.0, time.time() - self.first_request_at - self.paused_sec)

    def usage(self) -> dict:
        u = {
            "forward_passes_used": self.forward_passes,
            "forward_passes_max": self.max_forward_passes,
            "forward_passes_remaining": max(0, self.max_forward_passes - self.forward_passes),
            "edited_params": self.edited_numel,
            "edited_fraction": _r(self.edited_numel / self.total_params, 8),
            "max_edit_params": self.max_edit_numel,
            "max_edit_fraction": self.max_edit_fraction,
            "edit_params_remaining": self.max_edit_numel - self.edited_numel,
            "time_used_sec": round(self.time_used(), 1),
            "clock_paused_sec": round(self.paused_sec, 1),
        }
        if self.time_limit_sec:
            u["time_limit_sec"] = self.time_limit_sec
            u["time_remaining_sec"] = round(max(0.0, self.time_limit_sec - self.time_used()), 1)
        return u

    def touch(self) -> None:
        """Start the clock at the solver's first request."""
        self.first_request_at = self.first_request_at or time.time()

    def check_time(self) -> None:
        if self.time_limit_sec and self.time_used() > self.time_limit_sec:
            raise ToolError(f"time limit reached ({self.time_limit_sec / 60:g} min): no more edits or forward "
                            "passes; the edits saved now are final")

    def _t(self) -> float:
        return round(time.time() - self.started_at, 3)

    # ------------------------------------------------------------ reading

    def info(self) -> dict:
        tensors = [{"name": n, "shape": list(p.shape), "numel": p.numel()} for n, p in self.params.items()]
        metric, metric_desc = SCORE_INFO[self.objective]
        fields = {}
        for k, v in self.dev.items():
            if k == "source_index":
                continue
            fields[k] = ({"shape": list(v.shape), "dtype": str(v.dtype).replace("torch.", "")}
                         if isinstance(v, torch.Tensor) else {"shape": [len(v)], "dtype": "str"})
        info = {
            "architecture": self.spec.architecture,
            "task": self.spec.task,
            "objective": self.objective,
            "metric": metric,
            "metric_definition": metric_desc,
            "input": INPUT_DESC[self.spec.loader],
            "input_fields": self.input_fields,
            "target": TARGET_DESC[self.objective],
            "dev_examples": self.n_dev,
            "dev_fields": fields,
            "total_params": self.total_params,
            "tensors": tensors,
            "buffers": [{"name": n, "shape": list(b.shape)} for n, b in self.buffers.items()],
            "modules": [{"name": n, "type": type(m).__name__} for n, m in self._modules.items() if n],
            "reference_scores": self.episode.get("public_scores", {}),
            "usage": self.usage(),
        }
        if self.objective == "classification":
            info["num_classes"] = DATASETS[self.dataset_id].num_classes
        if self.objective == "lm":
            info["vocab_size"] = int(self.dev["vocab_size"])
        return info

    def get_tensor(self, name: str, which: str = "current") -> np.ndarray:
        if which not in ("current", "original"):
            raise ToolError("`which` must be 'current' or 'original'")
        if name in self.buffers:
            return self.buffers[name].detach().float().numpy().copy()
        if name not in self.params:
            raise ToolError(f"unknown tensor {name!r}; see NAMES (parameters) and BUFFERS")
        t = self.params[name].data if which == "current" else self.initial[name]
        return t.detach().float().numpy().copy()

    def dev_arrays(self) -> dict[str, np.ndarray]:
        out = {}
        for k, v in self.dev.items():
            if k == "source_index":
                continue
            out[k] = v.numpy() if isinstance(v, torch.Tensor) else np.array(list(v), dtype=str)
        return out

    # ------------------------------------------------------------ editing

    def save_edit(self, name: str, arr: np.ndarray) -> dict:
        self.check_time()
        if name in self.buffers:
            raise ToolError(f"{name} is a buffer: buffers are read-only")
        if name not in self.params:
            raise ToolError(f"unknown tensor {name!r}; see NAMES")
        p = self.params[name]
        if tuple(arr.shape) != tuple(p.shape):
            raise ToolError(f"shape {list(arr.shape)} != {list(p.shape)} for {name}")
        if arr.dtype == np.bool_ or not np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.complexfloating):
            raise ToolError(f"the array must hold real numbers, got dtype {arr.dtype}")
        new = torch.from_numpy(np.ascontiguousarray(arr)).to(p.dtype)
        if not torch.isfinite(new).all():
            raise ToolError(f"edit of {name} has non-finite values; rejected")
        if new.numel() and new.abs().max() > MAX_VALUE_ABS:
            raise ToolError(f"edit of {name} has |w| > {MAX_VALUE_ABS:g}; rejected")
        count = int((new != self.initial[name]).sum())
        total = self.edited_numel - self.changed.get(name, 0) + count
        if total > self.max_edit_numel:
            raise ToolError(
                f"edit rejected: {total} parameters ({total / self.total_params:.4%}) would differ from the weights "
                f"you were given; the budget is {self.max_edit_numel} ({self.max_edit_fraction:.2%})")
        with torch.no_grad():
            p.data.copy_(new)
        if count:
            self.changed[name] = count
        else:
            self.changed.pop(name, None)
        self.history.append({"action": "save_edit", "tensor": name, "changed": count, "t": self._t()})
        return {"saved": name, "changed_elements": count, "usage": self.usage()}

    def drop_edit(self, name: str) -> dict:
        self.check_time()
        if name not in self.params:
            raise ToolError(f"unknown tensor {name!r}; see NAMES")
        with torch.no_grad():
            self.params[name].data.copy_(self.initial[name])
        dropped = self.changed.pop(name, 0)
        self.history.append({"action": "drop_edit", "tensor": name, "t": self._t()})
        return {"dropped": name, "had_changed_elements": dropped, "usage": self.usage()}

    def edited(self) -> dict:
        return {"edited": dict(sorted(self.changed.items())), "usage": self.usage()}

    @contextmanager
    def _weights(self, which: str):
        """Run with the solver's edits ('edited') or with the weights as given ('original')."""
        if which == "edited" or not self.changed:
            yield
            return
        saved = {n: self.params[n].data.clone() for n in self.changed}
        try:
            with torch.no_grad():
                for n in saved:
                    self.params[n].data.copy_(self.initial[n])
            yield
        finally:
            with torch.no_grad():
                for n, t in saved.items():
                    self.params[n].data.copy_(t)

    # ------------------------------------------------------------ forward passes

    @staticmethod
    def _subset(data: dict, idx: torch.Tensor) -> dict:
        # 0-d tensors, such as an LM split's `vocab_size`, describe the split, not its examples
        return {k: ((v[idx] if v.ndim else v) if isinstance(v, torch.Tensor) else [v[i] for i in idx.tolist()])
                for k, v in data.items()}

    def _examples(self, req: dict) -> torch.Tensor:
        ex = req.get("examples")
        if ex is None:
            return torch.arange(min(self.n_dev, MAX_ITEMS))
        if isinstance(ex, dict):  # {"start": a, "stop": b}
            try:
                a, b = int(ex.get("start", 0)), int(ex.get("stop", self.n_dev))
            except (TypeError, ValueError):
                raise ToolError("`examples` range needs integer `start` / `stop`") from None
            if not 0 <= a < b <= self.n_dev:
                raise ToolError(f"`examples` range must satisfy 0 <= start < stop <= {self.n_dev}")
            idx = torch.arange(a, b)
        else:
            if not isinstance(ex, list) or not ex or not all(isinstance(i, int) and not isinstance(i, bool) for i in ex):
                raise ToolError("`examples` must be a list of dev indices or {\"start\": a, \"stop\": b}")
            idx = torch.tensor(ex, dtype=torch.long)
            if idx.min() < 0 or idx.max() >= self.n_dev:
                raise ToolError(f"dev indices must be in [0, {self.n_dev})")
        if idx.numel() > MAX_ITEMS:
            raise ToolError(f"at most {MAX_ITEMS} examples per pass")
        return idx

    def _int_range(self, field: str) -> tuple[int, int]:
        if field == "images":
            return 0, 255
        if field == "t":
            return 0, 999
        if field == "labels":
            return 0, DATASETS[self.dataset_id].num_classes - 1
        if field == "attention_mask":
            return 0, 1
        if field == "input_ids":
            emb = self.model.get_input_embeddings() if hasattr(self.model, "get_input_embeddings") else None
            return 0, (emb.num_embeddings - 1 if emb is not None else 2**31 - 1)
        return -(2**31), 2**31 - 1

    def _custom_inputs(self, arrays: dict[str, np.ndarray]) -> dict:
        if set(arrays) != set(self.input_fields):
            raise ToolError(f"custom inputs need exactly the fields {self.input_fields}, got {sorted(arrays)}")
        data, n = {}, None
        for k in self.input_fields:
            ref, a = self.dev[k], arrays[k]
            if a.ndim != ref.ndim or tuple(a.shape[1:]) != tuple(ref.shape[1:]):
                want = ", ".join(["N"] + [str(s) for s in ref.shape[1:]])
                raise ToolError(f"input `{k}` must have shape [{want}], got {list(a.shape)}")
            n = a.shape[0] if n is None else n
            if a.shape[0] != n or not 1 <= n <= MAX_ITEMS:
                raise ToolError(f"all input fields need the same N, with 1 <= N <= {MAX_ITEMS}")
            if ref.dtype.is_floating_point:
                if a.dtype == np.bool_ or not np.issubdtype(a.dtype, np.number) or not np.isfinite(a).all():
                    raise ToolError(f"input `{k}` must hold finite real numbers")
            else:
                if not np.issubdtype(a.dtype, np.integer):
                    raise ToolError(f"input `{k}` must hold integers ({str(ref.dtype).replace('torch.', '')})")
                lo, hi = self._int_range(k)
                if a.min() < lo or a.max() > hi:
                    raise ToolError(f"input `{k}` values must be in [{lo}, {hi}]")
            data[k] = torch.from_numpy(np.ascontiguousarray(a)).to(ref.dtype)
        if "labels" not in data:
            data["labels"] = torch.zeros(n, dtype=torch.long)  # not read by these loaders; batching expects it
        return data

    def forward(self, req: dict, inputs: dict[str, np.ndarray] | None = None) -> tuple[dict, dict[str, np.ndarray]]:
        """One forward pass. Only a pass that succeeds counts against the budget."""
        self.check_time()
        if self.forward_passes >= self.max_forward_passes:
            raise ToolError(f"forward-pass budget exhausted ({self.max_forward_passes} passes)")
        if not isinstance(req, dict):
            raise ToolError("the request must be a JSON object")
        unknown = set(req) - {"type", "model", "examples", "modules", "module_inputs", "outputs", "reduce"}
        if unknown:
            raise ToolError(f"unknown request keys {sorted(unknown)}")
        kind = req.get("type")
        if kind == "evaluate":
            if inputs:
                raise ToolError("`evaluate` runs on dev examples only (custom inputs have no targets); use `read`")
            summary, arrays = self._evaluate(req)
        elif kind == "read":
            summary, arrays = self._read(req, inputs)
        else:
            raise ToolError("`type` must be 'evaluate' or 'read'")
        self.forward_passes += 1
        summary["pass"] = self.forward_passes
        summary["usage"] = self.usage()
        self.history.append({"action": "forward", "type": kind, "model": req.get("model", "edited"),
                             "n": summary.get("n"), "t": self._t()})
        return summary, arrays

    def _per_example_score(self, out: torch.Tensor, data: dict) -> torch.Tensor:
        """Per-example contributions whose mean is exactly the split's score."""
        tgt = M.targets(self.model_id, data)
        if self.objective == "classification":
            return (out.argmax(-1) == tgt).float()
        if self.objective == "lm":
            # outputs are per-token NLLs [N, T-1]; all windows have the same length, so the mean of the
            # per-window scores is exactly the split's score
            return -out.mean(1)
        mse = (out - tgt).pow(2).flatten(1).mean(1)
        if self.objective == "denoising":
            return 1.0 - mse
        return 1.0 - mse / tgt.var()

    @torch.no_grad()
    def _evaluate(self, req: dict) -> tuple[dict, dict]:
        which = req.get("model", "edited")
        if which not in ("edited", "original", "both"):
            raise ToolError("`model` must be 'edited', 'original' or 'both'")
        for k in ("modules", "module_inputs", "outputs", "reduce"):
            if k in req:
                raise ToolError(f"`{k}` belongs to a `read` pass")
        idx = self._examples(req)
        data = self._subset(self.dev, idx)
        models = ["original", "edited"] if which == "both" else [which]
        summary: dict = {"type": "evaluate", "model": which, "n": int(idx.numel()),
                         "metric": SCORE_INFO[self.objective][0]}
        arrays: dict[str, np.ndarray] = {"index": idx.numpy()}
        per: dict[str, torch.Tensor] = {}
        for m in models:
            with self._weights(m):
                out = M.predict(self.model, self.model_id, data)
            s = self._per_example_score(out, data)
            per[m] = s
            summary[m] = self._eval_summary(out, data, s)
            arrays[f"{m}_outputs"] = out.numpy()
            arrays[f"{m}_per_example_score"] = s.numpy()
            if self.objective == "classification":
                nc = DATASETS[self.dataset_id].num_classes
                conf = torch.zeros(nc, nc, dtype=torch.long)
                conf.index_put_((data["labels"], out.argmax(-1)), torch.ones_like(data["labels"]), accumulate=True)
                arrays[f"{m}_confusion"] = conf.numpy()
        if which == "both":
            d = per["edited"] - per["original"]
            n = d.numel()
            se = float(d.std(unbiased=True) / math.sqrt(n)) if n > 1 else float("nan")
            paired = {"difference_edited_minus_original": _r(d.mean()), "standard_error": _r(se)}
            if self.objective == "classification":
                paired["wrong_to_right"] = int(((per["original"] == 0) & (per["edited"] == 1)).sum())
                paired["right_to_wrong"] = int(((per["original"] == 1) & (per["edited"] == 0)).sum())
            summary["paired"] = paired
        return summary, arrays

    def _eval_summary(self, out: torch.Tensor, data: dict, s: torch.Tensor) -> dict:
        res: dict = {"score": _r(s.mean())}
        if self.objective == "lm":
            nll = out.mean()
            res["nll"] = _r(nll)
            res["perplexity"] = _r(nll.exp(), 3)
            res["nll_by_position"] = [  # mean NLL of the targets at positions [lo, lo + 32)
                {"positions": [lo, min(lo + 32, out.shape[1]) - 1], "nll": _r(out[:, lo:lo + 32].mean())}
                for lo in range(0, out.shape[1], 32)
            ]
            return res
        if self.objective == "classification":
            labels = data["labels"]
            nc = DATASETS[self.dataset_id].num_classes
            pred = out.argmax(-1)
            res["nll"] = _r(torch.nn.functional.cross_entropy(out, labels))
            res["mean_confidence"] = _r(out.softmax(-1).max(-1).values.mean(), 4)
            res["per_class_accuracy"] = [_r((pred[labels == c] == c).float().mean(), 4) if (labels == c).any() else None
                                         for c in range(nc)]
            res["predicted_class_histogram"] = torch.bincount(pred, minlength=nc).tolist()
        else:
            tgt = M.targets(self.model_id, data)
            err = (out - tgt).pow(2)
            res["mse"] = _r(err.mean())
            res["mse_per_channel"] = [_r(v) for v in err.mean(dim=(0, 2, 3))]
            res["output_mean_per_channel"] = [_r(v, 5) for v in out.mean(dim=(0, 2, 3))]
            res["output_std_per_channel"] = [_r(v, 5) for v in out.std(dim=(0, 2, 3))]
            if self.objective == "denoising":
                t, per_ex = data["t"], err.flatten(1).mean(1)
                res["mse_by_timestep"] = [
                    {"t_range": [lo, lo + 99], "n": int(m.sum()), "mse": _r(per_ex[m].mean())}
                    for lo in range(0, 1000, 100) if (m := (t >= lo) & (t < lo + 100)).any()
                ]
        return res

    @torch.no_grad()
    def _read(self, req: dict, inputs: dict[str, np.ndarray] | None) -> tuple[dict, dict]:
        which = req.get("model", "edited")
        if which not in ("edited", "original"):
            raise ToolError("`model` must be 'edited' or 'original' for a read")
        if inputs:
            if req.get("examples") is not None:
                raise ToolError("give either `examples` (dev indices) or custom inputs, not both")
            data, idx = self._custom_inputs(inputs), None
        else:
            idx = self._examples(req)
            data = self._subset(self.dev, idx)
        n = M.num_examples(data)
        modules = req.get("modules") or []
        if isinstance(modules, str):
            modules = [modules]
        if not isinstance(modules, list) or not all(isinstance(m, str) and m in self._modules and m for m in modules):
            bad = [m for m in (modules if isinstance(modules, list) else [modules]) if m not in self._modules or not m]
            raise ToolError(f"unknown modules {bad[:5]}; see MODULES")
        capture_in = bool(req.get("module_inputs", False))
        want_out = bool(req.get("outputs", True))
        reduce = req.get("reduce")
        if reduce not in (None, "channel_stats"):
            raise ToolError("`reduce` must be omitted or 'channel_stats'")

        store: dict[str, list[torch.Tensor]] = {}
        stats: dict[str, list] = {}  # key -> [count, sum, sum of squares], per channel
        nbytes = 0

        def keep(key: str, t) -> None:
            nonlocal nbytes
            if not isinstance(t, torch.Tensor) or not t.is_floating_point():
                return
            t = t.detach().float()
            if reduce == "channel_stats" and key != "outputs":
                cd = _channel_dim(t)
                x = t.movedim(cd, -1).reshape(-1, t.shape[cd]).double()
                acc = stats.setdefault(key, [0, 0.0, 0.0])
                acc[0] += x.shape[0]
                acc[1] = acc[1] + x.sum(0)
                acc[2] = acc[2] + x.pow(2).sum(0)
                return
            nbytes += t.numel() * 4
            if nbytes > MAX_READ_BYTES:
                raise ToolError(f"the requested arrays exceed {MAX_READ_BYTES // 2**20} MB: ask for fewer examples "
                                "or modules, or use \"reduce\": \"channel_stats\" (this pass was not counted)")
            store.setdefault(key, []).append(t.clone())

        def hook(name):
            def fn(mod, args, kwargs, out):
                keep(f"{name}:out", _first_tensor(out))
                if capture_in:
                    x = _first_tensor(args)
                    keep(f"{name}:in", x if x is not None else _first_tensor(kwargs))
            return fn

        handles = [self._modules[nm].register_forward_hook(hook(nm), with_kwargs=True) for nm in modules]
        bs = M.BATCH_SIZE[self.objective]
        try:
            with self._weights(which):
                for s in range(0, n, bs):
                    out = M.forward(self.model_id, self.model, M.model_inputs(self.model_id, data, slice(s, s + bs)))
                    if want_out:
                        keep("outputs", out)
        finally:
            for h in handles:
                h.remove()
        arrays: dict[str, np.ndarray] = {}
        if idx is not None:
            arrays["index"] = idx.numpy()
        for k, parts in store.items():
            arrays[k] = torch.cat(parts).numpy()
        for k, (cnt, s1, s2) in stats.items():
            mean = s1 / cnt
            var = (s2 / cnt - mean.pow(2)).clamp_min(0)
            arrays[f"{k}:mean"] = mean.float().numpy()
            arrays[f"{k}:var"] = var.float().numpy()
        summary = {"type": "read", "model": which, "n": n, "custom_inputs": bool(inputs), "reduce": reduce,
                   "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in arrays.items()}}
        return summary, arrays

    # ------------------------------------------------------------ oracle and submission

    def oracle_restore(self, req: dict) -> dict:
        """Reference repair for Harbor's oracle agent: restore the clean value of every parameter that differs
        from it. Gated by a token that only the task's solution script holds."""
        token = str(req.get("token", ""))
        if hashlib.sha256(token.encode()).hexdigest() != self.episode["oracle_token_sha256"]:
            raise ToolError("unknown command")
        clean = M.editable_parameters(M.build_model(self.model_id))
        restored = []
        with torch.no_grad():
            for n, p in self.params.items():
                if not torch.equal(self.initial[n], clean[n]):
                    p.data.copy_(clean[n])
                    self.changed[n] = int((p.data != self.initial[n]).sum())
                    restored.append(n)
        self.history.append({"action": "oracle_restore", "t": self._t()})
        return {"restored": restored, "usage": self.usage()}

    def delta(self) -> dict[str, torch.Tensor]:
        """The elements that differ from the weights the solver was given, as flat indices and values."""
        out = {}
        for name in sorted(self.changed):
            cur = self.params[name].data.flatten()
            idx = (cur != self.initial[name].flatten()).nonzero().flatten()
            out[f"{name}::idx"] = idx.to(torch.int64).contiguous()
            out[f"{name}::val"] = cur[idx].contiguous()
        return out
