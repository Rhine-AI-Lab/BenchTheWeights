"""Renders the solver's prompt, `prompts/solver.md`, for one problem.

The prompt never says that the network was modified: it gives the network's test score, the score another
network with the same architecture reaches, the rules and the tools.
"""

from __future__ import annotations

from pathlib import Path

from .engine import INPUT_DESC, INPUT_FIELDS, TARGET_DESC
from .registry import DATASETS, MODELS, SCORE_INFO

TEMPLATE = Path(__file__).parent / "prompts" / "solver.md"

EVAL_SUMMARY = {
    "classification": "`score` (the accuracy), `nll`, `mean_confidence`, `per_class_accuracy` and "
                      "`predicted_class_histogram`",
    "denoising": "`score`, `mse`, `mse_per_channel`, `output_mean_per_channel`, `output_std_per_channel` and "
                 "`mse_by_timestep`",
    "reconstruction": "`score`, `mse`, `mse_per_channel`, `output_mean_per_channel` and `output_std_per_channel`",
    "lm": "`score` (minus the mean next-token NLL), `nll`, `perplexity` and `nll_by_position` (in blocks of 32 "
          "positions)",
}
EVAL_ARRAYS = {
    "classification": ", and `<model>_confusion` (rows: true class, columns: predicted class)",
    "denoising": "",
    "reconstruction": "",
    "lm": " (for this network `<model>_outputs` are the per-token NLLs, [N, 255])",
}
PAIRED_EXTRA = {
    "classification": ", and how many examples went from wrong to right and from right to wrong",
    "denoising": "",
    "reconstruction": "",
    "lm": "",
}


def fmt_score(objective: str, s: float) -> str:
    return f"{100 * s:.1f}%" if objective == "classification" else f"{s:.4f}"


PYTHON_ENV = ("Python 3 is installed with numpy, scipy, scikit-learn, pandas, sympy, networkx and safetensors. "
              "There is no PyTorch, no GPU and no internet.")


def render_instruction(model_id: str, reference: float, current: float, limits: dict, work: str = "/work",
                       python_env: str = PYTHON_ENV) -> str:
    spec = MODELS[model_id]
    obj = spec.objective
    metric, metric_desc = SCORE_INFO[obj]
    minutes = float(limits["time_limit_sec"]) / 60
    values = {
        "ARCH": spec.architecture,
        "TASK": spec.task,
        "METRIC": metric,
        "METRIC_DESC": metric_desc,
        "CURRENT": fmt_score(obj, current),
        "REFERENCE": fmt_score(obj, reference),
        "TIME_MIN": f"{minutes:g}",
        "MAX_PASSES": int(limits["max_forward_passes"]),
        "MAX_EDIT_PCT": f"{100 * float(limits['max_edit_fraction']):g}",
        "DEV_N": DATASETS[spec.dataset_id].n_dev,
        "INPUT_DESC": INPUT_DESC[spec.loader],
        "TARGET_DESC": TARGET_DESC[obj],
        "INPUT_FIELDS": ", ".join(f"`{f}`" for f in INPUT_FIELDS[spec.loader]),
        "EVAL_SUMMARY": EVAL_SUMMARY[obj],
        "EVAL_ARRAYS": EVAL_ARRAYS[obj],
        "PAIRED_EXTRA": PAIRED_EXTRA[obj],
        "WORK": work.rstrip("/"),
        "PYTHON_ENV": python_env,
    }
    text = TEMPLATE.read_text()
    for k, v in values.items():
        text = text.replace("{{" + k + "}}", str(v))
    if "{{" in text:
        raise ValueError("unfilled placeholder in solver.md: " + text[text.index("{{"):][:40])
    return text
