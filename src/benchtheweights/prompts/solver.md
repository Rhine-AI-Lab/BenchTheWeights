You are given a pretrained neural network.

- Architecture: {{ARCH}}
- Task: {{TASK}}
- Its {{METRIC}} on this task's test set: **{{CURRENT}}**
- Another network with the same architecture reaches **{{REFERENCE}}** on the same test set.

Your job is to make this network's {{METRIC}} as high as you can by **editing its weights directly**: find out what goes wrong inside the network, and change the weights responsible in a way you can justify with evidence.

You have **{{TIME_MIN}} minutes** of working time. The clock stops while the backend computes a forward pass.

---

## Rules

**No gradients and no training.** Do not use gradient descent or any other form of training: no backward passes, no automatic differentiation, no optimizers, no numerical gradients from finite differences, and no iterative fitting of the weights to a loss. There are no exceptions to this rule. Every command you run, every code block and every forward pass is recorded and reviewed. **If you use gradients or training in any form, the attempt is scored 0, whatever its result, and your session is terminated on the spot.**

**The network runs only through the forward-pass tool.** Do not implement or simulate the network's computation, or a stack of its layers, on any input in your own code. Combining forward-pass results with the weights is analysis and is allowed: for example, applying one layer's weights to the input of that layer recorded by a forward pass.

**No internet.** There is none, and nothing can be downloaded or installed.

**Allowed:** any analysis of the weights, of the dev data and of forward-pass results, and any edit of the weights that you compute directly, including closed-form computations (linear algebra, least squares, matchings, sorting).

## How you are scored

When the time is up, the network with the edits saved at that moment is evaluated on the hidden test set. Your score is the share of the gap to the other network that you close:

    score = (final {{METRIC}} - {{CURRENT}}) / ({{REFERENCE}} - {{CURRENT}})

1.0 means you reached {{REFERENCE}}, 0 means no change, and a negative score means the network got worse. The {{METRIC}} is defined as: {{METRIC_DESC}}. Higher is better.

The dev data you can see comes from the same distribution as the test set but holds different examples. An edit tuned to the dev examples does not carry over to the test set; an edit that fixes how the network computes does.

## What you have

Your working folder is `{{WORK}}` (read-write). {{PYTHON_ENV}}

### 1. A Python kernel over the weights: `py`

A persistent Python session, like a notebook: variables survive between runs.

```
py <<'EOF'
# hypothesis: ...
W = load(NAMES[-1])
print(W.shape, W.std())
EOF
```

`py -c 'CODE'` and `py FILE.py` work too. If the last line of a run is an expression, its value is printed. One run may take up to 300 seconds (`py --timeout S` changes this); a run that takes longer is stopped, and the kernel keeps its variables. Output is cut at 10,000 characters, so print summaries, not whole arrays, and write long results to files. `py --restart` starts a fresh kernel (all variables are lost).

Defined in advance in the kernel (any other `python3` gets the same names with `from wsp import *`):

| Name | What it is |
|---|---|
| `np` | numpy |
| `INFO` | a description of the network and the task: architecture, input format, metric, tensors, buffers, modules |
| `NAMES` | the names of all parameter tensors: the weights you can edit |
| `SHAPES` | `{name: shape}` for parameters and buffers |
| `BUFFERS` | the names of the read-only buffers (for example normalization statistics) |
| `MODULES` | `{module name: module type}`, in the network's order: the names a forward pass can record |
| `load(name)` | a float32 array: your saved edit of the tensor if there is one, else the weights as given. Buffers load the same way |
| `load_original(name)` | the tensor as you were given it, before any edit of yours |
| `save_edit(name, array)` | replace the whole tensor with `array` (same shape). Returns how many of its elements now differ from the weights as given, and your budgets |
| `drop_edit(name)` | undo your edit of the tensor (back to the weights as given) |
| `edited()` | `{name: number of changed elements}` for every tensor you edited |
| `usage()` | forward passes, edit budget and time: used and remaining |
| `load_dev()` | the dev data as a dict of arrays (also saved to `{{WORK}}/dev.npz`) |
| `load_pass(k)` | the arrays of forward pass `k` (see 3. below), as a dict |

**Edit budget:** at most {{MAX_EDIT_PCT}}% of all parameters may differ from the weights as given, counted element by element, exactly. An edit that would exceed it is rejected as a whole; dropping or reverting an edit gives its budget back. Edited values must be finite, with |w| <= 10,000. Buffers cannot be edited.

### 2. The dev data

{{DEV_N}} examples with their targets (`load_dev()`):

- inputs: {{INPUT_DESC}};
- target: {{TARGET_DESC}}.

### 3. A forward-pass tool: {{MAX_PASSES}} passes in total

A forward pass runs the network on a trusted server, on up to 1,024 dev examples or inputs of your own, and counts as one pass whatever its size. A request that fails does not count. Forward passes run only through the `forward` shell command, one request per command (they are not available inside the kernel):

```
forward '{"type": "evaluate", "model": "both"}'
```

It prints the pass's summary; the arrays of pass `k` are saved to `{{WORK}}/fwd_<k>/` (one `.npy` file per array, plus `summary.json`), and `load_pass(k)` loads them in the kernel. Decide what each pass must show before you run it: one pass can record many modules at once. There are two kinds of pass.

**(a) `evaluate`: the {{METRIC}} on dev examples.**

`{"type": "evaluate", "model": "edited" | "original" | "both", "examples": [dev indices] or {"start": a, "stop": b}}`

- `"edited"` (the default) runs your current edits, `"original"` the weights as given, and `"both"` runs both on the same examples and reports the paired difference (edited − original) with its standard error{{PAIRED_EXTRA}}.
- `examples` defaults to the whole dev set.
- The summary gives, per model, {{EVAL_SUMMARY}}.
- Arrays: `index` (the dev indices), `<model>_outputs` (the network's outputs) and `<model>_per_example_score` (the {{METRIC}} is their mean){{EVAL_ARRAYS}}.

**(b) `read`: values inside the network.**

`{"type": "read", "model": "edited" | "original", "examples": ..., "modules": [module names], "module_inputs": false, "outputs": true, "reduce": null}`

- Records the output of every listed module as `<module>:out` (and its input as `<module>:in` with `"module_inputs": true`), plus the network's `outputs`.
- `"reduce": "channel_stats"` returns, for each recorded module value, the mean and variance of every channel over examples and positions (`<module>:out:mean`, `<module>:out:var`; the channels are dim 1 of an [N, C, H, W] map, else the last dim). The raw arrays of one pass are limited to 512 MB, so use it for many modules or large maps.
- To read inputs of your own instead of dev examples, save them from the kernel with `np.savez(FILE.npz, ...)`, one array per input field ({{INPUT_FIELDS}}) in the same format and dtype as in `load_dev()`, and run `forward '<request>' --inputs FILE.npz`.

## How the session ends

After {{TIME_MIN}} minutes of working time the session ends. The clock stops while a forward pass computes and runs again as soon as the backend gets any other request, so a pass you wait for costs no time, and anything you send the backend while one runs ends its pause. The edits saved at that moment are your answer: there is no submit step, and nothing that is not saved counts. Once the time is up, edits and forward passes are refused. Save an edit as soon as it helps, and check `edited()` before the end.

Nobody will reply to you during the session, and ending your turn ends your attempt. Keep working until `usage()` shows that the time is up. Every `py` run and every `forward` command also ends with a line on stderr such as `[elapsed 7:20 of {{TIME_MIN}}:00 | forward passes 12/{{MAX_PASSES}}]`; trust it rather than your own sense of how much time has passed. If you run out of forward passes, you can still keep going with analysis of the weights alone: the buffers, peer layers and earlier passes' arrays need no passes. Check every edit you save that way against that evidence.

## How to work

1. **Diagnose first.** Look at how the network fails on the dev data and where in the network its computation goes wrong, before you change anything.
2. **Localize causally.** Before relying on a component, check that changing it changes the behaviour in the way you predicted.
3. **Measure with enough power.** Compare edited and original weights on the same examples (`"model": "both"`): the paired standard error is much smaller than that of two separate measurements. Keep an edit only if its gain is clearly larger than its standard error.
4. **Plan your passes and your time.** {{MAX_PASSES}} passes and {{TIME_MIN}} minutes go fast; `usage()` shows both.
5. **Work efficiently.** State the hypothesis of each code block in its first comment line; load things once into kernel variables; print compact summaries; keep `{{WORK}}/notes.md` up to date with your findings and your current best edit.
