<p align="center">
  <img src="assets/logo.png" width="200" alt="Bench the Weights. Repair logo: a robot lifting a barbell whose plates are drawn as graphs">
</p>

<h1 align="center">Bench the Weights. Repair</h1>

<p align="center">
  <a href="https://rhinelab.ai/blog">Blog post</a> ·
  <a href="PAPER_URL">Paper</a> ·
  <a href="https://benchtheweights.com">Leaderboard</a>
</p>

A benchmark of AI agents that debug neural networks by editing their weights. Each problem is a pretrained
network with a few targeted bugs injected into its weights. The agent gets the modified network, its test
score and the score of the original network, presented as another network with the same architecture. It
is not told that anything was changed. It must find the bugs by running the network and reading its weights,
then repair them with direct edits: no gradients, no training.

## Release

| Date | What | Where |
|---|---|---|
| 24 Nov 2026 | The harness, the task format and the solver prompt | this repository |
| 1 Dec 2026 | The bug-injection pipeline: its agent, prompt and the three tests | `bug_injection/` |
| 8 Dec 2026 | The 42 problems: modified networks, answer keys, evaluation data | `problems/` |

Until the problems are public, we can run the evaluation on your model with the same harness, budget and
problems as the published results.

## The solver's setup

Each trial is a [Harbor](https://github.com/harbor-framework/harbor) task with three containers:

- **solver**: the agent, a persistent Python kernel over the weights (`py`) and a forward-pass command
  (`forward`). No network weights, data or autograd library, and no internet except an egress proxy to
  the agent's model API.
- **backend**: holds the network, the starting weights and the dev split. It serves every read, forward
  pass and edit, counts them against the budgets, and signs the edits it accepts.
- **verifier**: a separate environment with no network. It checks the signature, applies the edits and
  scores the network on a hidden test split.

| Limit | Value |
|---|---|
| Working time | 25 min; the clock pauses while a forward pass computes |
| Forward passes | 50, each on up to 1,024 dev examples or inputs of the agent's own |
| Edits | 3% of all parameters, counted element by element |
| Gradients | none: no backward passes, optimisers or iterative fitting |

The full prompt is [`src/benchtheweights/prompts/solver.md`](src/benchtheweights/prompts/solver.md). The
score is the share of the lost performance the repair recovers on the hidden split:

```
recovery = (repaired - modified) / (original - modified)
```

## Layout

```
src/benchtheweights/
  prompts/solver.md   the solver's prompt
  registry.py         pinned networks and datasets, metric definitions
  models.py           building and deterministic evaluation of each network
  engine.py           weights, edits, forward passes and their accounting
  backend/            HTTP server for the solver's API; Harbor's collect hook
  client/             the solver's side, stdlib + numpy: wsp.py (API), kernel.py (`py`), forward_cli.py (`forward`)
  submission.py       the signed bundle of final edits
  verifier.py         scores a bundle on the hidden split
  task.py             writes a problem's Harbor task
  instruction.py      fills in the solver's prompt for a problem
  prepare.py          downloads the checkpoints and builds the dev and hidden splits
docker/               shared images: btw-runtime (torch, CPU) and btw-agent (the solver's workspace)
scripts/              build_images.sh, make_task.py (package a problem as a task)
harnesses/
  claude_code/        Claude Code tasks and runner
  codex/              Codex tasks (OpenAI, OpenRouter, Alibaba) and runner
  antigravity/        Antigravity (agy) tasks for Gemini, Google sign-in and runner
  egress/             the proxy that lets the solver reach its model API and nothing else
  adversary.py        a scripted agent that tries to break the isolation
tests/                end-to-end tests on a small synthetic network
```

## Running a problem

Needs Docker and Harbor 0.18.0 (`uv tool install harbor==0.18.0`).

```bash
pip install -e '.[runtime,prepare]' --extra-index-url https://download.pytorch.org/whl/cpu
python -m benchtheweights.prepare --models <model_id>          # the network's checkpoint and splits, into ./cache
scripts/build_images.sh                                           # btw-runtime, btw-agent
scripts/make_task.py problems/hb-004 --out tasks                  # the problem's images and base task
harnesses/claude_code/make_task.py tasks/hb-004 --out tasks-cc
harnesses/codex/make_task.py tasks/hb-004 --out tasks-codex       # --provider openrouter: --out tasks-codex-openrouter
harnesses/antigravity/make_task.py tasks/hb-004 --out tasks-agy
```

Check every task before a model run. The scripted agents make no model calls:

```bash
harbor run -p tasks-cc/hb-004 -a oracle -o jobs     # reward 1.0: restores the original weights
harbor run -p tasks-cc/hb-004 -a nop -o jobs        # reward 0.0
PYTHONPATH=harnesses harbor run -p tasks-cc/hb-004 --agent-import-path adversary:AdversaryAgent -o jobs   # reward 0.0
```

Then run a model, with the credentials set up as in the next section:

```bash
harnesses/claude_code/run.sh tasks-cc/hb-004 claude-opus-5-5 xhigh
harnesses/codex/run.sh openai tasks-codex/hb-004 gpt-6.2-astra xhigh
harnesses/codex/run.sh openrouter tasks-codex-openrouter/hb-004 gemini-4-neon xhigh
harnesses/antigravity/run.sh tasks-agy/hb-004 gemini-4-neon xhigh
```

Each script takes `<task> <model> [effort] [attempts]` (Codex takes the provider first) and prints the score
of every trial when Harbor is done.

### Credentials and subscriptions

Every harness runs on a subscription you already have or on an API key. The credential goes into the solver's
container for the trial only, and the egress proxy lets the container reach that one provider and nothing else.
Every runner ends with a check that lists any job files that look like they contain a token; do not share a
job folder that the check flags.

**Claude Code with a Claude Pro or Max plan.** Create a long-lived token once, on any machine where you are
signed in to Claude Code, and export it before a run:

```bash
claude setup-token                       # prints a token for your subscription
export CLAUDE_CODE_OAUTH_TOKEN=...       # or ANTHROPIC_API_KEY=... for API billing
harnesses/claude_code/run.sh tasks-cc/hb-004 claude-opus-5-5 xhigh
```

**Codex with a ChatGPT plan.** Sign in once with `codex login`, which writes `~/.codex/auth.json`. When
`OPENAI_API_KEY` is not set, `run.sh openai` copies that file into the container (Harbor's
`CODEX_FORCE_AUTH_JSON`), and Codex refreshes the sign-in itself through `auth.openai.com`, which the proxy
allows.

```bash
codex login                              # once; skip it and export OPENAI_API_KEY for API billing
harnesses/codex/run.sh openai tasks-codex/hb-004 gpt-6.2-astra xhigh
```

**Other models through Codex.** For OpenRouter, put the key in a Codex auth file, which Harbor copies into the
container so the key never appears on a command line:

```bash
mkdir -p ~/.config/openrouter
printf '{"OPENAI_API_KEY": "%s"}\n' "$OPENROUTER_API_KEY" > ~/.config/openrouter/auth.json
chmod 600 ~/.config/openrouter/auth.json
harnesses/codex/run.sh openrouter tasks-codex-openrouter/hb-004 claude-opus-5-5 xhigh
```

The runner takes the benchmark's model names and maps them to OpenRouter's in `SLUGS`
(`harnesses/codex/openrouter_agent.py`); add a line there for another model.

For Alibaba's Token Plan (Qwen, GLM), write `ALIBABA_TOKEN_PLAN_KEY=...` to
`~/.config/alibaba/token-plan.env` (chmod 600) and run `harnesses/codex/run.sh alibaba`. `ALIBABA_BASE_URL`
changes the endpoint from the Singapore default.

**Antigravity with a Google account (Gemini plan).** agy signs in through the browser, so sign in once,
interactively, inside the task's solver image. `login.sh` saves the files agy writes to `~/.config/agy-auth`
(chmod 700). Each trial copies them into the container and deletes them before Harbor collects the logs.

```bash
harnesses/antigravity/login.sh tasks-agy/hb-004    # open the URL it prints, paste the code back, then /exit
harnesses/antigravity/run.sh tasks-agy/hb-004 gemini-4-neon xhigh
```

Close any interactive agy session that uses the same account before a run: one left open made a trial fall back
to a sign-in prompt. With a Gemini API key instead, convert the tasks with `make_task.py --auth apikey` and
export `GEMINI_API_KEY`.

A trial at the published settings is one long agent session, so it counts against a plan's usage limits like
any other session. Check your plan's terms for automated use before you run a batch.

`jobs/<job>/<trial>/verifier/metrics.json` has the recovery, the forward passes and the edited fraction. The
backend's request log and the signed edits are in `artifacts/srv/export`, the egress log in
`artifacts/var/log/egress`.

### Settings of the published results

| Harness | Version | Settings |
|---|---|---|
| Harbor | 0.18.0 | one attempt per trial |
| Claude Code | 2.1.318 | `reasoning_effort=xhigh`, WebSearch and WebFetch denied, background tasks off |
| Codex | 0.167.0 | `reasoning_effort=xhigh`, web search disabled, activity watcher on |
| Antigravity | 1.5.11 | `--effort xhigh` (`high` for Gemini 4 Carbon and 3.8 Flash), URL reads and subagents denied, busy hook on, 10-min no-output watchdog |

Codex can keep working while a forward pass computes. Its image runs a watcher that reads the Codex session
log and reports the agent's working turns to the backend, which takes them out of the clock pause, so Codex
is credited only for the time it waits, as Claude Code is.

agy has no setting to keep the agent waiting on a command: it hands the model a turn about 10 s into any long
one. A hook refuses every tool but waiting, and refuses to stop, while a command runs. After each trial,
`agent/protocol_check.json` lists the forward passes that still ran in the background and marks the trial
invalid if it used a web tool or another model answered. Google's release server only serves the latest agy;
`make_task.py --agy-url --agy-sha512` pins another release.

## Adding a network

A network needs a `DatasetSpec` and a `ModelSpec` in `registry.py`. If it uses a new loader, modality or
objective, it also needs:

- `registry.py`: the files to download with the weights (`HF_EXTRA_FILES`);
- `prepare.py`: a function in `PREPARERS` that builds the dev and hidden splits from the dataset's parquet
  table, and the minimum clean score in `MIN_CLEAN_SCORE`;
- `models.py`: a builder in `BUILDERS`, and cases in `model_inputs`, `forward` and `targets`;
- `engine.py`: the input fields and their description for the solver (`INPUT_FIELDS`, `INPUT_DESC`,
  `TARGET_DESC`);
- `instruction.py`: what an `evaluate` pass reports for a new objective.

A problem folder holds `meta.json` (with `model_id_in_registry`) and the starting weights,
`corrupted.safetensors`. `python -m benchtheweights.prepare --models <model_id>` writes the checkpoint and the
splits to the cache (`checkpoints/<model_id>/`, `data/<dataset_id>/{dev,hidden}.pt`) and checks the clean
scores; `scripts/make_task.py` reads them from there.

## Tests

```bash
pip install -e '.[runtime,dev]' --extra-index-url https://download.pytorch.org/whl/cpu
pytest
```

## Citation

```bibtex
@misc{rhine2026benchtheweightsrepair,
  title        = {Bench the Weights. Repair: Can {AI} Agents Debug Neural Networks?},
  author       = {Faro, Robin and Lopardo, Antonio and Miele, Andrea and Lucchi, Aur{\'e}lien and Bogunovic, Ilija},
  year         = {2026},
  month        = nov,
  howpublished = {Rhine AI Lab research blog},
  url          = {https://rhinelab.ai/blog}
}
```
