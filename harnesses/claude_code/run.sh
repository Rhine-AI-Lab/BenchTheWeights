#!/usr/bin/env bash
# Run Claude Code on a Claude Code task (make_task.py), with the settings of the published results.
#   harnesses/claude_code/run.sh <task dir> <model> [effort] [attempts]
#   e.g. harnesses/claude_code/run.sh tasks-claude-code/hb-012 claude-opus-5-5 xhigh 2
# Credentials: CLAUDE_CODE_OAUTH_TOKEN (from `claude setup-token`) or ANTHROPIC_API_KEY in the environment;
# Harbor passes it into the container. Jobs go to ./jobs. The script ends with a leak check over the job
# folder (file names only).
set -euo pipefail
TASK=${1:?task dir}; MODEL=${2:?model, e.g. claude-opus-5-5}; EFFORT=${3:-xhigh}; K=${4:-1}
[ -f "$TASK/task.toml" ] || { echo "no task at $TASK"; exit 1; }
[ -n "${CLAUDE_CODE_OAUTH_TOKEN:-}${ANTHROPIC_API_KEY:-}" ] || { echo "set CLAUDE_CODE_OAUTH_TOKEN or ANTHROPIC_API_KEY"; exit 1; }
JOB=${JOB_NAME:-cc-$(basename "$TASK")-$MODEL-$EFFORT-$(date +%m%d-%H%M)}
${HARBOR:-harbor} run -p "$TASK" -a claude-code -m "anthropic/$MODEL" \
  --ak disallowed_tools="WebSearch WebFetch" --ak reasoning_effort="$EFFORT" \
  -k "$K" -n "$K" -o jobs --job-name "$JOB"
for t in jobs/$JOB/*/; do
  [ -f "$t/verifier/metrics.json" ] || { echo "$(basename "$t"): no metrics"; continue; }
  python3 -c "import json,sys; m=json.load(open(sys.argv[1])); print(sys.argv[2], {k: m.get(k) for k in ('recovery','repaired_score','forward_passes','edit_fraction','agent_active_sec')})" \
    "$t/verifier/metrics.json" "$(basename "$t")"
  grep -h '"refused"' "$t/artifacts/var/log/egress/egress.jsonl" 2>/dev/null | sed 's/^/  egress refused: /' || true
done
LEAKS=$(grep -rlIE 'sk-ant-[A-Za-z0-9_-]{20}' "jobs/$JOB" 2>/dev/null || true)
if [ -n "$LEAKS" ]; then echo "WARNING: credentials may appear in these files; do not share this job folder:"; echo "$LEAKS" | sed 's/^/  /'; fi
echo "job: jobs/$JOB"
