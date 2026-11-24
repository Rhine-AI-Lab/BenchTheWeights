#!/usr/bin/env bash
# Run agy on an Antigravity task (make_task.py), with the settings of the published results.
#   harnesses/antigravity/run.sh <task dir> <model> [effort] [attempts]
#   e.g. harnesses/antigravity/run.sh tasks-agy/hb-004 gemini-4-neon xhigh      (`agy models` lists the names)
# Credentials: the Google sign-in from login.sh (~/.config/agy-auth, or $AGY_AUTH_DIR), or GEMINI_API_KEY for a
# task converted with --auth apikey. Close any interactive agy session on the same account first: one left open
# made a trial fall back to a sign-in prompt. Jobs go to ./jobs. The script ends with a leak check over the job
# folder (file names only).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TASK=${1:?task dir}; MODEL=${2:?model}; EFFORT=${3:-xhigh}; K=${4:-1}
[ -f "$TASK/.agy_version" ] || { echo "no Antigravity task at $TASK"; exit 1; }
export AGY_VERSION=$(cat "$TASK/.agy_version")
if [ -z "${GEMINI_API_KEY:-}" ]; then
  [ -d "${AGY_AUTH_DIR:-$HOME/.config/agy-auth}/root" ] || { echo "no Google sign-in: run harnesses/antigravity/login.sh $TASK"; exit 1; }
fi
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
JOB=${JOB_NAME:-agy-$(basename "$TASK")-$MODEL-$EFFORT-$(date +%m%d-%H%M)}
RC=0
${HARBOR:-harbor} run -p "$TASK" --agent-import-path agent:Antigravity -m "google/$MODEL" \
  --ak reasoning_effort="$EFFORT" -k "$K" -n "$K" -o jobs --job-name "$JOB" || RC=$?
[ "$RC" = 0 ] || echo "harbor exited with $RC"
for t in jobs/$JOB/*/; do
  [ -f "$t/verifier/metrics.json" ] || { echo "$(basename "$t"): no metrics"; continue; }
  python3 -c "import json,sys; m=json.load(open(sys.argv[1])); print(sys.argv[2], {k: m.get(k) for k in ('recovery','repaired_score','forward_passes','edit_fraction','agent_active_sec')})" \
    "$t/verifier/metrics.json" "$(basename "$t")"
  [ -f "$t/agent/protocol_check.json" ] && python3 -c "import json,sys; c=json.load(open(sys.argv[1])); print('  check:', 'INVALID ' + '; '.join(c['reasons']) if c['invalid'] else 'ok', '| models', c['models_seen'], '| background forward s', c['background_forward_sec'])" "$t/agent/protocol_check.json"
  grep -o '"target": "[^"]*"' "$t/artifacts/var/log/egress/egress.jsonl" 2>/dev/null | sort | uniq -c | sed 's/^/  egress /' || true
done
LEAKS=$(grep -rlIE 'ya29\.[A-Za-z0-9_-]{20}|1//0[A-Za-z0-9_-]{20}|AIza[0-9A-Za-z_-]{35}|"refresh_token"|"access_token"' "jobs/$JOB" 2>/dev/null || true)
if [ -n "$LEAKS" ]; then echo "WARNING: credentials may appear in these files; do not share this job folder:"; echo "$LEAKS" | sed 's/^/  /'; fi
echo "job: jobs/$JOB"
