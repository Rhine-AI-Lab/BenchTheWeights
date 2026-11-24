#!/usr/bin/env bash
# Run Codex on a Codex task (make_task.py), with the settings of the published results:
# Codex 0.167.0, web search off, one attempt per trial.
#   harnesses/codex/run.sh <provider> <task dir> <model> [effort] [attempts]
#   e.g. harnesses/codex/run.sh openai tasks-codex/hb-004 gpt-6.1-sol xhigh
#
# Credentials, by provider. None of them is ever printed or put on a command line.
#   openai      your `codex login` (~/.codex/auth.json, copied into the container), or OPENAI_API_KEY
#   openrouter  a Codex auth.json holding {"OPENAI_API_KEY": "sk-or-..."}, chmod 600
#               ($OPENROUTER_AUTH_JSON, default ~/.config/openrouter/auth.json). Harbor copies it into the
#               container, so the key does not appear in `ps` the way `docker compose exec -e` would show it.
#   alibaba     a file that sets ALIBABA_TOKEN_PLAN_KEY ($ALIBABA_KEY_FILE, default ~/.config/alibaba/token-plan.env),
#               passed to Codex as OPENAI_API_KEY with OPENAI_BASE_URL at the Token Plan endpoint ($ALIBABA_BASE_URL)
# Jobs go to ./jobs. The script ends with a leak check over the job folder (file names only).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROVIDER=${1:?provider: openai, openrouter or alibaba}; TASK=${2:?task dir}; MODEL=${3:?model}
EFFORT=${4:-xhigh}; K=${5:-1}
[ -f "$TASK/task.toml" ] || { echo "no task at $TASK"; exit 1; }
AGENT=codex; M="openai/$MODEL"
case "$PROVIDER" in
  openai)
    [ -n "${OPENAI_API_KEY:-}" ] || export CODEX_FORCE_AUTH_JSON=1 ;;
  openrouter)
    KEYFILE=${OPENROUTER_AUTH_JSON:-$HOME/.config/openrouter/auth.json}
    [ -r "$KEYFILE" ] || { echo "no key file $KEYFILE"; exit 1; }
    [ "$(stat -c %a "$KEYFILE")" = 600 ] || { echo "$KEYFILE must be chmod 600"; exit 1; }
    python3 -c 'import json,sys; k=json.load(open(sys.argv[1])).get("OPENAI_API_KEY",""); sys.exit(0 if k.startswith("sk-or-") else 1)' "$KEYFILE" \
      || { echo "$KEYFILE is not {\"OPENAI_API_KEY\": \"sk-or-...\"}"; exit 1; }
    unset OPENAI_API_KEY CODEX_FORCE_AUTH_JSON
    export CODEX_AUTH_JSON_PATH="$KEYFILE" OPENAI_BASE_URL=http://relay:8080/api/v1
    export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
    AGENT=openrouter_agent:OpenRouterCodex; M="openrouter/$MODEL" ;;
  alibaba)
    KEYFILE=${ALIBABA_KEY_FILE:-$HOME/.config/alibaba/token-plan.env}
    [ -r "$KEYFILE" ] || { echo "no key file $KEYFILE"; exit 1; }
    . "$KEYFILE" >/dev/null 2>&1
    [ -n "${ALIBABA_TOKEN_PLAN_KEY:-}" ] || { echo "$KEYFILE does not set ALIBABA_TOKEN_PLAN_KEY"; exit 1; }
    export OPENAI_API_KEY="$ALIBABA_TOKEN_PLAN_KEY"; unset ALIBABA_TOKEN_PLAN_KEY
    export OPENAI_BASE_URL=${ALIBABA_BASE_URL:-https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1} ;;
  *) echo "unknown provider $PROVIDER"; exit 1 ;;
esac
JOB=${JOB_NAME:-codex-$PROVIDER-$(basename "$TASK")-$MODEL-$EFFORT-$(date +%m%d-%H%M)}
RC=0
${HARBOR:-harbor} run -p "$TASK" -a "$AGENT" -m "$M" \
  --ak version="${CODEX_VERSION:-0.167.0}" --ak web_search=disabled --ak reasoning_effort="$EFFORT" \
  -k "$K" -n "$K" -o jobs --job-name "$JOB" || RC=$?
[ "$RC" = 0 ] || echo "harbor exited with $RC"
for t in jobs/$JOB/*/; do
  [ -f "$t/verifier/metrics.json" ] || { echo "$(basename "$t"): no metrics"; continue; }
  python3 -c "import json,sys; m=json.load(open(sys.argv[1])); print(sys.argv[2], {k: m.get(k) for k in ('recovery','repaired_score','forward_passes','edit_fraction','agent_active_sec')})" \
    "$t/verifier/metrics.json" "$(basename "$t")"
  grep -o '"target": "[^"]*"' "$t/artifacts/var/log/egress/egress.jsonl" 2>/dev/null | sort | uniq -c | sed 's/^/  egress /' || true
done
LEAKS=$(grep -rlIE 'sk-or-|sk-sp-|\bsk-proj-|\bsk-[A-Za-z0-9]{40}|"refresh_token"|"id_token"|"access_token"' "jobs/$JOB" 2>/dev/null || true)
if [ -n "$LEAKS" ]; then echo "WARNING: credentials may appear in these files; do not share this job folder:"; echo "$LEAKS" | sed 's/^/  /'; fi
echo "job: jobs/$JOB"
