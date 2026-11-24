#!/bin/sh
# `forward` in the Codex image: make sure the activity watcher runs, then run the pass.
python3 /opt/wsp/activity_watch.py --ensure || { echo '{"ok": false, "error": "activity watcher did not start"}'; exit 3; }
exec python3 /opt/wsp/forward_cli.py "$@"
