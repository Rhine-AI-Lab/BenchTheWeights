#!/usr/bin/env bash
# Sign agy in with a Google account (Gemini plan), once, inside a task's solver image. agy keeps its sign-in in
# the OS keyring and falls back to files in a container; this saves exactly the files it wrote.
#   harnesses/antigravity/login.sh tasks-agy/hb-004
# Interactive: open the URL agy prints, sign in, paste the code back, wait for the prompt, then type /exit.
# The files go to ~/.config/agy-auth (or $AGY_AUTH_DIR), chmod 700; their contents are never printed.
set -euo pipefail
TASK=${1:?task dir (make_task.py)}
IMG=$(awk '/^FROM /{print $2}' "$TASK/environment/Dockerfile")
OUT=${AGY_AUTH_DIR:-$HOME/.config/agy-auth}
C=agy-login-$(id -un)
DIFF=$(mktemp)
trap 'rm -f "$DIFF"; docker rm -f "$C" >/dev/null 2>&1 || true' EXIT
docker rm -f "$C" >/dev/null 2>&1 || true
echo "image $IMG; sign in, then /exit"
docker run -it --name "$C" -w /work "$IMG" agy || true
docker diff "$C" | awk '$1!="D"{print $2}' | grep -E '^/root/' | grep -vE '^/root/(\.local/bin|\.cache/antigravity/staging)' > "$DIFF"
umask 077; rm -rf "$OUT"; mkdir -p "$OUT"
docker cp "$C:/root/." "$OUT/root" >/dev/null
# keep only the files agy wrote, not the image's own
( cd "$OUT/root" && find . -type f | sed 's|^\.|/root|' ) | grep -vxF -f "$DIFF" | sed "s|^/root|$OUT/root|" | xargs -r rm -f
find "$OUT" -type d -empty -delete
chmod -R go-rwx "$OUT"
echo "saved $(find "$OUT" -type f | wc -l) files under $OUT:"
find "$OUT" -type f -printf '  %P (%s bytes)\n'
