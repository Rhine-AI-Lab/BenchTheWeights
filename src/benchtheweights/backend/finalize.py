"""Harbor's `verifier.collect` hook: freeze the backend's state into /srv/export.

Runs inside the backend container after the solver's container has stopped.
"""

import json
import sys
import urllib.request


def main() -> None:
    episode = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "/srv/episode.json"))
    req = urllib.request.Request(
        "http://127.0.0.1:8000/internal/finalize",
        data=json.dumps({"key": episode["hmac_key"]}).encode(),
        headers={"Content-Type": "application/json"},
    )
    print(urllib.request.urlopen(req, timeout=120).read().decode())


if __name__ == "__main__":
    main()
