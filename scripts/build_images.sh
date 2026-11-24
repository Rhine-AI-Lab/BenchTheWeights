#!/bin/bash
# Build the two shared images: btw-runtime (base of every problem's backend and verifier) and btw-agent
# (the solver's workspace). Tags follow the package version.
set -euo pipefail
cd "$(dirname "$0")/.."
TAG=$(python3 -c "import re; print(re.search(r'__version__ = \"(.+)\"', open('src/benchtheweights/__init__.py').read()).group(1))")
docker build -f docker/runtime.Dockerfile -t btw-runtime:$TAG .
docker build -f docker/agent.Dockerfile -t btw-agent:$TAG .
docker images --format '{{.Repository}}:{{.Tag}}  {{.Size}}' | grep "^btw-.*:$TAG"
