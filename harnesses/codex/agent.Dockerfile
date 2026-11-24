# The solver's image for Codex: btw-agent plus Node and the Codex CLI, installed at build time, and the
# activity watcher that `forward` starts. At run time the container has no internet; its only way out is the
# egress proxy, to the model API.
ARG AGENT_IMAGE
FROM ${AGENT_IMAGE}
ARG CODEX_VERSION
RUN apt-get update && apt-get install -y --no-install-recommends nodejs npm ripgrep ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g @openai/codex@${CODEX_VERSION} && npm cache clean --force \
    && codex --version
COPY activity_watch.py /opt/wsp/activity_watch.py
COPY forward.sh /usr/local/bin/forward
RUN chmod 0755 /usr/local/bin/forward && chmod 0644 /opt/wsp/activity_watch.py
