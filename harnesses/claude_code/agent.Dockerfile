# The solver's image for Claude Code: btw-agent plus Claude Code, installed at build time. At run time the
# container has no internet; its only way out is the egress proxy, to the Anthropic API.
ARG AGENT_IMAGE
FROM ${AGENT_IMAGE}
ARG CC_VERSION
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://downloads.claude.ai/claude-code-releases/bootstrap.sh | bash -s -- ${CC_VERSION} \
    && /root/.local/bin/claude --version
ENV PATH=/root/.local/bin:${PATH}
# System-wide policy, whatever CLAUDE_CONFIG_DIR is: no web tools. WebSearch runs on the API side, so the
# proxy alone would not stop it.
RUN mkdir -p /etc/claude-code \
    && printf '{\n  "permissions": {"deny": ["WebSearch", "WebFetch"]}\n}\n' > /etc/claude-code/managed-settings.json
