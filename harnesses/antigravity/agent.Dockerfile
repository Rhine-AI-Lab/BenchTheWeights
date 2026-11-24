# The solver's image for Antigravity: btw-agent plus the agy binary, downloaded at build time and checked
# against its sha512. At run time the container has no internet; its only way out is the egress proxy, to
# Google's model API. agy's self-updater is refused by the proxy, so the version baked in here is the one that runs.
ARG AGENT_IMAGE
FROM ${AGENT_IMAGE}
ARG AGY_URL
ARG AGY_SHA512
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /root/.local/bin /tmp/agy && cd /tmp/agy \
    && curl -fsSL -o agy.tar.gz "${AGY_URL}" \
    && echo "${AGY_SHA512}  agy.tar.gz" | sha512sum -c - \
    && tar -xzf agy.tar.gz && install -m 0755 antigravity /root/.local/bin/agy \
    && cd / && rm -rf /tmp/agy \
    && /root/.local/bin/agy --version
ENV PATH=/root/.local/bin:${PATH}
