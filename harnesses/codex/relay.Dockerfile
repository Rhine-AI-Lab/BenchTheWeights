# The prompt-cache relay for Codex on OpenRouter (cache_relay.py), on the egress proxy's image.
ARG EGRESS_IMAGE
FROM ${EGRESS_IMAGE}
COPY cache_relay.py /opt/cache_relay.py
RUN mkdir -p /var/log/relay
CMD ["python", "/opt/cache_relay.py"]
