# Egress proxy: CONNECT to an allowlist, every request logged. Also the base of the Codex OpenRouter relay.
FROM python:3.12-slim@sha256:b18b261e1b4974b4f321451ad5251ded85abb22c2b8b262da527dda1d031ab30
COPY egress_proxy.py /opt/egress_proxy.py
RUN mkdir -p /var/log/egress
ENV EGRESS_ALLOW=api.anthropic.com:443 EGRESS_PORT=3128 PYTHONUNBUFFERED=1
EXPOSE 3128
CMD ["python", "/opt/egress_proxy.py"]
