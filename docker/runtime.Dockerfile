# Base of every problem's backend and verifier images: the pinned CPU torch stack.
# The problem images add the package, the checkpoint, the starting weights and the data splits.
FROM python:3.12-slim@sha256:b18b261e1b4974b4f321451ad5251ded85abb22c2b8b262da527dda1d031ab30
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    BTW_CACHE=/opt/btw/cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
COPY requirements-runtime.txt /tmp/requirements-runtime.txt
RUN pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple \
      -r /tmp/requirements-runtime.txt
