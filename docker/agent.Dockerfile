# The solver's workspace: the `py` kernel and the `forward` client (stdlib + numpy), the scientific Python
# stack for analysis, and tmux/asciinema for terminal agents. The container has no internet at run time, so
# nothing can be installed later. No network weights, data, torch or other autograd library.
FROM python:3.12-slim@sha256:b18b261e1b4974b4f321451ad5251ded85abb22c2b8b262da527dda1d031ab30
RUN apt-get update && apt-get install -y --no-install-recommends tmux asciinema procps jq less \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy==2.1.3 scipy==1.14.1 scikit-learn==1.5.2 pandas==2.2.3 sympy==1.13.3 \
      networkx==3.4.2 safetensors==0.4.5
COPY src/benchtheweights/client/wsp.py src/benchtheweights/client/kernel.py src/benchtheweights/client/forward_cli.py /opt/wsp/
RUN printf '#!/bin/sh\nexec python3 /opt/wsp/kernel.py "$@"\n' > /usr/local/bin/py \
    && printf '#!/bin/sh\nexec python3 /opt/wsp/forward_cli.py "$@"\n' > /usr/local/bin/forward \
    && chmod +x /usr/local/bin/py /usr/local/bin/forward && mkdir -p /work && chmod 777 /work
ENV PYTHONPATH=/opt/wsp WSP_BACKEND=http://backend:8000 WSP_WORK=/work
WORKDIR /work
