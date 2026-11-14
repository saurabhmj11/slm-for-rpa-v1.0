# syntax=docker/dockerfile:1.6
# OOA Assistant as a server for One Agent Studio (CPU only). See DEPLOY.md.
#
# Build from the "rpa model" folder, giving the platform repo as a second build context (only its flow schema and
# docs/llms.txt are copied in, so the checker matches that platform version):
#   docker build -f assistant/Dockerfile --build-context platform=/path/to/oneofficeautomation-platform \
#       -t ooa-assistant:1 .
# Run (model files mounted read-only at /models, see DEPLOY.md for the folder layout):
#   docker run -d -p 8765:8765 -e OOA_ASSISTANT_TOKEN=... -v /srv/ooa-models:/models:ro -v ooa-data:/data ooa-assistant:1

FROM python:3.12-slim AS nanochat
ARG NANOCHAT_REPO=https://github.com/karpathy/nanochat.git
# the nanochat version the model was trained with
ARG NANOCHAT_COMMIT=92d63d4e8bb4df75c3b71618f31ddde2378b2bcd
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --filter=blob:none "$NANOCHAT_REPO" /src/nanochat \
    && git -C /src/nanochat checkout --detach "$NANOCHAT_COMMIT" \
    && mkdir -p /opt/nanochat && cp -r /src/nanochat/nanochat /opt/nanochat/nanochat


FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
# CPU-only PyTorch (no CUDA libraries), then what the app and nanochat's inference code import
RUN pip install "torch==2.9.1" --index-url https://download.pytorch.org/whl/cpu \
    && pip install rustbpe tiktoken filelock jsonschema

COPY --from=nanochat /opt/nanochat /opt/nanochat
COPY --from=platform contracts /opt/platform/contracts
COPY --from=platform docs/llms.txt /opt/platform/docs/llms.txt
WORKDIR /app
COPY assistant /app/assistant
COPY serving /app/serving
COPY training /app/training
COPY data/catalog/catalog.jsonl /app/data/catalog/catalog.jsonl

RUN useradd --system --uid 10001 --home-dir /data ooa && mkdir -p /models /data && chown ooa /data
USER ooa
VOLUME ["/data"]

# /models/<OOA_MODEL_PACK>/ holds nanochat_base/ (tokenizer/ and chatsft_checkpoints/), optionally sft/catalog.jsonl
# and VERSION. OOA_ASSISTANT_TOKEN (required: the server listens on all interfaces) is read by app.py.
ENV OOA_MODEL_PACK=current OOA_PORT=8765 OOA_MAX_WAITING=4 OOA_TIME_BUDGET=120
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ['OOA_PORT'], timeout=4)"
CMD ["sh", "-c", "exec python /app/assistant/app.py --host 0.0.0.0 --port \"$OOA_PORT\" \
--model-dir \"/models/$OOA_MODEL_PACK/nanochat_base\" --nanochat /opt/nanochat --platform /opt/platform \
--data-dir /data --max-waiting \"$OOA_MAX_WAITING\" --time-budget \"$OOA_TIME_BUDGET\""]
