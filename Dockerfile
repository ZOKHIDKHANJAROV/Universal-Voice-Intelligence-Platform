# API, call bridge and web UI (console at /, live monitor at /monitor).

# CUDA runtime for faster-whisper (CTranslate2): cuBLAS 12 and cuDNN 9.
# By default pip downloads them (INSTALL_GPU=1, ~1.2 GB). On a slow network,
# any local image that already has them can donate the libraries instead:
#   CUDA_LIBS_IMAGE=universal-voice-intelligence-platform-navoiy-tts INSTALL_GPU=0
ARG CUDA_LIBS_IMAGE=python:3.12-slim
FROM ${CUDA_LIBS_IMAGE} AS cuda-libs
RUN mkdir -p /cuda \
    && for f in /usr/local/cuda/lib64/libcublas*.so.12* /usr/lib/x86_64-linux-gnu/libcudnn*.so.9*; do \
         if [ -e "$f" ]; then cp -a "$f" /cuda/; fi; \
       done

FROM python:3.12-slim
ARG INSTALL_GPU=1
# Piper voices the Russian phrases that are not pre-rendered yet.
ARG EXTRAS=tts-ru
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10 \
    LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib:/opt/cuda/lib
WORKDIR /app
COPY --from=cuda-libs /cuda/ /opt/cuda/lib/
COPY pyproject.toml README.md ./
# The pip cache survives failed builds, so a dropped connection does not
# start the downloads over.
RUN --mount=type=cache,target=/root/.cache/pip \
    mkdir app && touch app/__init__.py \
    && extras="$EXTRAS" && if [ "$INSTALL_GPU" = "1" ]; then extras="gpu${extras:+,$extras}"; fi \
    && pip install ".${extras:+[$extras]}" \
    && pip uninstall -y univoice-ai && rm -rf app build ./*.egg-info
COPY app ./app
COPY scripts ./scripts
EXPOSE 8000 9019
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
