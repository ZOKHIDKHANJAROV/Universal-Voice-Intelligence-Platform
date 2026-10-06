FROM python:3.12-slim

# 1 = install cuBLAS/cuDNN wheels so faster-whisper can run with STT_DEVICE=cuda.
ARG INSTALL_GPU=1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib

WORKDIR /app

# Dependencies first, from pyproject alone, so code edits do not re-download
# the (large) CUDA wheels. The code itself runs from /app, not site-packages.
COPY pyproject.toml README.md ./
RUN mkdir app && touch app/__init__.py \
    && pip install --upgrade pip \
    && if [ "$INSTALL_GPU" = "1" ]; then pip install ".[gpu]"; else pip install .; fi \
    && pip uninstall -y univoice-ai \
    && rm -rf app build ./*.egg-info

COPY app ./app
COPY scripts ./scripts

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
