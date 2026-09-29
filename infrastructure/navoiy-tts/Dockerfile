FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV LANG=C.UTF-8     LC_ALL=C.UTF-8     DEBIAN_FRONTEND=noninteractive     PYTHONUNBUFFERED=1     PYTHONDONTWRITEBYTECODE=1     CONDA_DEFAULT_ENV=cosyvoice     PATH=/opt/conda/envs/cosyvoice/bin:/opt/conda/bin:$PATH     PYTHONPATH=/opt/CosyVoice:/opt/CosyVoice/third_party/Matcha-TTS

RUN apt-get update && apt-get install -y --no-install-recommends     git git-lfs build-essential curl wget ffmpeg sox libsox-dev     && rm -rf /var/lib/apt/lists/*     && git lfs install

RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh     && bash /tmp/miniforge.sh -b -p /opt/conda     && rm /tmp/miniforge.sh

RUN conda create -y -n cosyvoice python=3.10     && conda install -y -n cosyvoice -c conda-forge pynini==2.1.5     && conda clean -afy

RUN git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git /opt/CosyVoice     && cd /opt/CosyVoice     && git checkout 074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc     && git submodule update --init --recursive

RUN cd /opt/CosyVoice     && pip install --no-cache-dir setuptools==81.0.0 wheel     && pip install --no-cache-dir --no-build-isolation --no-deps openai-whisper==20231117     && grep -vE '^tensorrt-(cu12|cu12-bindings|cu12-libs)==' requirements.txt | grep -vE '^(torch|torchaudio)==' > /tmp/requirements.runtime.txt     && pip install --no-cache-dir torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121     && pip install --no-cache-dir -r /tmp/requirements.runtime.txt     && pip install --no-cache-dir huggingface_hub==0.36.2 numpy==1.26.4 transformers==4.51.3

WORKDIR /opt/univoice-tts
COPY infrastructure/navoiy-tts/server.py ./server.py

EXPOSE 8100

CMD ["python", "server.py"]
