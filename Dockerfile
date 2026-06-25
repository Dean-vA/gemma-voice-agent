# Gateway image. Base carries CUDA 12.8 + cuDNN so the Transformers backend can
# use the RTX 5090 (Blackwell / sm_120). The vLLM backend runs in its own
# container (vllm/vllm-openai) and this gateway only talks to it over HTTP.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface

# libsndfile is required by soundfile; ffmpeg helps librosa decode odd formats.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# torch/torchvision already ship in the base image; install the rest.
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install --upgrade "transformers>=4.57"

COPY app ./app
COPY web ./web

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
