# Gateway image. Base carries CUDA 12.8 + cuDNN so the Transformers backend can
# use the RTX 5090 (Blackwell / sm_120). The vLLM backend runs in its own
# container (vllm/vllm-openai) and this gateway only talks to it over HTTP.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    UV_THREADPOOL_SIZE=32
# UV_THREADPOOL_SIZE: uvicorn runs on uvloop, which does DNS lookups on libuv's
# thread pool (4 threads by default). Probing TTS engines that aren't deployed
# ties those up for seconds per lookup, and requests to vLLM then queue behind
# them -- multi-second stalls in time-to-first-token.

# libsndfile is required by soundfile; ffmpeg helps librosa decode odd formats.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# torch/torchvision/torchaudio already ship in the base image, built for its
# CUDA. Pin them while installing the rest: otherwise a dependency that wants a
# newer torch (compressed-tensors) upgrades torch alone, which leaves
# torchaudio/torchvision unloadable and breaks transformers' audio imports.
RUN pip install --upgrade pip && \
    pip freeze | grep -E '^(torch|torchvision|torchaudio)==' > /tmp/torch-constraints.txt && \
    pip install -c /tmp/torch-constraints.txt -r requirements.txt && \
    pip install -c /tmp/torch-constraints.txt --upgrade "transformers>=4.57"

COPY app ./app
COPY web ./web

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
