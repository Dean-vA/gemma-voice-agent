# One TTS engine per image, selected at build time so heavy/conflicting deps
# (especially XTTS/Coqui) never collide. Rebuild with --build-arg TTS_ENGINE=... to switch.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ARG TTS_ENGINE=kokoro
ENV TTS_ENGINE=${TTS_ENGINE} \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 ffmpeg espeak-ng curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Base service deps
RUN pip install --upgrade pip && \
    pip install fastapi "uvicorn[standard]" pydantic numpy soundfile

# Engine-specific deps (only the selected engine is installed)
RUN set -eux; \
    case "$TTS_ENGINE" in \
      kokoro) pip install kokoro misaki ;; \
      piper) pip install piper-tts onnxruntime; \
             mkdir -p /voices; \
             base="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium"; \
             curl -fsSL "$base/en_US-amy-medium.onnx" -o /voices/en_US-amy-medium.onnx; \
             curl -fsSL "$base/en_US-amy-medium.onnx.json" -o /voices/en_US-amy-medium.onnx.json ;; \
      xtts) pip install coqui-tts ;; \
      chatterbox) pip install chatterbox-tts ;; \
      *) echo "unknown TTS_ENGINE: $TTS_ENGINE" && exit 1 ;; \
    esac

COPY tts ./tts

EXPOSE 8200
CMD ["uvicorn", "tts.app:app", "--host", "0.0.0.0", "--port", "8200"]
