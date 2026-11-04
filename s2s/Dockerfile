# The Reachy speech pipeline in one container, on one GPU: what backend.py runs on Windows for
# the Reachy Mini, started the same way. backend.py starts the transcription service and
# llama.cpp itself, then runs Hugging Face speech-to-speech 1.0.0 with Ruxithid's patches and
# Qwen3-TTS in its own process. About 22 GB of GPU memory with everything loaded.
#
# Built on the official llama.cpp CUDA server image (b11429, the nearest published build to
# the Windows b11406). The two Python parts keep separate environments, as on Windows, because
# their pinned torch/transformers versions differ.
FROM ghcr.io/ggml-org/llama.cpp:server-cuda12-b11429

ENV PYTHONUNBUFFERED=1 \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    HF_HOME=/root/.cache/huggingface \
    LLAMA_CACHE=/root/.cache/llama.cpp \
    NLTK_DATA=/usr/local/share/nltk_data

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 libportaudio2 ffmpeg sox curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN uv python install 3.12

# Transcription + speaker id (speaker_id/server.py): Parakeet and TitaNet through NeMo.
RUN uv venv --python 3.12 /opt/stt && \
    uv pip install --python /opt/stt/bin/python \
        --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match \
        torch==2.10.0+cu128 torchaudio==2.10.0+cu128 \
        "nemo_toolkit[asr]==2.7.2" fastapi==0.135.3 uvicorn==0.44.0 soundfile==0.13.1 \
        librosa==0.11.0 python-multipart==0.0.24 numpy==2.4.4

# The speech server: speech-to-speech, Qwen3-TTS, the patches. Pinned to the Windows environment.
COPY s2s/requirements.txt /tmp/requirements.txt
RUN uv venv --python 3.12 /opt/speech && \
    uv pip install --python /opt/speech/bin/python \
        --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match \
        torch==2.11.0+cu128 torchaudio==2.11.0+cu128 -r /tmp/requirements.txt && \
    /opt/speech/bin/python -c "import nltk; [nltk.download(p, download_dir='$NLTK_DATA') for p in ('punkt_tab', 'averaged_perceptron_tagger_eng')]"

# llama-server finds its libraries (libllama-server-impl.so and the ggml backends) next to it
# only when started from /app, the base image's working directory; ours is /srv.
RUN printf '#!/bin/sh\nexport LD_LIBRARY_PATH=/app${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\nexec /app/llama-server "$@"\n' \
        > /usr/local/bin/llama-server && chmod +x /usr/local/bin/llama-server

COPY s2s/app/ /srv/
WORKDIR /srv

# Where backend.py finds each part (on Windows these are its own paths). Nobody is enrolled,
# so the transcript never names a speaker. Logs (table log, llama.cpp) go to /srv/logs.
ENV BACKEND_STT_PYTHON=/opt/stt/bin/python \
    BACKEND_LLAMA_SERVER=/usr/local/bin/llama-server \
    BACKEND_LLM_LOG=/srv/logs/llama.log \
    SPEAKER_ID_DATA=/srv/voices
RUN mkdir -p /srv/logs /srv/voices

EXPOSE 8765
# The base image checks llama.cpp's port; this checks the speech server's. The first start
# downloads the models (about 15 GB), hence the long start period.
HEALTHCHECK --interval=15s --timeout=5s --start-period=1800s --retries=5 \
    CMD bash -c 'echo > /dev/tcp/127.0.0.1/8765' || exit 1
# No on-screen head and no PC audio: there is no screen or Discord on a server, and DM-only
# mode needs PC audio, so it is off too. Add --voice "..." to set the voice.
ENTRYPOINT ["/opt/speech/bin/python", "backend.py", "--no-head", "--no-pc-audio"]
