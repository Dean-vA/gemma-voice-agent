# vLLM server image WITH audio support. The stock vllm/vllm-openai image omits
# the vllm[audio] extras, so Gemma 4's audio input fails to decode. We add
# libsndfile + soundfile/librosa so WAV/PCM audio loads.
FROM vllm/vllm-openai:latest

USER root
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir soundfile librosa

# entrypoint (vLLM OpenAI API server) is inherited; compose supplies the args.
