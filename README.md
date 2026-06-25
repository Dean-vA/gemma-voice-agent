# Gemma 4 E4B Voice Agent (audio-in → text-out)

A containerized, latency-instrumented conversational agent built on **Gemma 4 E4B**'s
native audio input. Speak → get a text reply. Built to **measure performance** on a
local **RTX 5090** as a stepping stone toward **humanoid-robot conversation**.

- **Two backends, env-switchable**: `vllm` (low TTFT, OpenAI-compatible server) and
  `transformers` (in-process reference/fallback).
- **Audio-native**: Gemma encodes the raw speech directly (perceives tone/intent),
  rather than a cascaded VAD→ASR→LLM pipeline. TTFT is the headline metric and is
  reported per turn.
- **Browser control panel** at `http://localhost:8000` that stands in for the robot:
  push-to-talk or continuous VAD turn-taking, streamed reply, live latency.
- **Multi-turn** conversation memory (prior user audio + assistant text kept,
  trimmed to `MAX_HISTORY_TURNS`).

> The control panel calls the *same* gateway API the real robot will use, so it
> doubles as the robot's integration test.

## Prerequisites

- **Docker Desktop** with the **WSL2 backend + GPU support** enabled.
- Recent **NVIDIA driver** supporting **CUDA 12.8** (required for Blackwell/sm_120).
- A **Hugging Face token** with the **Gemma 4 license accepted** (the repos are gated):
  visit the model page, accept terms, then put the token in `.env`.

Verify GPU passthrough:

```bash
docker run --rm --gpus all pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime \
  python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.is_available())"
# -> NVIDIA GeForce RTX 5090 True
```

## Setup

```bash
cp .env.example .env
# edit .env: set HF_TOKEN, choose BACKEND (vllm | transformers), QUANT_MODE, etc.
```

## Run

**Low-latency path (vLLM + gateway):**

```bash
docker compose --profile vllm up --build
```

**Reference path (Transformers only, in-process):**

```bash
# set BACKEND=transformers in .env first
docker compose --profile transformers up --build
```

First start downloads the model into the `hf-cache` volume (multi-GB, one-time).
Then open the console:

```
http://localhost:8000
```

> Browser mic capture needs a secure context — `localhost` qualifies, so no HTTPS
> setup is needed for local testing.

## API (what the robot will call)

| Endpoint        | Purpose                                                            |
|-----------------|-------------------------------------------------------------------|
| `POST /chat`        | multipart `audio` (+ optional `session_id`, `instruction`) → `{reply, metrics}` |
| `POST /chat/stream` | same input → **SSE** token stream + final metrics                 |
| `POST /reset`       | clear a session's history                                         |
| `GET  /health`      | backend, quant mode, GPU, VRAM                                    |

Example:

```bash
curl -F audio=@samples/hello.wav -F session_id=demo http://localhost:8000/chat
```

## Benchmark

Put 16 kHz mono WAVs in `samples/` (see `samples/README.md`), then:

```bash
pip install requests
python scripts/benchmark.py --iterations 20
```

Outputs p50/p95 **TTFT** and **tokens/sec** — compare `vllm` vs `transformers` by
switching `BACKEND` and re-running.

## Configuration (`.env`)

| Var | Meaning |
|-----|---------|
| `BACKEND` | `vllm` or `transformers` |
| `MODEL_ID` / `QAT_MODEL_ID` | base checkpoint / QAT w4a16 checkpoint |
| `QUANT_MODE` | transformers backend: `qat` \| `bnb4` \| `bf16` |
| `MAX_AUDIO_SECONDS` | per-clip cap (model limit is 30 s) |
| `MAX_HISTORY_TURNS` | conversation memory depth |
| `MAX_NEW_TOKENS` | reply length cap |
| `SYSTEM_PROMPT` | keep short + stable to maximize prefix-cache hits |

## Notes & known caveats

- **Attention backend**: Gemma 4's mixed head dims (256 local / 512 global) disable
  FlashAttention-2; check the vLLM logs for the active kernel and confirm prefix
  caching is hitting the stable system prompt.
- **Exact ids/classes**: the QAT repo id (`QAT_MODEL_ID`) and the Transformers model
  class are resolved at runtime with fallbacks; adjust `.env` if the published id
  differs.
- **If audio-native TTFT is too high for the robot**, the next step is a cascaded
  streaming pipeline (VAD → streaming ASR → text LLM → TTS). The gateway, sessions,
  and metrics here are reusable for it.
```
