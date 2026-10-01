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
  push-to-talk or continuous turn-taking (server-side VAD, barge-in), streamed
  reply, live latency.
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

> Browser mic **and webcam** capture need a secure context — `localhost` qualifies,
> so no HTTPS setup is needed for local testing.

**Show the robot something (image input).** Gemma 4 E4B is multimodal, so the console
can attach a still image to a spoken turn. Click **📷** in the control bar, **📸 Snap**
a webcam frame, then hold-to-talk and ask about it (e.g. "what am I holding?"). The frame
is sent with that turn only. No extra setup — the vLLM server already accepts an image
alongside audio.

## API (what the robot will call)

| Endpoint        | Purpose                                                            |
|-----------------|-------------------------------------------------------------------|
| `POST /chat`        | multipart `audio` (+ optional `session_id`, `instruction`, `image`) → `{reply, metrics}` |
| `POST /chat/stream` | same input → **SSE** token stream + final metrics                 |
| `POST /converse`    | same input (+ `engine`) → **SSE** text + sentence-chunked TTS audio |
| `POST /transcribe`  | `audio` + `engine` (`gemma` \| `parakeet`) → `{text, ms}` (+ WER given `reference`) |
| `WS   /ws/converse` | continuous 16 kHz PCM mic stream in → turn events, text and TTS audio out |
| `POST /reset`       | clear a session's history                                         |
| `GET  /health`      | backend, quant mode, GPU, VRAM                                    |

Example:

```bash
curl -F audio=@samples/hello.wav -F session_id=demo http://localhost:8000/chat
```

### Streamed voice loop (`/ws/converse`)

The **Continuous** toggle streams the mic to the gateway, which does the turn-taking
itself — the same split as the Reachy Mini conversation app and its
`huggingface/speech-to-speech` server, so a robot only has to stream audio:

- **Silero VAD** (32 ms chunks) finds speech; a 500 ms pre-roll keeps the first word.
- **Smart Turn v3** judges whether a pause really ends the turn. The reply is
  prepared straight away but held for a grace period: `REOPEN_MS` (800 ms) when the
  turn sounds complete, `SMART_TURN_MAX_WAIT_MS` (2 s) when it doesn't. Speech
  resuming before the reply is released continues the same turn.
- **Barge-in**: once the reply has started, talking over it cancels generation and
  stops playback. It relies on the client's echo cancellation; set
  `VAD_BARGE_IN=false` if the robot hears itself.
- Utterances with less than `VAD_MIN_SPEECH_MS` (384 ms) of speech are ignored, as
  upstream — lower it if short answers like "yes" get dropped.

Both models run on CPU (onnxruntime), so they take no VRAM. The message protocol is
documented at the top of `app/realtime.py`. To exercise it without a microphone:

```bash
pip install websockets soundfile librosa
python scripts/ws_client.py samples/hello.wav                 # one turn
python scripts/ws_client.py a.wav b.wav --barge-in            # talk over the reply
```

Tests for the segmentation logic: `pip install pytest && pytest tests`.

### Transcription engines (Gemma vs Parakeet)

Gemma answers from the raw audio; the *transcript* shown in the console is a
separate pass. Two engines can produce it, selectable in the UI (or with
`asr_engine` on the API, default `ASR_ENGINE`):

- `gemma` — a second call to the LLM asking for a verbatim transcript.
- `parakeet` — Parakeet TDT 0.6b v3 through `nano-parakeet`, loaded in the gateway
  (the default STT of the Reachy Mini backend). Uses some VRAM; `PARAKEET=false`
  skips loading it.

To compare them on your own voice, switch the console to **ASR eval** (the tab
next to Conversation): hold the talk button and read each test sentence aloud.
Every clip is saved to `samples/asr/` and transcribed by both engines, with its
word error rate and latency. For repeated timings and a CSV:

```bash
pip install requests
python scripts/asr_eval.py --runs 5
```

With Continuous on, eval mode is hands-free: read the sentence and the same
server-side VAD cuts the clip and moves to the next one.

The test sentences live in `web/asr-script.js`. `POST /transcribe` (`audio`, `engine`,
optional `reference`) is the endpoint both use.

### Cascade mode and time to first speech

By default Gemma answers from the audio itself. The **🧠 hears audio / reads
transcript** selector (or `llm_input=transcript` on the API, default `LLM_INPUT`)
switches to a cascade: Parakeet transcribes the turn and Gemma answers from that
text, with no audio in the prompt.

To compare the two on time to first speech, open the **Speech latency** tab: it
replays the clips recorded in ASR eval through `/converse` in both
configurations and reports, per clip and overall, how long until the first
spoken sentence is ready (with ASR, LLM first token and first-sentence TTS
broken out). The same as a script, with a CSV:

```bash
python scripts/speech_latency_eval.py
```

Repeat runs of one clip hit vLLM's prompt cache and look faster than a fresh
utterance, so the default is one run per clip.

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
| `VAD_*`, `SMART_TURN*`, `REOPEN_MS` | streamed-loop turn detection (see `.env.example`) |
| `ASR_ENGINE`, `PARAKEET*` | transcript engine and whether Parakeet is loaded |
| `LLM_INPUT` | `audio` (Gemma hears the speech) or `transcript` (cascade via Parakeet) |

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
