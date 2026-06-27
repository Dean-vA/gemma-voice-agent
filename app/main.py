"""FastAPI gateway: audio-in -> text-out Gemma 4 E4B voice agent.

Owns sessions, audio decoding and latency metrics; delegates generation to a
pluggable backend (vLLM or Transformers). Serves the control-panel frontend so
the same API the robot will call can be exercised by hand.
"""
from __future__ import annotations

import base64
import json
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.audio import decode_audio
from app.backends import make_backend
from app.backends.base import Turn
from app.config import get_settings
from app.metrics import TurnTimer, gpu_info
from app.schemas import ChatResponse, HealthResponse
from app.tts_client import TTSClient, split_sentences

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

# session_id -> list[Turn]   (in-memory; fine for a single-process test harness)
SESSIONS: dict[str, list[Turn]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    backend = make_backend(settings)
    await backend.load()
    app.state.backend = backend
    app.state.settings = settings
    # One client per configured TTS engine; reachability is checked on demand.
    app.state.tts_clients = {
        name: TTSClient(url, settings.tts_voice) for name, url in settings.tts_engines.items()
    }
    yield


def _pick_tts(engine: str | None) -> TTSClient:
    clients = app.state.tts_clients
    name = engine or app.state.settings.tts_engine
    return clients.get(name) or clients[app.state.settings.tts_engine]


app = FastAPI(title="Gemma 4 E4B Voice Agent", lifespan=lifespan)


def _trim_history(history: list[Turn], max_turns: int) -> list[Turn]:
    # Keep the most recent 2*max_turns messages (user+assistant pairs).
    limit = max_turns * 2
    if len(history) > limit:
        del history[: len(history) - limit]
    return history


# Gemma doesn't emit a transcript of the user's speech as a side output; it goes
# straight to a response. To *show* what was said we run a separate short ASR
# pass (Gemma is trained for transcription). Optional, since it costs a call.
_ASR_SYSTEM = "You are a precise speech transcription engine."
_ASR_INSTRUCTION = (
    "Transcribe the spoken audio verbatim. Output only the exact words spoken, "
    "with correct punctuation and casing, and nothing else."
)


async def _transcribe(samples, timer: TurnTimer | None = None) -> str:
    backend = app.state.backend
    settings = app.state.settings

    async def _run() -> str:
        out: list[str] = []
        async for chunk in backend.stream(_ASR_SYSTEM, [], samples, _ASR_INSTRUCTION, max(64, settings.max_new_tokens)):
            out.append(chunk)
        return "".join(out).strip()

    if timer is None:
        return await _run()
    # The transcription pass is a separate LLM call; time it as its own component.
    async with timer.record_async("asr"):
        return await _run()


async def _run_turn(session_id: str, audio_bytes: bytes, instruction: str, image_bytes: bytes | None = None):
    """Decode audio, stream a reply, update history. Yields (chunk, timer)."""
    settings = app.state.settings
    backend = app.state.backend

    timer = TurnTimer(backend.name)
    decoded = decode_audio(audio_bytes, settings.sample_rate, settings.max_audio_seconds)
    timer.m.audio_seconds = decoded.duration_s
    timer.mark_preprocess_done()

    history = SESSIONS.setdefault(session_id, [])

    parts: list[str] = []

    async def _gen():
        async for chunk in backend.stream(
            settings.system_prompt, history, decoded.samples, instruction,
            settings.max_new_tokens, user_image=image_bytes,
        ):
            timer.mark_first_token()
            timer.add_token()  # chunk-granular; refined to token count at finish
            parts.append(chunk)
            yield chunk

    return decoded, history, parts, timer, _gen


async def _read_image(image: UploadFile | None) -> bytes | None:
    """Read an optional image upload to bytes (None if absent/empty)."""
    if image is None:
        return None
    data = await image.read()
    return data or None


@app.post("/chat", response_model=ChatResponse)
async def chat(audio: UploadFile, session_id: str = Form(None), instruction: str = Form(""),
               transcribe: bool = Form(False), image: UploadFile = File(None)):
    sid = session_id or uuid.uuid4().hex
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(400, "empty audio upload")

    image_bytes = await _read_image(image)
    decoded, history, parts, timer, gen = await _run_turn(sid, audio_bytes, instruction, image_bytes)
    user_text = await _transcribe(decoded.samples, timer) if transcribe else instruction

    async for _ in gen():
        pass

    reply = "".join(parts)
    metrics = timer.finish(output_tokens=_approx_tokens(reply))

    history.append(Turn(role="user", text=user_text or instruction, audio=decoded.samples, image=image_bytes))
    history.append(Turn(role="assistant", text=reply))
    _trim_history(history, app.state.settings.max_history_turns)

    return ChatResponse(session_id=sid, reply=reply, transcript=user_text or None, metrics=metrics.as_dict())


@app.post("/chat/stream")
async def chat_stream(audio: UploadFile, session_id: str = Form(None), instruction: str = Form(""),
                      transcribe: bool = Form(False), image: UploadFile = File(None)):
    sid = session_id or uuid.uuid4().hex
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(400, "empty audio upload")

    image_bytes = await _read_image(image)
    decoded, history, parts, timer, gen = await _run_turn(sid, audio_bytes, instruction, image_bytes)

    async def _sse():
        yield _event("session", {"session_id": sid})
        user_text = instruction
        if transcribe:
            user_text = await _transcribe(decoded.samples, timer) or instruction
            yield _event("transcript", {"text": user_text})
        async for chunk in gen():
            yield _event("token", {"text": chunk})
        reply = "".join(parts)
        metrics = timer.finish(output_tokens=_approx_tokens(reply))
        history.append(Turn(role="user", text=user_text, audio=decoded.samples, image=image_bytes))
        history.append(Turn(role="assistant", text=reply))
        _trim_history(history, app.state.settings.max_history_turns)
        yield _event("done", {"reply": reply, "metrics": metrics.as_dict()})

    return StreamingResponse(_sse(), media_type="text/event-stream")


@app.post("/converse")
async def converse(audio: UploadFile, session_id: str = Form(None), instruction: str = Form(""),
                   transcribe: bool = Form(False), engine: str = Form(""), image: UploadFile = File(None)):
    """Voice loop: audio in -> Gemma streams text -> sentence-chunked TTS -> audio out.

    Streams SSE: `transcript` (user's words, if requested), `token` (text),
    `audio` (base64 wav per sentence), `done` (metrics). Speaks sentence-by-
    sentence so the first audio lands before the full reply. `engine` selects
    which TTS service to use.
    """
    sid = session_id or uuid.uuid4().hex
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(400, "empty audio upload")

    image_bytes = await _read_image(image)
    decoded, history, parts, timer, gen = await _run_turn(sid, audio_bytes, instruction, image_bytes)
    tts: TTSClient = _pick_tts(engine)

    async def _sse():
        yield _event("session", {"session_id": sid})
        user_text = instruction
        if transcribe:
            user_text = await _transcribe(decoded.samples, timer) or instruction
            yield _event("transcript", {"text": user_text})
        buffer = ""
        idx = 0
        first_audio = False

        async def _speak(index: int, sentence: str):
            nonlocal first_audio
            wav, t = await tts.synthesize(sentence)
            timer.add_tts_segment(text=sentence, **t)
            if not first_audio:
                timer.m.time_to_first_audio_ms = (time.perf_counter() - timer._t_start) * 1000.0
                first_audio = True
            return _event("audio", {"index": index, "sentence": sentence, "wav_base64": _b64(wav),
                                    "synth_ms": round(t["client_ms"], 1),
                                    "server_ms": round(t["server_ms"], 1) if "server_ms" in t else None})

        async for chunk in gen():
            yield _event("token", {"text": chunk})
            buffer += chunk
            sentences, buffer = split_sentences(buffer)
            for s in sentences:
                if not s.strip():
                    continue
                yield await _speak(idx, s)
                idx += 1
        # flush any trailing partial sentence
        tail = buffer.strip()
        if tail:
            yield await _speak(idx, tail)

        reply = "".join(parts)
        # finish() aggregates the per-call TTS timings and resolves the engine
        # name from the X-Engine header, so no extra health round-trip is needed.
        metrics = timer.finish(output_tokens=_approx_tokens(reply))
        history.append(Turn(role="user", text=user_text, audio=decoded.samples, image=image_bytes))
        history.append(Turn(role="assistant", text=reply))
        _trim_history(history, app.state.settings.max_history_turns)
        yield _event("done", {"reply": reply, "metrics": metrics.as_dict()})

    return StreamingResponse(_sse(), media_type="text/event-stream")


@app.get("/tts/engines")
async def tts_engines():
    """List configured TTS engines and which are currently reachable."""
    import asyncio

    clients = app.state.tts_clients
    healths = await asyncio.gather(*(c.health() for c in clients.values()))
    engines = [
        {"name": name, "reachable": h.get("reachable", False),
         "sample_rate": h.get("sample_rate"), "engine": h.get("engine")}
        for name, h in zip(clients.keys(), healths)
    ]
    return {"engines": engines, "default": app.state.settings.tts_engine}


@app.post("/reset")
async def reset(session_id: str = Form(...)):
    SESSIONS.pop(session_id, None)
    return {"ok": True, "session_id": session_id}


@app.get("/health", response_model=HealthResponse)
async def health():
    settings = app.state.settings
    backend = app.state.backend
    return HealthResponse(
        status="ok",
        backend=backend.name,
        quant_mode=settings.quant_mode,
        settings={
            "model_id": settings.model_id,
            "qat_model_id": settings.qat_model_id,
            "max_audio_seconds": settings.max_audio_seconds,
            "max_new_tokens": settings.max_new_tokens,
            "sample_rate": settings.sample_rate,
        },
        gpu=gpu_info(),
        backend_info=await backend.health(),
        tts=await _pick_tts(settings.tts_engine).health(),
    )


def _event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _approx_tokens(text: str) -> int:
    # Cheap proxy for token count when the backend streams text chunks rather
    # than raw tokens; good enough for tokens/sec instrumentation.
    return max(1, round(len(text) / 4))


# Serve the control-panel frontend at "/". Mounted last so API routes win.
if WEB_DIR.exists():
    @app.get("/")
    async def index():
        return FileResponse(WEB_DIR / "index.html")

    app.mount("/web", StaticFiles(directory=WEB_DIR), name="web")
