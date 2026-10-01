"""FastAPI gateway: audio-in -> text-out Gemma 4 E4B voice agent.

Owns sessions, audio decoding and latency metrics; delegates generation to a
pluggable backend (vLLM or Transformers). Serves the control-panel frontend so
the same API the robot will call can be exercised by hand.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.audio import DecodedAudio, decode_audio, to_wav_bytes
from app.backends import make_backend
from app.backends.base import Turn, TurnState
from app.config import get_settings
from app.metrics import TurnTimer, gpu_info
from app.realtime import RealtimeSession
from app.schemas import ChatResponse, HealthResponse
from app.tts_client import TTSClient, split_sentences
from app.wer import word_errors

log = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

# session_id -> list[Turn]   (in-memory; fine for a single-process test harness)
SESSIONS: dict[str, list[Turn]] = {}

# session_id -> persona system prompt override (additive; sessions without an
# entry use settings.system_prompt exactly as before)
PERSONAS: dict[str, str] = {}


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
    # VAD models for /ws/converse. The HTTP endpoints work without them, so a
    # missing dependency only disables the streamed loop.
    try:
        from app.vad import load_vad

        app.state.vad = await asyncio.to_thread(load_vad, settings)
    except Exception as exc:  # noqa: BLE001
        log.warning("VAD unavailable, /ws/converse disabled: %s", exc)
        app.state.vad = None
    # Optional Parakeet ASR; Gemma's own transcription is always available.
    try:
        from app.asr import load_parakeet

        app.state.parakeet = await asyncio.to_thread(load_parakeet, settings)
    except Exception as exc:  # noqa: BLE001
        log.warning("Parakeet unavailable, transcription falls back to Gemma: %s", exc)
        app.state.parakeet = None
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


def _slim_history(history: list[Turn], keep_audio_turns: int) -> None:
    # Opt-in (-1 = keep all): drop replayed audio from user turns older than the
    # last N, but only when the turn has transcript text to fall back on —
    # without it, audio is the only record of what the user said.
    if keep_audio_turns < 0:
        return
    with_audio = [t for t in history if t.role == "user" and t.audio is not None]
    for turn in with_audio[: max(0, len(with_audio) - keep_audio_turns)]:
        if turn.text:
            turn.audio = None


# Gemma doesn't emit a transcript of the user's speech as a side output; it goes
# straight to a response. To *show* what was said we run a separate short ASR
# pass (Gemma is trained for transcription). Optional, since it costs a call.
_ASR_SYSTEM = "You are a precise speech transcription engine."
_ASR_INSTRUCTION = (
    "Transcribe the spoken audio verbatim. Output only the exact words spoken, "
    "with correct punctuation and casing, and nothing else."
)


def _asr_engine(engine: str | None) -> str:
    """Resolve the requested ASR engine, falling back to Gemma if Parakeet is not loaded."""
    name = (engine or app.state.settings.asr_engine).lower()
    if name == "parakeet" and app.state.parakeet is not None:
        return "parakeet"
    return "gemma"


async def _transcribe(samples, timer: TurnTimer | None = None, engine: str | None = None) -> str:
    backend = app.state.backend
    settings = app.state.settings
    engine = _asr_engine(engine)

    async def _run() -> str:
        if engine == "parakeet":
            return await app.state.parakeet.transcribe(samples)
        out: list[str] = []
        async for chunk in backend.stream(_ASR_SYSTEM, [], samples, _ASR_INSTRUCTION, max(64, settings.max_new_tokens)):
            out.append(chunk)
        return "".join(out).strip()

    if timer is None:
        return await _run()
    # Transcription is a separate model call; time it as its own component.
    async with timer.record_async("asr", engine=engine):
        return await _run()


def _decode(audio_bytes: bytes) -> tuple[DecodedAudio, TurnTimer]:
    """Start the turn's timer and decode the upload to 16 kHz mono samples."""
    settings = app.state.settings
    timer = TurnTimer(app.state.backend.name)
    decoded = decode_audio(audio_bytes, settings.sample_rate, settings.max_audio_seconds)
    timer.m.audio_seconds = decoded.duration_s
    timer.mark_preprocess_done()
    return decoded, timer


async def _hear(samples, instruction: str, timer: TurnTimer, state: TurnState, *,
                transcribe: bool, asr_engine: str, llm_input: str) -> tuple[str | None, bool]:
    """Work out what the user said, as far as this turn needs to know it.

    Sets ``state.user_text`` and returns ``(llm_text, transcribed)``:
    ``llm_text`` is the transcript to send to the LLM in place of the audio
    (cascade mode), or None when the LLM should hear the audio itself.
    """
    mode = (llm_input or app.state.settings.llm_input).lower()
    cascade = mode == "transcript"
    timer.m.llm_input = "transcript" if cascade else "audio"
    state.user_text = instruction
    if not (cascade or transcribe):
        return None, False
    # A cascade is only as fast as its ASR, so it defaults to Parakeet.
    text = await _transcribe(samples, timer, asr_engine or ("parakeet" if cascade else ""))
    state.user_text = text or instruction
    if cascade and not text:
        timer.m.llm_input = "audio"  # nothing recognised: let Gemma listen instead
    return (text or None) if cascade else None, True


async def _stream_reply(session_id: str, samples, instruction: str, timer: TurnTimer,
                        state: TurnState, image_bytes: bytes | None = None, user_text: str | None = None):
    """Stream the reply to one spoken turn, collecting chunks in ``state.parts``."""
    settings = app.state.settings
    history = SESSIONS.setdefault(session_id, [])
    system_prompt = PERSONAS.get(session_id) or settings.system_prompt
    async for chunk in app.state.backend.stream(
        system_prompt, history, samples, instruction,
        settings.max_new_tokens, user_image=image_bytes, user_text=user_text,
    ):
        timer.mark_first_token()
        timer.add_token()  # chunk-granular; refined to token count at finish
        state.parts.append(chunk)
        yield chunk


def commit_turn(session_id: str, state: TurnState, samples, image_bytes: bytes | None = None) -> Turn:
    """Append the finished (or interrupted) turn to the session history.

    Returns the stored assistant turn, so the streamed loop can rewrite it to
    what was actually spoken if the visitor cuts the reply off."""
    settings = app.state.settings
    history = SESSIONS.setdefault(session_id, [])
    reply = Turn(role="assistant", text=state.reply)
    history.append(Turn(role="user", text=state.user_text, audio=samples, image=image_bytes))
    history.append(reply)
    _trim_history(history, settings.max_history_turns)
    _slim_history(history, settings.history_keep_audio_turns)
    return reply


async def turn_events(session_id: str, samples, instruction: str, timer: TurnTimer, *,
                      transcribe: bool = False, asr_engine: str = "", speak: bool = False, engine: str = "",
                      image_bytes: bytes | None = None, state: TurnState | None = None,
                      commit: bool = True, llm_input: str = "", user_note: str = ""):
    """Run one turn and yield ``(event, data)`` pairs.

    Events: ``transcript`` (user's words, if requested), ``token`` (text),
    ``audio`` (base64 wav per sentence, when ``speak``), ``done`` (metrics).
    Speaks sentence-by-sentence so the first audio lands before the full reply.
    Shared by the SSE endpoints and the streamed /ws/converse loop; the latter
    passes ``state`` and ``commit=False`` because it decides itself when (and
    whether) the turn enters the history. ``user_note`` is stored as the user's
    line when there is no transcript (e.g. a greeting made from silence).
    """
    state = state if state is not None else TurnState()
    llm_text, transcribed = await _hear(samples, instruction, timer, state, transcribe=transcribe,
                                        asr_engine=asr_engine, llm_input=llm_input)
    if transcribed:
        yield "transcript", {"text": state.user_text}

    tts: TTSClient | None = _pick_tts(engine) if speak else None
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
        return "audio", {"index": index, "sentence": sentence, "wav_base64": _b64(wav),
                         "synth_ms": round(t["client_ms"], 1),
                         "server_ms": round(t["server_ms"], 1) if "server_ms" in t else None}

    async for chunk in _stream_reply(session_id, samples, instruction, timer, state, image_bytes, llm_text):
        yield "token", {"text": chunk}
        if tts is None:
            continue
        buffer += chunk
        sentences, buffer = split_sentences(buffer)
        for s in sentences:
            if not s.strip():
                continue
            yield await _speak(idx, s)
            idx += 1
    # flush any trailing partial sentence
    tail = buffer.strip()
    if tts is not None and tail:
        yield await _speak(idx, tail)

    reply = state.reply
    # finish() aggregates the per-call TTS timings and resolves the engine
    # name from the X-Engine header, so no extra health round-trip is needed.
    metrics = timer.finish(output_tokens=_approx_tokens(reply))
    if commit:
        if user_note and not state.user_text:
            state.user_text = user_note
        commit_turn(session_id, state, samples, image_bytes)
    yield "done", {"reply": reply, "metrics": metrics.as_dict()}


async def _read_image(image: UploadFile | None) -> bytes | None:
    """Read an optional image upload to bytes (None if absent/empty)."""
    if image is None:
        return None
    data = await image.read()
    return data or None


@app.post("/chat", response_model=ChatResponse)
async def chat(audio: UploadFile, session_id: str = Form(None), instruction: str = Form(""),
               transcribe: bool = Form(False), asr_engine: str = Form(""), llm_input: str = Form(""),
               image: UploadFile = File(None)):
    sid = session_id or uuid.uuid4().hex
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(400, "empty audio upload")

    image_bytes = await _read_image(image)
    decoded, timer = _decode(audio_bytes)
    state = TurnState()
    llm_text, _ = await _hear(decoded.samples, instruction, timer, state, transcribe=transcribe,
                              asr_engine=asr_engine, llm_input=llm_input)

    async for _ in _stream_reply(sid, decoded.samples, instruction, timer, state, image_bytes, llm_text):
        pass

    reply = state.reply
    metrics = timer.finish(output_tokens=_approx_tokens(reply))
    commit_turn(sid, state, decoded.samples, image_bytes)

    return ChatResponse(session_id=sid, reply=reply, transcript=state.user_text or None, metrics=metrics.as_dict())


def _sse_turn(sid: str, decoded: DecodedAudio, timer: TurnTimer, instruction: str, **kwargs) -> StreamingResponse:
    async def _sse():
        yield _event("session", {"session_id": sid})
        async for event, data in turn_events(sid, decoded.samples, instruction, timer, **kwargs):
            yield _event(event, data)

    return StreamingResponse(_sse(), media_type="text/event-stream")


@app.post("/chat/stream")
async def chat_stream(audio: UploadFile, session_id: str = Form(None), instruction: str = Form(""),
                      transcribe: bool = Form(False), asr_engine: str = Form(""), llm_input: str = Form(""),
                      image: UploadFile = File(None)):
    sid = session_id or uuid.uuid4().hex
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(400, "empty audio upload")

    image_bytes = await _read_image(image)
    decoded, timer = _decode(audio_bytes)
    return _sse_turn(sid, decoded, timer, instruction, transcribe=transcribe, asr_engine=asr_engine,
                     llm_input=llm_input, image_bytes=image_bytes)


@app.post("/converse")
async def converse(audio: UploadFile, session_id: str = Form(None), instruction: str = Form(""),
                   transcribe: bool = Form(False), asr_engine: str = Form(""), llm_input: str = Form(""),
                   engine: str = Form(""), image: UploadFile = File(None), user_note: str = Form("")):
    """Voice loop: audio in -> Gemma streams text -> sentence-chunked TTS -> audio out.

    Streams SSE: `transcript` (user's words, if requested), `token` (text),
    `audio` (base64 wav per sentence), `done` (metrics). `engine` selects
    which TTS service to use. `user_note` is what the history records as the
    user's turn when nothing was transcribed (e.g. "(the visitor walked up)").
    """
    sid = session_id or uuid.uuid4().hex
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(400, "empty audio upload")

    image_bytes = await _read_image(image)
    decoded, timer = _decode(audio_bytes)
    return _sse_turn(sid, decoded, timer, instruction, transcribe=transcribe, asr_engine=asr_engine,
                     llm_input=llm_input, speak=True, engine=engine, image_bytes=image_bytes,
                     user_note=user_note)


@app.websocket("/ws/converse")
async def ws_converse(ws: WebSocket):
    """Streamed voice loop: the client sends a continuous 16 kHz PCM mic stream
    and the server does VAD, turn detection and barge-in (see app/realtime.py)."""
    await RealtimeSession(ws, app.state, turn_events=turn_events, commit_turn=commit_turn).run()


@app.get("/asr/engines")
async def asr_engines():
    """List the transcription engines and which are loaded."""
    return {"engines": [{"name": "gemma", "available": True},
                        {"name": "parakeet", "available": app.state.parakeet is not None}],
            "default": _asr_engine(None), "llm_input": app.state.settings.llm_input}


@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile, engine: str = Form(""), reference: str = Form("")):
    """Transcribe one clip with the chosen ASR engine and report how long it took.
    With ``reference`` (what was actually said) the word errors are scored too.

    Unlike the `transcribe` flag on the chat endpoints this never falls back:
    asking for an engine that is not loaded is an error, so a result can never
    be attributed to the wrong engine.
    """
    name = (engine or app.state.settings.asr_engine).lower()
    if name not in ("gemma", "parakeet"):
        raise HTTPException(400, f"unknown ASR engine {name!r}")
    if name == "parakeet" and app.state.parakeet is None:
        raise HTTPException(503, "parakeet is not loaded (see gateway logs)")
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(400, "empty audio upload")
    settings = app.state.settings
    decoded = decode_audio(audio_bytes, settings.sample_rate, settings.max_audio_seconds)
    t0 = time.perf_counter()
    text = await _transcribe(decoded.samples, engine=name)
    ms = (time.perf_counter() - t0) * 1000.0
    result = {"engine": name, "text": text, "ms": round(ms, 1), "audio_seconds": round(decoded.duration_s, 3)}
    if reference:
        result |= word_errors(reference, text)
    return result


def _asr_eval_dir() -> Path:
    return Path(app.state.settings.asr_eval_dir)


def _asr_eval_manifest() -> dict:
    path = _asr_eval_dir() / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@app.get("/eval/asr/clips")
async def asr_eval_clips():
    """Read-aloud clips recorded so far: {id: {text, file, audio_seconds}}."""
    return _asr_eval_manifest()


@app.get("/eval/asr/clips/{clip_id}.wav")
async def asr_eval_clip_audio(clip_id: str):
    """The saved audio of one read-aloud clip (the latency eval replays them)."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", clip_id):
        raise HTTPException(400, "bad clip id")
    path = _asr_eval_dir() / f"{clip_id}.wav"
    if not path.exists():
        raise HTTPException(404, "no such clip")
    return FileResponse(path, media_type="audio/wav")


@app.post("/eval/asr/clips")
async def asr_eval_save_clip(audio: UploadFile, id: str = Form(...), text: str = Form(...)):
    """Save one read-aloud clip (16 kHz mono WAV) with its reference text, for
    scripts/asr_eval.py. Re-recording an id overwrites it."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", id):
        raise HTTPException(400, "bad clip id")
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(400, "empty audio upload")
    settings = app.state.settings
    decoded = decode_audio(audio_bytes, settings.sample_rate, settings.max_audio_seconds)
    folder = _asr_eval_dir()
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{id}.wav").write_bytes(to_wav_bytes(decoded.samples, settings.sample_rate))
    manifest = _asr_eval_manifest()
    manifest[id] = {"text": text, "file": f"{id}.wav", "audio_seconds": round(decoded.duration_s, 3)}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"ok": True, "id": id, "clips": len(manifest)}


@app.get("/tts/engines")
async def tts_engines():
    """List configured TTS engines and which are currently reachable."""
    clients = app.state.tts_clients
    healths = await asyncio.gather(*(c.health() for c in clients.values()))
    engines = [
        {"name": name, "reachable": h.get("reachable", False),
         "sample_rate": h.get("sample_rate"), "engine": h.get("engine")}
        for name, h in zip(clients.keys(), healths)
    ]
    return {"engines": engines, "default": app.state.settings.tts_engine}


@app.post("/persona")
async def persona(session_id: str = Form(None), system_prompt: str = Form(""),
                  reset_history: bool = Form(True)):
    """Set (or clear) a per-session persona system prompt. Empty prompt reverts
    to the configured default. Creates the session id if needed so a persona
    can be chosen before the first spoken turn."""
    sid = session_id or uuid.uuid4().hex
    prompt = system_prompt.strip()
    if prompt:
        PERSONAS[sid] = prompt
    else:
        PERSONAS.pop(sid, None)
    if reset_history:
        SESSIONS.pop(sid, None)
    return {"ok": True, "session_id": sid, "default": not prompt,
            "history_reset": reset_history}


@app.post("/reset")
async def reset(session_id: str = Form(...)):
    SESSIONS.pop(session_id, None)
    PERSONAS.pop(session_id, None)
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
