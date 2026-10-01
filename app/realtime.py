"""Streamed voice loop over a WebSocket: VAD, turn detection and barge-in.

The client (web console today, the robot later) sends its microphone as a
continuous stream and just reacts to events — the same division of labour as
the Reachy Mini conversation app and its backend, huggingface/speech-to-speech,
whose turn handling (VAD/vad_handler.py, pipeline/speculative_turns.py) this
follows.

Client -> server
    binary frames            int16 mono PCM at 16 kHz, any frame size
    {"type": "config", session_id?, instruction?, transcribe?, asr_engine?, llm_input?, speak?, engine?,
                       respond?,    respond=false: detect turns only, generate no reply
                       vad?}        per-connection overrides of the gateway's turn-detection
                                    settings (any of VAD_OPTIONS below), applied live
    {"type": "image", "data": <base64 jpeg/png> | null}   still for the next turn
    {"type": "playback", "active": bool}                  client is playing TTS
    {"type": "played", "state": "cut", "index": int, "fraction": float}
                                    the client cut reply sentence `index` off after
                                    `fraction` of it (barge-in); the history then keeps
                                    only what was actually spoken

Server -> client  ({"type": <event>, ...})
    session         {session_id}
    vad_config      {<every VAD option and its effective value>, smart_turn_available}
    speech_started  {audio_start_ms}                     once per turn
    speech_stopped  {audio_start_ms, audio_end_ms}       once per turn, when it is final
    cancelled       {reason: "barge_in"}                 stop/drop the current reply
    transcript / token / audio / done                    same payloads as the SSE endpoints
    error           {message}

Turn logic. Silero closes a segment after a short silence; that is only a
*soft* end. Smart Turn then picks a grace period (short if the turn sounds
complete, long if not). The reply is prepared speculatively during the grace
but nothing is released to the client until it has passed, so:
  * speech resuming before release reopens the same turn — the earlier audio is
    kept, the speculative reply is dropped, and the client never saw a stop;
  * once output has been released the turn is committed: new speech starts a
    new turn, and if the reply is still in flight that is a barge-in, which
    cancels it and keeps what was said so far in the history.
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from app.backends.base import TurnState
from app.metrics import TurnTimer
from app.vad.segmenter import CHUNK_SAMPLES, SAMPLE_RATE, Segmenter, SpeechStarted, SpeechStopped, VADConfig

log = logging.getLogger(__name__)

# How long a release waits on unconfirmed speech before giving up on it
# (SpeculativeTurnTracker._PENDING_REOPEN_WAIT_TIMEOUT_S upstream).
_PENDING_REOPEN_TIMEOUT_S = 2.0


# Turn-detection options a client may override for its own connection: name ->
# (type, min, max). Everything else stays whatever the gateway was started with.
VAD_OPTIONS: dict[str, tuple[type, float, float]] = {
    "vad_threshold": (float, 0.05, 0.95),
    "vad_min_silence_ms": (int, 32, 3000),
    "vad_min_speech_ms": (int, 32, 3000),
    "vad_min_speech_continuation_ms": (int, 0, 3000),
    "vad_speech_pad_ms": (int, 0, 2000),
    "smart_turn": (bool, 0, 1),
    "smart_turn_threshold": (float, 0.0, 1.0),
    "smart_turn_max_wait_ms": (int, 1, 10000),
    "smart_turn_incomplete_delay_ms": (int, 0, 10000),
    "reopen_ms": (int, 0, 10000),
    "unanswered_reopen_ms": (int, 0, 60000),
    "vad_barge_in": (bool, 0, 1),
}


def _ms(samples: int) -> float:
    return samples * 1000.0 / SAMPLE_RATE


@dataclass
class _Turn:
    start_sample: int
    audio: np.ndarray | None = None  # everything said so far this turn (all segments)
    end_sample: int | None = None    # stream offset of the latest soft end
    image: bytes | None = None
    state: TurnState = field(default_factory=TurnState)
    task: asyncio.Task | None = None
    committed: bool = False   # output released; the turn can no longer reopen
    finished: bool = False    # reply complete (or failed)
    sentences: dict = field(default_factory=dict)   # index -> sentence text sent as audio
    stored_reply: object | None = None              # the assistant Turn in the history


class RealtimeSession:
    def __init__(self, ws: WebSocket, app_state, *, turn_events, commit_turn) -> None:
        self.ws = ws
        self.app_state = app_state
        self.settings = app_state.settings
        self._turn_events = turn_events
        self._commit_turn = commit_turn
        self._send_lock = asyncio.Lock()

        self.session_id: str | None = None
        self.instruction = ""
        self.transcribe = False
        self.asr_engine = ""
        self.llm_input = ""
        self.speak = True
        self.engine = ""
        self.respond = True

        self._image: bytes | None = None
        self._playing = False             # client reports TTS playback in progress
        self._turn: _Turn | None = None
        self._last_reply: _Turn | None = None   # newest turn whose reply is in the history
        self._remainder = np.zeros(0, dtype=np.float32)
        self._gated = False

    async def run(self) -> None:
        await self.ws.accept()
        vad = getattr(self.app_state, "vad", None)
        if vad is None or self.settings.sample_rate != SAMPLE_RATE:
            await self._send("error", {"message": "server-side VAD is unavailable (see gateway logs)"})
            await self.ws.close()
            return

        self.silero = vad.silero.stream()
        self._smart_turn_model = vad.smart_turn
        # This connection's turn-detection options: the gateway's settings,
        # until the client overrides some in a config message.
        self.opts = {name: getattr(self.settings, name) for name in VAD_OPTIONS}
        self.segmenter = Segmenter(self._vad_config())
        self.segmenter.reopenable = self._reopenable
        try:
            while True:
                msg = await self.ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    await self._on_audio(msg["bytes"])
                elif msg.get("text"):
                    await self._on_message(json.loads(msg["text"]))
        except WebSocketDisconnect:
            pass
        finally:
            if self._turn is not None:
                await self._cancel(self._turn)

    # ---- per-connection turn-detection options ------------------------------
    @property
    def smart_turn(self):
        """The Smart Turn model, or None when unavailable or switched off for this connection."""
        return self._smart_turn_model if self.opts["smart_turn"] else None

    @property
    def _unanswered_reopen_ms(self) -> int:
        o = self.opts
        return max(o["reopen_ms"], o["unanswered_reopen_ms"],
                   o["smart_turn_max_wait_ms"] if self.smart_turn is not None else 0)

    def _vad_config(self) -> VADConfig:
        o = self.opts
        return VADConfig(threshold=o["vad_threshold"], min_silence_ms=o["vad_min_silence_ms"],
                         min_speech_ms=o["vad_min_speech_ms"],
                         min_speech_continuation_ms=o["vad_min_speech_continuation_ms"],
                         speech_pad_ms=o["vad_speech_pad_ms"])

    async def _apply_vad_options(self, overrides: dict) -> None:
        for name, value in overrides.items():
            spec = VAD_OPTIONS.get(name)
            if spec is None or value is None:
                continue
            kind, lo, hi = spec
            try:
                self.opts[name] = bool(value) if kind is bool else kind(min(hi, max(lo, float(value))))
            except (TypeError, ValueError):
                continue
        self.segmenter.configure(self._vad_config())
        await self._send("vad_config", {**self.opts, "smart_turn_available": self._smart_turn_model is not None})

    # ---- client messages -----------------------------------------------------
    async def _on_message(self, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "config":
            self.instruction = msg.get("instruction") or ""
            self.transcribe = bool(msg.get("transcribe", self.transcribe))
            self.asr_engine = msg.get("asr_engine") or ""
            self.llm_input = msg.get("llm_input") or ""
            self.speak = bool(msg.get("speak", self.speak))
            self.engine = msg.get("engine") or ""
            self.respond = bool(msg.get("respond", True))
            if isinstance(msg.get("vad"), dict):
                await self._apply_vad_options(msg["vad"])
            # No id = the client wants a fresh session (first connect, or Reset).
            sid = msg.get("session_id") or uuid.uuid4().hex
            if sid != self.session_id:
                self.session_id = sid
                await self._send("session", {"session_id": sid})
        elif kind == "image":
            data = msg.get("data")
            self._image = base64.b64decode(data) if data else None
        elif kind == "playback":
            self._playing = bool(msg.get("active"))
        elif kind == "played":
            self._on_played(msg)

    def _on_played(self, msg: dict) -> None:
        """The client cut a reply off mid-playback: make the history say only
        what was actually spoken. Gemma writes a reply far faster than it is
        spoken, so without this it would believe the visitor heard all of it."""
        turn = self._last_reply
        if msg.get("state") != "cut" or turn is None or turn.stored_reply is None:
            return
        try:
            index = int(msg.get("index"))
            fraction = min(1.0, max(0.0, float(msg.get("fraction", 0.0))))
        except (TypeError, ValueError):
            return
        if index not in turn.sentences:
            return  # a clip from an older reply
        spoken = [turn.sentences[i] for i in sorted(turn.sentences) if i < index]
        words = turn.sentences[index].split()
        partial = " ".join(words[: int(len(words) * fraction)])
        said = " ".join(t for t in [*spoken, partial] if t).strip()
        turn.stored_reply.text = (said + " — " if said else "") + "[interrupted by the visitor]"
        log.info("barge-in: history keeps what was spoken (%d of %d sentences, %.0f%% of the last)",
                 len(spoken), len(turn.sentences), fraction * 100)

    async def _on_audio(self, data: bytes) -> None:
        pcm = np.frombuffer(data[: len(data) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0
        buf = np.concatenate([self._remainder, pcm]) if self._remainder.size else pcm
        n = len(buf) // CHUNK_SAMPLES * CHUNK_SAMPLES
        self._remainder = buf[n:]
        for i in range(0, n, CHUNK_SAMPLES):
            await self._on_chunk(buf[i:i + CHUNK_SAMPLES])

    async def _on_chunk(self, chunk: np.ndarray) -> None:
        # Barge-in disabled: don't listen while the robot is replying/speaking,
        # so its own voice can't start a turn.
        if not self.opts["vad_barge_in"] and self._replying():
            if not self._gated:
                self._gated = True
                self.silero.reset()
            self.segmenter.skip(len(chunk))
            return
        self._gated = False

        for ev in self.segmenter.feed(chunk, self.silero.prob(chunk)):
            if isinstance(ev, SpeechStarted):
                await self._on_speech_started(ev)
            elif isinstance(ev, SpeechStopped):
                await self._on_speech_stopped(ev)

    def _replying(self) -> bool:
        turn = self._turn
        return self._playing or (turn is not None and turn.committed and not turn.finished)

    # ---- turn detection ------------------------------------------------------
    def _reopenable(self, start_sample: int) -> bool:
        """Can speech starting at this stream offset still continue the current turn?

        Any turn whose output hasn't been released may reopen, up to
        unanswered_reopen_ms after its last soft end (on the audio clock).
        """
        turn = self._turn
        if turn is None or turn.committed or turn.end_sample is None:
            return False
        return max(0.0, _ms(start_sample - turn.end_sample)) <= self._unanswered_reopen_ms

    async def _on_speech_started(self, ev: SpeechStarted) -> None:
        turn = self._turn
        if turn is not None and not turn.committed and (ev.candidate or self._reopenable(ev.start_sample)):
            # The user carried on: same turn, the speculative reply is stale.
            # The client still has this turn's speech open, so it hears nothing.
            await self._cancel(turn)
            return

        if turn is not None and not turn.finished:
            # Barge-in on a reply in flight. (Upstream leaves the reply running
            # when the speech was only confirmed as it ended, ev.interrupt
            # False, and queues the next one; we have no response queue, so
            # the newer turn always wins.)
            await self._cancel(turn)
            if turn.committed and turn.state.reply:
                turn.stored_reply = self._commit_turn(self._sid(), turn.state, turn.audio, turn.image)
                self._last_reply = turn
            if turn.committed:
                await self._send("cancelled", {"reason": "barge_in"})
        self._turn = _Turn(start_sample=ev.start_sample)
        await self._send("speech_started", {"audio_start_ms": round(_ms(ev.start_sample))})

    async def _on_speech_stopped(self, ev: SpeechStopped) -> None:
        turn = self._turn
        if turn is None or turn.committed:
            return  # no open turn for this segment (cannot happen: a start always precedes)
        audio = ev.audio if turn.audio is None else np.concatenate([turn.audio, ev.audio])
        # Gemma takes at most max_audio_seconds per clip; keep the latest part.
        max_samples = int(self.settings.max_audio_seconds * SAMPLE_RATE)
        turn.audio = audio[-max_samples:]
        turn.end_sample = ev.end_sample
        turn.image = self._image or turn.image
        self._image = None
        turn.state = TurnState()
        turn.task = asyncio.create_task(self._respond(turn, ev, time.monotonic()))

    def _grace(self, complete: bool | None) -> tuple[float, float]:
        """(reopen grace, processing delay) in seconds for a soft end."""
        o = self.opts
        if complete is None or complete:  # Smart Turn off/failed, or turn sounds complete
            return o["reopen_ms"] / 1000.0, 0.0
        return o["smart_turn_max_wait_ms"] / 1000.0, min(o["smart_turn_incomplete_delay_ms"], o["smart_turn_max_wait_ms"]) / 1000.0

    async def _respond(self, turn: _Turn, ev: SpeechStopped, soft_end: float) -> None:
        """Prepare the reply to a soft-ended turn; release it once the grace has passed."""
        queue: asyncio.Queue = asyncio.Queue()
        pump: asyncio.Task | None = None
        try:
            timer = TurnTimer(self.app_state.backend.name, audio_seconds=len(turn.audio) / SAMPLE_RATE)
            # Speech end -> segment close (the VAD's own silence wait).
            timer.add_span("endpoint", _ms(ev.end_sample - ev.last_speech_sample))

            complete: bool | None = None
            smart_turn = self.smart_turn
            if smart_turn is not None:
                t0 = time.perf_counter()
                try:
                    prob = await asyncio.to_thread(smart_turn.predict, turn.audio)
                    complete = prob > self.opts["smart_turn_threshold"]
                    timer.add_span("smart_turn", (time.perf_counter() - t0) * 1000.0,
                                   probability=round(prob, 3), complete=complete)
                except Exception:  # noqa: BLE001
                    log.exception("Smart Turn inference failed; using the default reopen grace")
            grace_s, delay_s = self._grace(complete)

            if not self.respond:
                # Turn detection only (e.g. hands-free recording of read-aloud
                # clips): report where the turn ended and say nothing.
                await self._wait_for_release(soft_end + grace_s)
                turn.committed = turn.finished = True
                await self._send("speech_stopped", {"audio_start_ms": round(_ms(turn.start_sample)),
                                                   "audio_end_ms": round(_ms(turn.end_sample))})
                return

            if delay_s:
                await asyncio.sleep(delay_s)

            async def _pump() -> None:
                try:
                    async for item in self._turn_events(
                        self._sid(), turn.audio, self.instruction, timer,
                        transcribe=self.transcribe, asr_engine=self.asr_engine, llm_input=self.llm_input,
                        speak=self.speak, engine=self.engine,
                        image_bytes=turn.image, state=turn.state, commit=False,
                    ):
                        queue.put_nowait(item)
                except Exception as exc:  # noqa: BLE001
                    log.exception("turn failed")
                    queue.put_nowait(("error", {"message": str(exc)}))
                queue.put_nowait(None)

            pump = asyncio.create_task(_pump())

            await self._wait_for_release(soft_end + grace_s)
            turn.committed = True
            held_ms = (time.monotonic() - soft_end) * 1000.0
            await self._send("speech_stopped", {"audio_start_ms": round(_ms(turn.start_sample)),
                                               "audio_end_ms": round(_ms(turn.end_sample))})

            first_audio_ms: float | None = None
            while (item := await queue.get()) is not None:
                event, data = item
                if event == "audio":
                    turn.sentences[data.get("index", len(turn.sentences))] = data.get("sentence", "")
                    if first_audio_ms is None:
                        first_audio_ms = (time.perf_counter() - timer._t_start) * 1000.0
                elif event == "done":
                    # The reply was prepared during the grace; report when it
                    # actually reached the client.
                    data["metrics"]["components"].append({"name": "reopen_grace", "ms": round(held_ms, 2)})
                    if first_audio_ms is not None:
                        data["metrics"]["time_to_first_audio_ms"] = first_audio_ms
                    turn.stored_reply = self._commit_turn(self._sid(), turn.state, turn.audio, turn.image)
                    self._last_reply = turn
                    turn.finished = True
                await self._send(event, data)
            turn.finished = True
        finally:
            if pump is not None and not pump.done():
                pump.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await pump

    async def _wait_for_release(self, deadline: float) -> None:
        """Block until the reopen grace is over and no tentative speech is pending."""
        pending_since: float | None = None
        while True:
            now = time.monotonic()
            if self.segmenter.pending_reopen:
                pending_since = pending_since or now
                if now - pending_since < _PENDING_REOPEN_TIMEOUT_S:
                    await asyncio.sleep(0.02)
                    continue
            else:
                pending_since = None
            if now >= deadline:
                return
            await asyncio.sleep(min(0.02, deadline - now))

    async def _cancel(self, turn: _Turn) -> None:
        task = turn.task
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    # ---- helpers -------------------------------------------------------------
    def _sid(self) -> str:
        if self.session_id is None:
            self.session_id = uuid.uuid4().hex
        return self.session_id

    async def _send(self, event: str, data: dict) -> None:
        try:
            async with self._send_lock:
                await self.ws.send_text(json.dumps({"type": event, **data}))
        except (WebSocketDisconnect, RuntimeError):
            pass  # client went away; the receive loop will notice and clean up
