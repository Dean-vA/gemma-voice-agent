"""Streamed (WebSocket) conversation mode for the G1 — experimental.

The default presence loop records one utterance with the on-robot energy VAD,
POSTs it to /converse and blocks until the reply has played. This module is the
alternative: the mic is streamed continuously to the gateway's /ws/converse,
which does the turn-taking itself (Silero VAD + Smart Turn) and can be
interrupted mid-reply (barge-in). It is the same division of labour as the
Reachy Mini conversation app.

Because the mic stays open while George talks, his own voice has to be removed
from it first: see aec.py (the echo canceller Reachy's SDK uses in software).

This file deliberately imports nothing robot-specific (no DDS, camera or
Flask): everything it needs is passed in, so it can be exercised off-robot.
"""
from __future__ import annotations

import base64
import json
import os
import queue
import socket
import ssl
import struct
import threading
import time
from urllib.parse import urlparse

import numpy as np

import aec as aec_mod

RATE = 16000


# ============================ WebSocket client ================================
class WebSocketClient:
    """Minimal RFC 6455 client on the standard library (the robot image has no
    WebSocket package). Text and binary messages, ping/pong, fragmentation."""

    def __init__(self, url: str, timeout: float = 10.0) -> None:
        u = urlparse(url)
        secure = u.scheme == "wss"
        port = u.port or (443 if secure else 80)
        sock = socket.create_connection((u.hostname, port), timeout=timeout)
        if secure:
            sock = ssl.create_default_context().wrap_socket(sock, server_hostname=u.hostname)
        key = base64.b64encode(os.urandom(16)).decode()
        path = (u.path or "/") + (f"?{u.query}" if u.query else "")
        sock.sendall((f"GET {path} HTTP/1.1\r\nHost: {u.hostname}:{port}\r\nUpgrade: websocket\r\n"
                      f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
        head = b""
        while b"\r\n\r\n" not in head:
            chunk = sock.recv(4096)
            if not chunk:
                raise ConnectionError("gateway closed the connection during the WebSocket handshake")
            head += chunk
        status, _, rest = head.partition(b"\r\n\r\n")
        first_line = status.split(b"\r\n", 1)[0]
        if b" 101" not in first_line:
            raise ConnectionError("WebSocket upgrade refused: " + first_line.decode(errors="replace"))
        self._sock = sock
        self._buf = rest
        self._send_lock = threading.Lock()
        self.closed = False
        sock.settimeout(None)

    def _send(self, opcode: int, payload: bytes) -> None:
        n = len(payload)
        head = bytes([0x80 | opcode])
        if n < 126:
            head += bytes([0x80 | n])
        elif n < 65536:
            head += bytes([0x80 | 126]) + struct.pack(">H", n)
        else:
            head += bytes([0x80 | 127]) + struct.pack(">Q", n)
        mask = os.urandom(4)
        masked = (np.frombuffer(payload, dtype=np.uint8) ^ np.resize(np.frombuffer(mask, dtype=np.uint8), n)).tobytes() if n else b""
        with self._send_lock:
            self._sock.sendall(head + mask + masked)

    def send_json(self, obj: dict) -> None:
        self._send(0x1, json.dumps(obj).encode())

    def send_bytes(self, data: bytes) -> None:
        self._send(0x2, data)

    def _read(self, n: int) -> bytes:
        while len(self._buf) < n:
            chunk = self._sock.recv(65536)
            if not chunk:
                raise ConnectionError("WebSocket closed")
            self._buf += chunk
        out, self._buf = self._buf[:n], self._buf[n:]
        return out

    def recv(self):
        """Next message as str (text) or bytes (binary); None once closed."""
        message, kind = b"", None
        while True:
            b0, b1 = self._read(2)
            opcode, fin, n = b0 & 0x0F, b0 & 0x80, b1 & 0x7F
            if n == 126:
                n = struct.unpack(">H", self._read(2))[0]
            elif n == 127:
                n = struct.unpack(">Q", self._read(8))[0]
            mask = self._read(4) if b1 & 0x80 else None
            payload = self._read(n)
            if mask:
                payload = (np.frombuffer(payload, dtype=np.uint8) ^ np.resize(np.frombuffer(mask, dtype=np.uint8), n)).tobytes()
            if opcode == 0x8:
                self.closed = True
                return None
            if opcode == 0x9:
                self._send(0xA, payload)
                continue
            if opcode == 0xA:
                continue
            if opcode in (0x1, 0x2):
                kind = opcode
            message += payload
            if fin:
                return message.decode() if kind == 0x1 else message

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            try:
                self._send(0x8, b"")
            except OSError:
                pass
        try:
            self._sock.close()
        except OSError:
            pass


# ============================== sample clock ==================================
class SampleClock:
    """Where the microphone sample counter is *right now*.

    Mic audio arrives in bursts (the G1 array sends 160 ms packets), so "samples
    read so far" only updates in steps. This ties the sample counter to the wall
    clock, anchored on the earliest-arriving audio, so that an event between two
    reads (a clip being sent to the speaker) can be placed to within a few ms."""

    def __init__(self) -> None:
        self._offset = None      # samples - time*RATE, for the timeliest frame seen
        self.samples = 0

    def advance(self, n: int) -> None:
        self.samples += n
        off = self.samples - time.monotonic() * RATE
        # Late frames have a smaller offset; follow the largest, but let it sag
        # slowly so a drifting mic clock is tracked.
        self._offset = off if self._offset is None else max(off, self._offset - 0.5)

    def now(self) -> int:
        if self._offset is None:
            return self.samples
        return int(time.monotonic() * RATE + self._offset)


# ================================ playback ====================================
class Player:
    """Plays reply sentences one after another on the robot speaker, and can be
    cut off mid-sentence (barge-in). ``to_pcm`` turns a WAV payload into
    (16 kHz mono int16 bytes, seconds); ``play`` hands PCM to the speaker and
    returns at once; ``stop`` silences it."""

    def __init__(self, to_pcm, play, stop, on_clip=None, on_active=None, on_cut=None,
                 drain_pad: float = 0.25) -> None:
        self._to_pcm, self._play, self._stop = to_pcm, play, stop
        self._on_clip, self._on_active, self._on_cut = on_clip, on_active, on_cut
        self._drain_pad = drain_pad
        self._q: queue.Queue = queue.Queue()
        self._cut = threading.Event()
        self._closed = False
        self.active = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def put(self, wav_bytes: bytes, index=None) -> None:
        """Queue one reply sentence; ``index`` is the gateway's sentence index,
        reported back through ``on_cut`` if it is cut off mid-playback."""
        self._q.put((wav_bytes, index))

    def flush(self) -> None:
        """Stop speaking now and forget everything queued."""
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        self._cut.set()
        try:
            self._stop()
        except Exception as e:
            print(f"[stream] stop playback failed: {e}")

    def close(self) -> None:
        self._closed = True
        self.flush()
        self._q.put(None)

    def _set_active(self, on: bool) -> None:
        if on != self.active:
            self.active = on
            if self._on_active:
                self._on_active(on)

    def _run(self) -> None:
        while not self._closed:
            try:
                item = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break
            item, index = item
            self._cut.clear()
            try:
                pcm, dur = self._to_pcm(item)
                self._set_active(True)
                if self._on_clip:
                    self._on_clip(pcm)                    # reference for the echo canceller, at send time
                t0 = time.time()
                self._play(pcm)                           # returns at once on the G1; tolerate a blocking one
                cut = self._cut.wait(max(0.0, dur + 0.05 - (time.time() - t0)))   # until played, or interrupted
                if cut and self._on_cut and index is not None and not self._closed:
                    try:                                   # tell the gateway how much was heard
                        self._on_cut(index, min(1.0, (time.time() - t0) / max(dur, 1e-3)))
                    except Exception as e:
                        print(f"[stream] cut report failed: {e}")
            except Exception as e:
                print(f"[stream] play failed: {e}")
                cut = True
            if self._q.empty():
                if not cut:
                    self._cut.wait(self._drain_pad)        # let the speaker tail decay
                if self._q.empty():
                    self._set_active(False)


# ============================== the session ===================================
class StreamSession:
    """One streamed conversation: from the greeting until the visitor leaves.

    opts()       -> dict of current options (re-read continuously, so the
                    control centre can change them live); see presence.STREAM
    mic          -> object with read() returning one 30 ms int16 frame @16 kHz
    play(pcm), stop_playback(), to_pcm(wav_bytes) -> (pcm, seconds)
    pub(kind, **data)   publishes state to the control centre (may be a no-op)
    grab_image() -> JPEG bytes or None, sent with each turn
    """

    def __init__(self, url, session_id, opts, mic, play, stop_playback, to_pcm, pub=None,
                 grab_image=None, set_led=None, drain_pad: float = 0.25, echo_delay_ms=None) -> None:
        self.url, self.session_id, self.opts = url, session_id, opts
        self.mic, self.pub = mic, (pub or (lambda *a, **k: None))
        self.grab_image, self.set_led = grab_image, (set_led or (lambda *a: None))
        # echo_delay_ms: the speaker delay measured in an earlier session, if any
        self.status = {"connected": False, "aec": "off", "echo_delay_ms": echo_delay_ms, "barge_in": False,
                       "gateway": None, "error": ""}
        self._clock = SampleClock()
        self._aec = None
        self._aec_key = None
        self._sent_cfg = None
        self._ws = None
        self._image = None
        self._turn_t0 = None
        self.last_activity = time.time()   # last speech heard / reply playing (presence keep-alive)
        self._reply = []
        self._heard = ""
        self._marks = set()
        self._raw = np.zeros(0, dtype=np.int16)     # recent raw mic audio, for delay estimation
        self._raw_start = 0
        self._pending_clips = []                    # (mic sample at feed, reference pcm)
        self._delays = []
        self._lock = threading.Lock()
        self.player = Player(to_pcm, play, stop_playback, on_clip=self._on_clip,
                             on_active=self._on_playback, on_cut=self._on_cut, drain_pad=drain_pad)

    # ---- configuration -------------------------------------------------------
    def _barge_in(self, o: dict) -> bool:
        """Barge-in is only safe once the robot voice is out of the mic.

        With the modern canceller that is as soon as it is running. The legacy
        one needs the speaker delay first (measured from the first reply, or set
        by hand), and we only switch over between replies, never mid-sentence,
        because clips already queued were placed with the old figure."""
        if not o.get("barge_in", True):
            return False
        if self._aec is None:
            # Echo cancellation switched off on purpose: the operator's call.
            # Asked for but unavailable: stay half-duplex rather than have the
            # robot interrupt itself.
            return not o.get("aec")
        if not self._aec.legacy:
            return True                      # AEC3
        if self.status["barge_in"]:
            return True                      # already on: stay on
        known = o.get("aec_delay_ms") not in (None, "", "auto") or self.status["echo_delay_ms"] is not None
        return known and not self.player.active

    def _config_message(self, o: dict) -> dict:
        self.status["barge_in"] = self._barge_in(o)
        return {"type": "config", "session_id": self.session_id, "instruction": o.get("instruction", ""),
                "transcribe": bool(o.get("transcribe")), "asr_engine": o.get("asr_engine", ""),
                "llm_input": o.get("llm_input", ""), "speak": True, "engine": o.get("tts_engine", ""),
                "vad": {**o.get("vad", {}), "vad_barge_in": self.status["barge_in"]}}

    def _sync_options(self, o: dict) -> None:
        """(Re)build the echo canceller and push option changes to the gateway."""
        key = (bool(o.get("aec")), bool(o.get("aec_noise_suppression", True)), bool(o.get("aec_gain_control", True)))
        if key != self._aec_key:
            self._aec_key = key
            if self._aec is not None:
                self._aec.close()
                self._aec = None
            if key[0]:
                self._aec, why = aec_mod.create(noise_suppression=key[1], gain_control=key[2])
                if self._aec is not None:
                    self.status["aec"] = f"on ({self._aec.where}, {'legacy canceller' if self._aec.legacy else 'AEC3'})"
                else:
                    self.status["aec"] = "unavailable: " + why
                print(f"[stream] echo cancellation {self.status['aec']}")
            else:
                self.status["aec"] = "off"
        if self._aec is not None:
            self._aec.delay_ms = self._aec_delay(o)
        msg = self._config_message(o)
        if msg != self._sent_cfg:
            self._ws.send_json(msg)
            self._sent_cfg = msg

    def _aec_delay(self, o: dict) -> float:
        """Where to place the reference, in ms after a clip is sent to the speaker.

        A figure set by hand wins. Otherwise: AEC3 finds the delay itself and
        only needs the reference early, so 0; the legacy canceller needs it
        DELAY_MARGIN_MS ahead of the measured speaker delay."""
        fixed = o.get("aec_delay_ms")
        if fixed not in (None, "", "auto"):
            return max(0.0, float(fixed))
        if not self._aec.legacy:
            return 0.0
        measured = self.status["echo_delay_ms"]
        return max(0.0, measured - aec_mod.DELAY_MARGIN_MS) if measured is not None else 60.0

    # ---- playback hooks ------------------------------------------------------
    def _on_clip(self, pcm: bytes) -> None:
        now = self._clock.now()
        if self._aec is not None:
            self._aec.feed_far(pcm, at_sample=now)
        with self._lock:
            self._pending_clips.append((now, np.frombuffer(pcm, dtype=np.int16)))

    def _on_cut(self, index: int, fraction: float) -> None:
        """A reply sentence was cut off (barge-in): tell the gateway how much of it
        was heard, so its history keeps only what George actually said."""
        try:
            if self._ws is not None:
                self._ws.send_json({"type": "played", "state": "cut", "index": index,
                                    "fraction": round(fraction, 3)})
        except OSError:
            pass
        print(f"[stream] reply cut at sentence {index} ({fraction:.0%} played)")

    def _on_playback(self, active: bool) -> None:
        self.pub("speaking", on=active)
        try:
            if self._ws is not None:
                self._ws.send_json({"type": "playback", "active": active})
        except OSError:
            pass
        if active:
            self.set_led(0, 80, 160)        # blue: speaking
        else:
            self.set_led(0, 200, 0)         # green: listening

    def _interrupt(self) -> None:
        self.player.flush()
        if self._aec is not None:
            self._aec.flush_far()

    # ---- speaker delay estimate ---------------------------------------------
    def _track_delay(self, frame: np.ndarray) -> None:
        """Keep a few seconds of raw mic audio and, once a clip George played
        should be audible in it, measure how late it arrived."""
        keep = RATE * 6
        with self._lock:
            self._raw = np.concatenate([self._raw, frame])
            if len(self._raw) > keep:
                self._raw_start += len(self._raw) - keep
                self._raw = self._raw[-keep:]
            if not self._pending_clips:
                return
            start, ref = self._pending_clips[0]
            ref = ref[: int(RATE * 1.5)]
            need = start + len(ref) + int(RATE * aec_mod.MAX_DELAY_MS / 1000)
            if self._clock.samples < need:
                return
            self._pending_clips.pop(0)
            lo = start - self._raw_start
            window = self._raw[lo: need - self._raw_start].copy() if lo >= 0 else None
        if window is None or len(ref) < RATE // 2:
            return
        delay, confidence = aec_mod.estimate_delay_ms(window, ref)
        if confidence >= aec_mod.MIN_DELAY_CONFIDENCE:
            self._delays = (self._delays + [delay])[-5:]
            self.status["echo_delay_ms"] = round(float(np.median(self._delays)), 1)
            self.status["echo_delay_confidence"] = round(confidence, 1)

    # ---- gateway events ------------------------------------------------------
    def _reader(self) -> None:
        try:
            while True:
                raw = self._ws.recv()
                if raw is None:
                    break
                if isinstance(raw, bytes):
                    continue
                self._on_event(json.loads(raw))
        except (ConnectionError, OSError) as e:
            if not self._ws.closed:
                self.status["error"] = str(e)
        finally:
            self.status["connected"] = False

    def _on_event(self, ev: dict) -> None:
        kind = ev.get("type")
        if kind in ("speech_started", "speech_stopped", "token", "audio"):
            self.last_activity = time.time()
        if kind == "speech_started":
            if self.player.active:
                print("[stream] barge-in: visitor spoke over the reply")
                self._interrupt()
            self.pub("vad", phase="recording", voiced=True, active=True)
            image = self.grab_image() if self.grab_image else None
            self._image = image
            if image:
                self._ws.send_json({"type": "image", "data": base64.b64encode(image).decode()})
        elif kind == "speech_stopped":
            self._turn_t0 = time.perf_counter()
            self._reply, self._heard, self._marks = [], "", set()
            secs = max(0.0, (ev.get("audio_end_ms", 0) - ev.get("audio_start_ms", 0)) / 1000.0)
            self.pub("vad", phase="listening", voiced=False, active=True)
            self.pub("turn", ev="start", phase="converse", session=self.session_id,
                     image_jpeg=self._image, audio_bytes=int(secs * RATE * 2),
                     audio_secs=round(secs, 2))
            self.set_led(200, 120, 0)       # amber: thinking
        elif kind == "transcript":
            self._heard = ev.get("text", "")
            self._mark("first_token")
            self.pub("turn", ev="heard", heard=self._heard)
            print(f"  heard: {self._heard!r}")
        elif kind == "token":
            self._mark("first_token")
            self._reply.append(ev.get("text", ""))
            self.pub("turn", ev="reply", reply="".join(self._reply))
        elif kind == "audio":
            self._mark("first_audio")
            try:
                self.player.put(base64.b64decode(ev["wav_base64"]), ev.get("index"))
            except Exception as e:
                print(f"[stream] audio decode failed: {e}")
        elif kind == "done":
            metrics = ev.get("metrics") or {}
            self.pub("turn", ev="metrics", metrics=metrics)
            total = (time.perf_counter() - self._turn_t0) * 1000 if self._turn_t0 else None
            reply = "".join(self._reply) or ev.get("reply", "")
            self.pub("turn", ev="done", metrics=metrics, heard=self._heard, reply=reply, total_ms=total)
            print(f"[stream] reply: {reply!r}")
        elif kind == "cancelled":
            self._interrupt()
            if self._turn_t0 is not None:
                self.pub("turn", ev="done", metrics={}, heard=self._heard,
                         reply="".join(self._reply) + " —", total_ms=(time.perf_counter() - self._turn_t0) * 1000)
        elif kind == "vad_config":
            self.status["gateway"] = {k: v for k, v in ev.items() if k != "type"}
        elif kind == "error":
            self.status["error"] = ev.get("message", "")
            print(f"[stream] gateway error: {self.status['error']}")

    def _mark(self, name: str) -> None:
        if name not in self._marks and self._turn_t0 is not None:
            self._marks.add(name)
            self.pub("turn", ev=name, ms=(time.perf_counter() - self._turn_t0) * 1000)

    # ---- main loop -----------------------------------------------------------
    def run(self, keep_going) -> str:
        """Stream until ``keep_going()`` returns a reason to stop (a non-empty
        string such as "left" or "stopped"), the mic stalls, or the link drops.
        Returns that reason ("error" for a failure)."""
        try:
            self._ws = WebSocketClient(self.url)
        except (OSError, ConnectionError) as e:
            self.status["error"] = f"could not connect: {e}"
            print(f"[stream] {self.status['error']}")
            self.player.close()
            return "error"
        self.status["connected"] = True
        reader = threading.Thread(target=self._reader, daemon=True)
        reader.start()
        self.set_led(0, 200, 0)
        reason, n = "error", 0
        try:
            while True:
                if n % 10 == 0:                         # ~3x a second: options, and should we go on?
                    stop = keep_going()
                    if stop:
                        reason = stop
                        break
                    if not self.status["connected"]:
                        break
                    self._sync_options(self.opts())
                n += 1
                frame = self.mic.read()
                gain = float(self.opts().get("mic_gain", 1.0))
                if gain != 1.0:
                    frame = np.clip(frame.astype(np.float32) * gain, -32768, 32767).astype(np.int16)
                self._clock.advance(len(frame))
                self._track_delay(frame)
                if self._aec is not None:
                    frame = self._aec.process(frame)
                if n % 3 == 0:
                    f = frame.astype(np.float32) / 32768.0
                    self.pub("vad", level=round(float(np.sqrt(np.mean(f * f))), 4), active=True)
                self._ws.send_bytes(frame.astype("<i2").tobytes())
        except (ConnectionError, OSError) as e:
            self.status["error"] = str(e)
            print(f"[stream] link lost: {e}")
        except Exception as e:                          # e.g. presence.MicStall
            self.status["error"] = str(e)
            print(f"[stream] stopped: {e}")
        finally:
            self.player.close()
            if self._aec is not None:
                self._aec.close()
            self._ws.close()
            self.status["connected"] = False
            self.pub("vad", active=False, phase="idle")
            self.pub("speaking", on=False)
        return reason
