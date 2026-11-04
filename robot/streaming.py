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
import collections
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


# George's echo profile (post-AEC frame RMS while he speaks), shared by every
# streamed session in this process: a new visitor's first reply would otherwise
# start with no echo level learned and a barge-in threshold far too low
# (seen on the robot: 0.024 against George's echo peaks).
_ECHO_LVL: list = []

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
                 grab_image=None, set_led=None, drain_pad: float = 0.25, echo_delay_ms=None,
                 on_prompt=None) -> None:
        self.url, self.session_id, self.opts = url, session_id, opts
        self.on_prompt = on_prompt        # (prompt, reply) -> None, e.g. a log line
        self._play_t0 = None               # when the current reply started playing
        # near-field barge-in gate: background level while George is silent, and
        # the gate's open/hangover state while he speaks
        self._ambient = []                 # frame RMS, last ~10 s with George silent
        self._near_thresh = 0.0            # near-field level: background x mult (>= min)
        self._play_lvl = _ECHO_LVL         # post-AEC frame RMS while George speaks (echo residual), last ~10 s; kept across visitors
        self._play_thresh = 0.0            # gate threshold while George speaks
        self.last_near_voice = 0.0         # last frame loud enough to be the visitor up close
        self._lvl = []                     # frame RMS since the last level log
        self._lvl_t0 = time.time()
        self._loud_run = 0                 # consecutive frames above the gate
        self._gate_open_until = 0.0
        self._gate_peak = 0.0              # loudest frame since the gate opened (for the log)
        self._gate_logged = False
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
        # Stop talking -> first sound, timed the same way for every link: the sample
        # the server says the speech ended at, mapped back to when it was sent.
        self._sent_samples = 0
        self._sent_log = collections.deque(maxlen=4000)   # (samples sent, monotonic), ~2 min
        self._speech_end_t = None
        self._sounded = True
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
                "debug_prompt": bool(o.get("debug_prompt")),
                "vad": {**o.get("vad", {}), "vad_barge_in": self.status["barge_in"]}}

    def _sync_options(self, o: dict) -> None:
        """(Re)build the echo canceller and push option changes to the gateway."""
        self._sync_aec(o)
        self._push_config(o)

    def _push_config(self, o: dict) -> None:
        msg = self._config_message(o)
        if msg != self._sent_cfg:
            self._ws.send_json(msg)
            self._sent_cfg = msg

    def _sync_aec(self, o: dict) -> None:
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
        self._first_sound()

    # ---- stop talking -> first sound ----------------------------------------
    def _speech_ended(self, audio_end_ms=None) -> None:
        """The server says the visitor stopped speaking at `audio_end_ms` of the
        audio this link sent. Use when that audio was sent if the figure fits what
        was sent; otherwise (a server counting from elsewhere) now."""
        t = time.monotonic()
        if audio_end_ms is not None:
            end = int(float(audio_end_ms) * RATE / 1000)
            if 0 <= self._sent_samples - end <= RATE * 30:
                for sent, at in reversed(self._sent_log):
                    if sent <= end:
                        t = at
                        break
        self._speech_end_t, self._sounded = t, False

    def _first_sound(self) -> None:
        if self._sounded or self._speech_end_t is None:
            return
        self._sounded = True
        ms = (time.monotonic() - self._speech_end_t) * 1000
        self.pub("turn", ev="end_to_sound", ms=ms)
        print(f"[stream] stop talking -> first sound {ms:.0f} ms")

    # ---- the link -------------------------------------------------------------
    def _connect(self):
        return WebSocketClient(self.url)

    def _on_connected(self) -> None:
        pass

    def _before_close(self, reason: str) -> None:
        pass

    def _send_frame(self, frame: np.ndarray) -> None:
        self._ws.send_bytes(frame.astype("<i2").tobytes())

    def _sent(self, frame: np.ndarray) -> None:
        self._sent_samples += len(frame)
        self._sent_log.append((self._sent_samples, time.monotonic()))

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
        self._play_t0 = time.time() if active else None     # start of this reply's playback
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

    # ---- near-field barge-in gate -------------------------------------------
    def _note_level(self, frame: np.ndarray, o: dict) -> float:
        """Level bookkeeping for every frame (after echo cancellation): the room's
        background (median RMS of the last ~10 s while George is silent), the
        near-field threshold (background x mult, never below min), and
        ``last_near_voice`` - the last frame loud enough to be the visitor up
        close, which is what presence counts as "talking" (crowd speech isn't).
        Logs a level summary every 5 s."""
        f = frame.astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(f * f)))
        now = time.time()
        if self._play_t0 is None:
            self._ambient.append(rms)
            if len(self._ambient) > 333:
                del self._ambient[: len(self._ambient) - 333]
        else:
            self._play_lvl.append(rms)
            if len(self._play_lvl) > 333:
                del self._play_lvl[: len(self._play_lvl) - 333]
        background = float(np.median(self._ambient)) if len(self._ambient) >= 30 else 0.0
        self._near_thresh = max(float(o.get("barge_gate_min", 0.02)),
                                background * float(o.get("barge_gate_mult", 3.0)))
        # While George speaks the gate must also clear his own leftover echo: 1.5x
        # its recent 90th percentile (the visitor up close measured 0.3-0.4, far above).
        # until ~1 s of his echo has been heard, assume the measured level at x1.5 speech gain
        echo_p90 = (float(np.percentile(self._play_lvl, 90)) if len(self._play_lvl) >= 30
                    else float(o.get("barge_echo_default", 0.04)))
        self._play_thresh = max(self._near_thresh, echo_p90 * float(o.get("barge_echo_mult", 1.5)))
        if rms > self._near_thresh and self._play_t0 is None and len(self._ambient) >= 30:
            self.last_near_voice = now           # (while George speaks his echo could count)
        self._lvl.append(rms)
        if now - self._lvl_t0 >= 5.0 and self._lvl:
            lv = np.array(self._lvl)
            print(f"[stream] mic: level p50 {np.median(lv):.3f} p90 {np.percentile(lv, 90):.3f} "
                  f"max {lv.max():.3f} · background {background:.3f} · near-voice > {self._near_thresh:.3f} "
                  f"({int((lv > self._near_thresh).sum())} frames)"
                  + (f" · George speaking: echo p90 {echo_p90:.3f}, barge-in > {self._play_thresh:.3f}" if self._play_t0 else ""))
            self._lvl, self._lvl_t0 = [], now
        return rms

    def _near_field_gate(self, frame: np.ndarray, o: dict, rms: float) -> np.ndarray:
        """While George speaks, pass the mic only when it is clearly louder than
        the room: the visitor talking up close, not a crowd a few metres away
        (on the array the visitor is several times louder). Opens after 4 loud
        frames (120 ms) and stays open 0.5 s after the last one, so a sentence
        isn't chopped. When George is silent everything passes, and the frames
        update the background level."""
        now = time.time()
        if self._play_t0 is None:                          # George silent: everything passes
            self._loud_run, self._gate_open_until, self._gate_logged = 0, 0.0, False
            return frame
        background = float(np.median(self._ambient)) if len(self._ambient) >= 30 else 0.0
        thresh = self._play_thresh
        self.status["barge_gate"] = {"background": round(background, 4), "threshold": round(thresh, 4)}
        if rms > thresh:
            self._loud_run += 1
            self._gate_peak = max(self._gate_peak, rms) if self._loud_run > 1 else rms
            if self._loud_run >= 4:
                self._gate_open_until = now + 0.5
                if not self._gate_logged:
                    self._gate_logged = True
                    print(f"[stream] barge gate open: level {self._gate_peak:.3f} > {thresh:.3f} "
                          f"(background {background:.3f} x{float(o.get('barge_gate_mult', 3.0)):g})")
        else:
            self._loud_run = 0
        if now < self._gate_open_until:
            return frame
        self._gate_logged = False
        return np.zeros_like(frame)

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
            self._speech_ended(ev.get("audio_end_ms"))
            self._turn_t0 = time.perf_counter()
            self._reply, self._heard, self._marks = [], "", set()
            secs = max(0.0, (ev.get("audio_end_ms", 0) - ev.get("audio_start_ms", 0)) / 1000.0)
            self.pub("vad", phase="listening", voiced=False, active=True)
            self.pub("turn", ev="start", phase="converse", session=self.session_id, transport="ws",
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
            prompt = ev.get("prompt")
            self.pub("turn", ev="done", metrics=metrics, heard=self._heard, reply=reply, total_ms=total,
                     prompt=prompt)
            print(f"[stream] reply: {reply!r}")
            if self.on_prompt and prompt:
                try:
                    self.on_prompt(prompt, reply)
                except Exception as e:
                    print(f"[stream] prompt log failed: {e}")
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
            self._ws = self._connect()
        except (OSError, ConnectionError) as e:
            self.status["error"] = f"could not connect: {e}"
            print(f"[stream] {self.status['error']}")
            self.player.close()
            return "error"
        self.status["connected"] = self.status["ever_connected"] = True
        reader = threading.Thread(target=self._reader, daemon=True)
        reader.start()
        self._on_connected()
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
                    frame = self._aec.process(frame)      # keeps adapting during the hold-off
                o = self.opts()
                rms = self._note_level(frame, o)          # background + near-field voice, every frame
                hold = float(o.get("barge_holdoff_ms", 0) or 0)
                if hold and self._play_t0 is not None and (time.time() - self._play_t0) * 1000.0 < hold:
                    frame = np.zeros_like(frame)          # reply just started: no echo barge-in
                elif o.get("barge_gate"):
                    frame = self._near_field_gate(frame, o, rms)
                if n % 3 == 0:
                    f = frame.astype(np.float32) / 32768.0
                    self.pub("vad", level=round(float(np.sqrt(np.mean(f * f))), 4), active=True)
                self._send_frame(frame)
                self._sent(frame)
        except (ConnectionError, OSError) as e:
            self.status["error"] = str(e)
            print(f"[stream] link lost: {e}")
        except Exception as e:                          # e.g. presence.MicStall
            self.status["error"] = str(e)
            print(f"[stream] stopped: {e}")
        finally:
            try:
                self._before_close(reason)
            except Exception as e:                      # never let it keep the link open
                print(f"[stream] closing: {e}")
            self.player.close()
            if self._aec is not None:
                self._aec.close()
            self._ws.close()
            self.status["connected"] = False
            self.pub("vad", active=False, phase="idle")
            self.pub("speaking", on=False)
        return reason


# ===================== the Reachy backend (speech-to-speech) =================
def words_heard(sentences, total_samples, heard_samples) -> str:
    """The words of a reply that had been played when its sound stopped.
    ``sentences`` holds (text, sample its sound starts at); a sentence runs to the
    next one's start, and the cut one is counted off in proportion (as the Reachy
    head does, head.py)."""
    words = []
    for i, (text, start) in enumerate(sentences):
        end = sentences[i + 1][1] if i + 1 < len(sentences) else total_samples
        if heard_samples >= end:
            words += text.split()
            continue
        if heard_samples > start and end > start:
            w = text.split()
            words += w[: int(len(w) * (heard_samples - start) / (end - start))]
        break
    return " ".join(words)


class S2SSession(StreamSession):
    """The same streamed conversation, but with the Reachy Mini's backend: Hugging
    Face speech-to-speech as the Ruxithid project runs it (s2s/ in this repo), over
    its OpenAI-realtime WebSocket (/v1/realtime). It does its own turn-taking
    (Silero + Smart Turn), transcribes with Parakeet, thinks with Gemma 4 12B and
    speaks with Qwen3-TTS. Echo cancellation, the near-field gate and the barge-in
    hold-off stay here on the robot, exactly as for the gateway link.

    The whole visit runs over it: the greeting, the conversation and the goodbye,
    so George has one voice throughout. The server has no camera.

    greeting / goodbye()  text of the typed turns that open and close the visit
                          (goodbye() is asked at the end; None = no goodbye)
    on_goodbye()          called as the goodbye starts (the wave)
    gain()                software gain for George's voice
    """

    FIRST_CLIP_S = 0.6     # first sound of a reply goes out once this much has arrived
    CLIP_S = 4.0           # after that, a clip per sentence, or this much at most

    def __init__(self, url, session_id, opts, mic, play, stop_playback, gain=None, greeting="",
                 goodbye=None, on_goodbye=None, **kw) -> None:
        self._gain = gain or (lambda: 1.0)
        super().__init__(url, session_id, opts, mic, play, stop_playback, self._pcm_clip, **kw)
        self.greeting, self.goodbye, self.on_goodbye = greeting, goodbye, on_goodbye
        self.status["backend"] = "reachy (speech-to-speech)"
        self._sent_instructions = None
        self._replying = False
        self._cur = None                 # the reply being written: see _new_reply
        self._reply_text = []
        self._buf = bytearray()
        self._clips = {}                   # clip index -> (reply, first sample, samples)
        self._clip_n = 0
        self._deaf = False                 # goodbye: the visitor has gone, hear nothing
        self._phase = "converse"
        self._said = threading.Event()     # a typed turn's reply has finished
        self._stage = {}                   # when this turn's events arrived; see _stage_metrics

    # ---- stage timings -----------------------------------------------------------
    # The server sends no timings of its own, so they're read off when its events
    # arrive here, from the end of the visitor's speech (as the robot sent it):
    #   vad  speech end -> speech_stopped (its silence wait + the network)
    #   asr  -> the transcript (Parakeet runs while the visitor speaks)
    #   llm  -> response.created, which comes with the first sentence: Smart Turn's
    #        wait and Gemma's first sentence together, not separable from here
    #   tts  -> the first audio (Qwen3-TTS); buffer_ms is how long the robot then
    #        collects audio (FIRST_CLIP_S) before the first sound
    def _stage_metrics(self) -> dict:
        st, end = self._stage, self._speech_end_t
        stop, heard, created, audio = (st.get(k) for k in ("stop", "heard", "created", "audio"))
        ms = lambda a, b: round(max(0.0, b - a) * 1000, 1)
        comps = []
        if end is not None and stop is not None:
            comps.append({"name": "vad", "ms": ms(end, stop)})
        if stop is not None and heard is not None:
            comps.append({"name": "asr", "ms": ms(stop, heard), "engine": "parakeet"})
        if stop is not None and created is not None:
            comps.append({"name": "llm", "ms": ms(max(stop, heard or stop), created)})
        if created is not None and audio is not None:
            tts = {"name": "tts", "ms": ms(created, audio)}
            if st.get("sound") is not None:
                tts["buffer_ms"] = ms(audio, st["sound"])
            comps.append(tts)
        # a cascade: Gemma reads Parakeet's transcript
        return {"components": comps, "llm_input": "transcript"} if comps else {}

    def _first_sound(self) -> None:
        sounded = self._sounded
        super()._first_sound()
        if not sounded and self._stage.get("audio") is not None:
            self._stage["sound"] = time.monotonic()
            self.pub("turn", ev="metrics", metrics=self._stage_metrics())

    # ---- audio ---------------------------------------------------------------
    def _pcm_clip(self, pcm: bytes):
        g = float(self._gain() or 1.0)
        if g != 1.0:
            pcm = np.clip(np.frombuffer(pcm, dtype=np.int16).astype(np.float32) * g,
                          -32768, 32767).astype(np.int16).tobytes()
        return pcm, len(pcm) / 2 / RATE

    def _on_playback(self, active: bool) -> None:
        # As for the gateway, minus the "playback" message the backend doesn't know.
        self._play_t0 = time.time() if active else None
        self.pub("speaking", on=active)
        self.set_led(*((0, 80, 160) if active else (0, 200, 0)))   # blue speaking, green listening

    def _send_frame(self, frame: np.ndarray) -> None:
        # Half-duplex unless barge-in is safe: the server hears silence while George
        # replies, as the gateway does with vad_barge_in off.
        if self._deaf or (not self.status["barge_in"] and (self._replying or self.player.active)):
            frame = np.zeros_like(frame)
        self._ws.send_json({"type": "input_audio_buffer.append",
                            "audio": base64.b64encode(frame.astype("<i2").tobytes()).decode()})

    # ---- the link --------------------------------------------------------------
    def _session_update(self, instructions: str) -> dict:
        return {"type": "session.update", "session": {
            "type": "realtime", "instructions": instructions,
            # Interrupting is decided here (near-field gate, hold-off), not by the
            # server, which would cut George off at any sound.
            "audio": {"input": {"turn_detection": {"type": "server_vad", "interrupt_response": False}},
                      "output": {}}}}

    def _connect(self):
        """Connect and wait for the server to take the session. It turns a new link
        away while it is still letting go of the last one ("slots are in use")."""
        o = self.opts()
        for attempt in range(8):
            ws = WebSocketClient(self.url)
            ws.send_json(self._session_update(o.get("instruction", "")))
            deadline = time.time() + 15
            busy = False
            try:
                ws._sock.settimeout(1.0)
                while time.time() < deadline:
                    try:
                        raw = ws.recv()
                    except socket.timeout:
                        continue
                    if raw is None:
                        break
                    ev = json.loads(raw) if isinstance(raw, str) else {}
                    if ev.get("type") == "session.updated":
                        ws._sock.settimeout(None)
                        self._sent_instructions = o.get("instruction", "")
                        return ws
                    if ev.get("type") == "error":
                        msg = str((ev.get("error") or {}).get("message", ""))
                        busy = "slots are in use" in msg
                        if not busy:
                            print(f"[s2s] server error while connecting: {msg}")
            except (ConnectionError, OSError):
                pass
            ws.close()
            if not busy and attempt >= 2:
                break
            time.sleep(0.7)
        raise ConnectionError("the Reachy backend did not take the session")

    def _push_config(self, o: dict) -> None:
        self.status["barge_in"] = self._barge_in(o)
        instructions = o.get("instruction", "")
        if instructions != self._sent_instructions:     # persona edited live
            self._ws.send_json(self._session_update(instructions))
            self._sent_instructions = instructions

    def _say(self, text: str, phase: str) -> None:
        """A typed turn: something said to George without anyone speaking it."""
        self._phase = phase
        self._said.clear()
        self._ws.send_json({"type": "conversation.item.create", "item": {
            "type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}})
        self._ws.send_json({"type": "response.create"})

    def _on_connected(self) -> None:
        if self.greeting:
            self._say(self.greeting, "greet")

    def _before_close(self, reason: str) -> None:
        text = self.goodbye() if (reason == "left" and self.goodbye) else None
        if not text or not self.status.get("connected"):
            return
        self._deaf = True
        if self.on_goodbye:
            self.on_goodbye()
        self._say(text, "goodbye")
        self._said.wait(15)
        t0 = time.time()
        while self.player.active and time.time() - t0 < 15:
            time.sleep(0.1)

    # ---- replies ---------------------------------------------------------------
    def _new_reply(self, ev: dict) -> dict:
        return {"id": str((ev.get("response") or {}).get("id") or ev.get("event_id")), "item": None,
                "sentences": [], "samples": 0, "text": None, "finished": False, "cut": False,
                "cancelled": False}

    def _flush_clip(self) -> None:
        reply = self._cur
        if not self._buf or reply is None:
            return
        n = len(self._buf) // 2
        index, self._clip_n = self._clip_n, self._clip_n + 1
        self._clips[index] = (reply, reply["samples"], n)
        while len(self._clips) > 200:
            self._clips.pop(next(iter(self._clips)))
        reply["samples"] += n
        self.player.put(bytes(self._buf[: n * 2]), index)
        self._buf.clear()

    def _on_cut(self, index: int, fraction: float) -> None:
        clip = self._clips.get(index)
        if clip is None:
            return
        reply, start, n = clip
        self._report_cut(reply, start + fraction * n)

    def _report_cut(self, reply: dict, heard_samples: float) -> None:
        """Tell the server which words were heard, so George remembers only those
        (the Reachy backend's heard.py; the same as the gateway's cut report)."""
        if reply["cut"]:
            return
        reply["cut"] = True
        heard = words_heard(reply["sentences"], reply["samples"], heard_samples)
        full = reply["text"] or " ".join(t for t, _ in reply["sentences"])
        if " ".join(heard.split()) == " ".join(full.split()):
            return
        print(f"[s2s] cut off after {len(heard.split())} of {len(full.split())} words")
        self.pub("turn", ev="cut", reply=(heard + " —") if heard else "—")
        try:
            self._ws.send_json({"type": "conversation.item.truncate", "item_id": reply["item"] or "unknown",
                                "content_index": 0, "audio_end_ms": int(heard_samples * 1000 / RATE),
                                "x_heard_text": heard, "x_full_text": full,
                                "x_finished": reply["finished"]})
        except OSError:
            pass

    def _interrupt_reply(self) -> None:
        print("[stream] barge-in: visitor spoke over the reply")
        reply = self._cur
        if self._replying:
            self._ws.send_json({"type": "response.cancel"})
        if reply is not None:
            reply["cancelled"] = True
        self._buf.clear()
        self._interrupt()                 # stops the speaker; the clip playing reports its cut

    # ---- server events -----------------------------------------------------------
    def _on_event(self, ev: dict) -> None:
        kind = ev.get("type", "")
        if kind.startswith(("input_audio_buffer.speech", "response.output_audio")):
            self.last_activity = time.time()
        if kind == "input_audio_buffer.speech_started":
            if self.player.active or self._replying:
                self._interrupt_reply()
            self.pub("vad", phase="recording", voiced=True, active=True)
        elif kind == "input_audio_buffer.speech_stopped":
            self._speech_ended(ev.get("audio_end_ms"))
            self._turn_t0 = time.perf_counter()
            self._stage = {"stop": time.monotonic()}
            self._heard, self._marks = "", set()
            self.pub("vad", phase="listening", voiced=False, active=True)
            self.pub("turn", ev="start", phase="converse", session=self.session_id, transport="s2s")
            self.set_led(200, 120, 0)       # amber: thinking
        elif kind == "conversation.item.input_audio_transcription.completed":
            self._heard = ev.get("transcript", "")
            if "stop" in self._stage:
                self._stage.setdefault("heard", time.monotonic())
            self.pub("turn", ev="heard", heard=self._heard)
            print(f"  heard: {self._heard!r}")
        elif kind == "response.created":
            self._replying = True
            self._cur = self._new_reply(ev)
            self._reply_text, self._buf = [], bytearray()
            if self._phase != "converse" or self._turn_t0 is None:
                self._turn_t0 = time.perf_counter()     # a typed turn: no speech to time from
                self._marks, self._stage = set(), {}
            elif "stop" in self._stage:
                self._stage.setdefault("created", time.monotonic())
                self.pub("turn", ev="start", phase=self._phase, session=self.session_id, transport="s2s")
        elif kind == "response.output_audio_transcript.delta":
            # Each sentence's text arrives just before its sound.
            sentence = str(ev.get("delta", "")).strip()
            if self._cur is not None and not self._cur["cancelled"] and sentence:
                self._flush_clip()
                self._cur["sentences"].append((sentence, self._cur["samples"]))
                self._reply_text.append(sentence)
                self.pub("turn", ev="reply", reply=" ".join(self._reply_text))
        elif kind == "response.output_audio.delta":
            reply = self._cur
            if reply is None or reply["cancelled"]:
                return
            reply["item"] = reply["item"] or ev.get("item_id")
            self._mark("first_audio")
            if "created" in self._stage:
                self._stage.setdefault("audio", time.monotonic())
            self._buf += base64.b64decode(ev.get("delta", ""))
            limit = self.FIRST_CLIP_S if reply["samples"] == 0 else self.CLIP_S
            if len(self._buf) >= limit * RATE * 2:
                self._flush_clip()
        elif kind == "response.output_audio_transcript.done":
            if self._cur is not None:
                self._cur["text"] = ev.get("transcript", "")
        elif kind == "response.function_call_arguments.done":
            # The backend turns a mood tag opening a reply into a gesture call, and
            # answers nothing more until it is acknowledged.
            self._ws.send_json({"type": "conversation.item.create", "item": {
                "type": "function_call_output", "call_id": ev.get("call_id"), "output": "ok"}})
        elif kind == "response.done":
            reply, self._replying = self._cur, False
            status = (ev.get("response") or {}).get("status")
            if reply is not None:
                reply["finished"] = status == "completed"
                if not reply["cancelled"]:
                    self._flush_clip()
            text = (reply or {}).get("text") or " ".join(self._reply_text)
            cut = reply is not None and (reply["cancelled"] or status not in (None, "completed"))
            total = (time.perf_counter() - self._turn_t0) * 1000 if self._turn_t0 else None
            self.pub("turn", ev="done", metrics=self._stage_metrics(), heard=self._heard,
                     reply=(text + " —") if cut and text else text, total_ms=total)
            self._stage = {}
            print(f"[s2s] {self._phase}: {text!r}" + (" (cut)" if cut else ""))
            self._phase, self._turn_t0 = "converse", None
            self._said.set()
        elif kind == "error":
            msg = str((ev.get("error") or {}).get("message", ""))
            if "No function_call with call_id" not in msg:    # a cancelled reply's gesture: harmless
                self.status["error"] = msg
                print(f"[s2s] server error: {msg}")
