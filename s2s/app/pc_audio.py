"""Let Ruxithid hear what this PC is playing, so players on Discord reach him directly.

His only ears are a microphone, but everything it picks up passes through the speech server in
this process. This captures sound playing on this PC and mixes it into that stream, so he hears
the players cleanly instead of through a speaker and across the room.

It listens to one app only (Discord), so music, videos and his own voice playing on this PC stay
out of his ears. It can listen to a whole output instead, which then includes all of those.

Wear headphones: if the same sound also comes out of speakers near him, he hears it twice.
While his mic is muted nothing is sent to the speech server, so nothing from this PC reaches him.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import ellip, sosfilt

RATE = 16000  # what the speech server works at
CONTROL_PORT = 8791  # the control page reads and sets the level here
SETTINGS = Path(__file__).parent / ".pc_audio"  # remembers the level and who he answers between runs
LOUD = 500  # average size of a 16-bit sample above which a stretch of sound counts as someone talking
UNITY = 50  # the level (0-100) at which sound is passed on unchanged; 100 doubles it, 0 is off
BACKLOG_SECONDS = 0.3  # most that may wait to be mixed in; more means he is hearing the past
CUSHION_SECONDS = 0.04  # held back before mixing starts, so uneven arrival doesn't leave holes
STALE_SECONDS = 0.5  # a gap this long in his mic stream (a mute) discards what was waiting
RECHECK_SECONDS = 10.0  # during silence, how often to look for a changed default output


class Downsampler:
    """Turn a stream of mono blocks at the output's rate into RATE, with no seams between blocks."""

    def __init__(self, rate: int) -> None:
        self.step = rate / RATE
        # Steep low-pass just under half of RATE, so nothing above it folds back into the speech band.
        self.sos = ellip(8, 0.1, 60, 0.45 * RATE, fs=rate, output="sos") if rate > RATE else None
        self.state = np.zeros((self.sos.shape[0], 2)) if self.sos is not None else None
        self.position = 0.0  # where the next output sample falls, counted in samples of the next block
        self.last = 0.0  # final sample of the previous block, to interpolate across the join

    def __call__(self, block: np.ndarray) -> np.ndarray:
        if self.sos is not None:
            block, self.state = sosfilt(self.sos, block, zi=self.state)
        count = len(block)
        positions = np.arange(self.position, count - 1 + 1e-9, self.step)
        self.position = (positions[-1] + self.step if len(positions) else self.position) - count
        joined = np.concatenate(([self.last], block))
        self.last = float(block[-1]) if count else self.last
        return np.interp(positions + 1, np.arange(count + 1), joined)


class PcAudio:
    def __init__(self, app: str | None = None, output: str | None = None) -> None:
        self.app = app  # listen to this app only, by the start of its program name ("Discord")
        self.wanted = output  # without an app: part of an output's name; None follows Windows' default
        self.level, self.dm_only = self._saved()  # dm_only: he answers only the DM (see dm_only.py)
        self._activity: deque[tuple[float, float, bool, bool]] = deque()  # when, how long, mic loud, this PC's sound loud
        self.device: str | None = None  # the app or output being listened to right now
        self.error: str | None = None
        self._lock = threading.Lock()
        self._waiting: deque[np.ndarray] = deque()
        self._waiting_samples = 0
        self._flowing = False  # the cushion has filled and sound is being mixed in
        self._last_capture = 0.0
        self._last_mix = 0.0
        self._peak = 0.0  # loudest recent sample as he would hear it, 0-1
        self._peak_at = 0.0

    # --- capture ---------------------------------------------------------------------------

    def _find_output(self, audio: Any) -> dict[str, Any]:
        if self.wanted is None:
            return audio.get_default_wasapi_loopback()
        for output in audio.get_loopback_device_info_generator():
            if self.wanted.lower() in output["name"].lower():
                return output
        raise LookupError(f"no output called {self.wanted!r}")

    def _find_app(self) -> Any:
        """The app's first process; Windows then includes the sound of everything it starts."""
        import psutil

        assert self.app is not None
        mine = [p for p in psutil.process_iter(["name", "create_time"]) if (p.info["name"] or "").lower().startswith(self.app.lower())]
        pids = {p.pid for p in mine}
        roots = [p for p in mine if p.ppid() not in pids]
        return min(roots, key=lambda p: p.info["create_time"], default=None)

    def _capture_app_forever(self) -> None:
        from proctap._native import ProcessLoopback

        while True:
            try:
                process = self._find_app()
            except OSError:  # a process vanished while being listed
                process = None
            if process is None:
                if self.device is not None:
                    print(f"pc-audio: {self.app} has closed; waiting for it to start again")
                self.device = None
                time.sleep(3)
                continue
            try:
                tap = ProcessLoopback(process.pid)
                form = tap.get_format()
                tap.start()
            except RuntimeError as exc:
                if str(exc) != self.error:
                    print(f"pc-audio: can't listen to {self.app} ({exc}); retrying")
                self.device, self.error = None, str(exc)
                time.sleep(3)
                continue
            floats = form["bits_per_sample"] == 32
            downsample = Downsampler(form["sample_rate"])
            print(f"pc-audio: listening to {self.app} only")
            self.device, self.error = self.app, None
            checked = time.monotonic()
            while True:
                try:
                    data = tap.read()
                except RuntimeError:
                    break
                if data:
                    block = np.frombuffer(data, np.float32 if floats else np.int16).reshape(-1, form["channels"]).mean(axis=1)
                    self._captured(downsample(block * 32768 if floats else block))
                else:
                    time.sleep(0.005)
                if time.monotonic() - checked > 2:
                    checked = time.monotonic()
                    if not process.is_running():
                        break
            try:
                tap.stop()
            except RuntimeError:
                pass

    def _capture_output_forever(self) -> None:
        import pyaudiowpatch

        audio = stream = None
        while True:
            # Windows only hands over a fresh list of outputs when the audio library restarts,
            # so a switch of headset is picked up by reopening, and only while nothing is playing.
            quiet = time.monotonic() - self._last_capture > RECHECK_SECONDS
            try:
                alive = stream is not None and stream.is_active()
            except OSError:
                alive = False
            if not alive or quiet:
                try:
                    if stream is not None:
                        stream.close()
                    if audio is not None:
                        audio.terminate()
                    audio = pyaudiowpatch.PyAudio()
                    output = self._find_output(audio)
                    channels, rate = int(output["maxInputChannels"]), int(output["defaultSampleRate"])
                    downsample = Downsampler(rate)

                    def heard(data: bytes, _frames: int, _time: Any, _status: int, channels: int = channels, downsample: Downsampler = downsample) -> tuple[None, int]:
                        self._captured(downsample(np.frombuffer(data, np.int16).reshape(-1, channels).mean(axis=1)))
                        return None, pyaudiowpatch.paContinue

                    stream = audio.open(
                        format=pyaudiowpatch.paInt16,
                        channels=channels,
                        rate=rate,
                        input=True,
                        input_device_index=output["index"],
                        frames_per_buffer=rate // 50,
                        stream_callback=heard,
                    )
                    name = output["name"].removesuffix(" [Loopback]")
                    if name != self.device:
                        print(f"pc-audio: listening to {name}")
                    self.device, self.error = name, None
                except (OSError, LookupError) as exc:
                    if str(exc) != self.error:
                        print(f"pc-audio: can't listen to this PC's sound ({exc}); retrying")
                    stream, self.device, self.error = None, None, str(exc)
                self._last_capture = time.monotonic()  # restart the quiet timer
            time.sleep(2)

    def _captured(self, samples: np.ndarray) -> None:
        now = time.monotonic()
        self._last_capture = now
        samples = samples * (self.level / UNITY)
        peak = float(np.abs(samples).max(initial=0.0)) / 32768
        if peak >= self._peak or now - self._peak_at > 0.5:
            self._peak, self._peak_at = min(peak, 1.0), now
        with self._lock:
            self._waiting.append(samples)
            self._waiting_samples += len(samples)
            if self._waiting_samples > BACKLOG_SECONDS * RATE:
                # The two audio clocks have drifted apart (or his stream stalled): skip ahead to now.
                while self._waiting_samples > 2 * CUSHION_SECONDS * RATE and len(self._waiting) > 1:
                    self._waiting_samples -= len(self._waiting.popleft())

    # --- mixing ----------------------------------------------------------------------------

    def mix(self, pcm: bytes) -> bytes:
        """Add this PC's sound to a stretch of his mic audio (16-bit mono at RATE)."""
        now = time.monotonic()
        gap, self._last_mix = now - self._last_mix, now
        mic = np.frombuffer(pcm, np.int16)
        with self._lock:
            if gap > STALE_SECONDS:
                self._waiting.clear()
                self._waiting_samples = 0
            if not self._waiting:
                self._flowing = False
            elif self._waiting_samples >= len(mic) + CUSHION_SECONDS * RATE:
                self._flowing = True  # enough for this stretch with the cushion left over
            taken, need = [], len(mic)
            while need and self._waiting and self._flowing:
                block = self._waiting.popleft()
                if len(block) > need:
                    self._waiting.appendleft(block[need:])
                    block = block[:need]
                taken.append(block)
                need -= len(block)
            self._waiting_samples -= len(mic) - need
        played = np.concatenate(taken) if taken else None
        self._activity.append((now, len(mic) / RATE, _is_loud(mic), played is not None and _is_loud(played)))
        while self._activity and now - self._activity[0][0] > 30:
            self._activity.popleft()
        if played is None or self.level == 0:
            return pcm
        mixed = mic.astype(np.float64)
        mixed[: len(played)] += played
        return np.clip(mixed, -32768, 32767).astype(np.int16).tobytes()

    def who_was_loud(self, seconds: float) -> tuple[float, float]:
        """How long his microphone, and this PC's sound, were each loud over the last while."""
        since = time.monotonic() - seconds
        recent = [entry for entry in list(self._activity) if entry[0] >= since]
        return sum(length for _, length, mic, _ in recent if mic), sum(length for _, length, _, pc in recent if pc)

    # --- level and status ------------------------------------------------------------------

    def _saved(self) -> tuple[int, bool]:
        try:
            saved = json.loads(SETTINGS.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return UNITY, False
        if isinstance(saved, int):  # the file used to hold the level alone
            saved = {"level": saved}
        return max(0, min(100, int(saved.get("level", UNITY)))), bool(saved.get("dm_only", False))

    def change(self, level: int | None = None, dm_only: bool | None = None) -> None:
        if level is not None:
            self.level = max(0, min(100, level))
        if dm_only is not None:
            self.dm_only = dm_only
        SETTINGS.write_text(json.dumps({"level": self.level, "dm_only": self.dm_only}), encoding="utf-8")

    def status(self) -> dict[str, Any]:
        now = time.monotonic()
        return {
            "level": self.level,
            "dm_only": self.dm_only,
            "app": self.app,  # set when he listens to one app only; device is None until it is running
            "device": self.device,
            "error": self.error,
            "sound": round(self._peak, 3) if now - self._peak_at < 2.5 else 0.0,  # how loud lately, as he would hear it
            "reaching": now - self._last_mix < 1.0,  # his mic stream is open, so it is being mixed in
        }

    def serve_controls(self) -> None:
        pc_audio = self

        class Controls(BaseHTTPRequestHandler):
            def reply(self, status: int, body: dict[str, Any]) -> None:
                payload = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self) -> None:
                self.reply(200, pc_audio.status()) if self.path == "/pc-audio" else self.reply(404, {})

            def do_POST(self) -> None:
                try:
                    body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                    level = int(body["level"]) if "level" in body else None
                    dm_only = bool(body["dm_only"]) if "dm_only" in body else None
                except (ValueError, KeyError, TypeError):
                    return self.reply(400, {"error": "expected {\"level\": 0-100} or {\"dm_only\": true or false}"})
                pc_audio.change(level, dm_only)
                self.reply(200, pc_audio.status())

            def log_message(self, *_: Any) -> None:
                pass

        ThreadingHTTPServer(("127.0.0.1", CONTROL_PORT), Controls).serve_forever()


def _is_loud(samples: np.ndarray) -> bool:
    return float(np.abs(samples).mean()) > LOUD


def install(app: str | None = None, output: str | None = None) -> PcAudio:
    """Start listening to this PC's sound and mix it into the audio the speech server takes in."""
    from speech_to_speech.api.openai_realtime.handlers import audio
    from speech_to_speech.api.openai_realtime.utils import resample

    pc_audio = PcAudio(app, output)
    capture = pc_audio._capture_app_forever if app else pc_audio._capture_output_forever
    threading.Thread(target=capture, name="pc-audio-capture", daemon=True).start()
    threading.Thread(target=pc_audio.serve_controls, name="pc-audio-controls", daemon=True).start()

    append_pcm = audio.AudioHandler.append_pcm

    def append_pcm_with_pc_audio(self: Any, conn_id: str, pcm_bytes: bytes, src_rate: int) -> list[bytes]:
        pcm_bytes = resample(pcm_bytes, src_rate, audio.PIPELINE_SAMPLE_RATE)
        return append_pcm(self, conn_id, pc_audio.mix(pcm_bytes), audio.PIPELINE_SAMPLE_RATE)

    audio.AudioHandler.append_pcm = append_pcm_with_pc_audio
    return pc_audio
