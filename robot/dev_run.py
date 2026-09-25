#!/usr/bin/env python3
"""
Laptop dev / simulation entrypoint for the G1 control centre.

Runs the REAL presence pipeline (webcam + the live /converse gateway + the web
control centre) with the robot-only actuators stubbed:
  - speaker  -> LocalAudio: plays 16 kHz PCM through the laptop (winsound on
                Windows, sounddevice elsewhere)
  - LED/arm  -> no-ops (state shown in the UI)
  - no unitree_sdk2py, no DDS, no controller keys

Drive it entirely from the web UI: Arm/Disarm + Hold-to-Talk (browser mic).
Native mic capture is disabled on this machine (PortAudio won't load on
win-arm64), so the browser "Talk" button is the mic path; the greeting still
fires on arrival and plays locally.

Run:
    uv venv && uv pip install -r requirements-dev.txt
    $env:PRESENCE_WEB_PASSWORD="..."; uv run python dev_run.py
    # open http://localhost:8080
"""
import os
import io
import time
import wave
import threading

import numpy as np

# Run from this file's directory so relative paths (prompts.json) resolve
# regardless of how we're launched (e.g. via the preview tool from the repo root).
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Defaults for laptop use (override via env before launch).
os.environ.setdefault("PRESENCE_WEB", "1")
os.environ.setdefault("PRESENCE_WEB_PASSWORD", "")   # open by default in dev; set to require a password
os.environ.setdefault("CAM_BACKEND", "auto")     # integer-index capture, not V4L2
os.environ.setdefault("CAM_INDEX", "0")
os.environ.setdefault("GEMMA_URL", "http://localhost:2730")

import presence
import control_center

TTS_RATE = 16000
DEV_GAIN = float(os.environ.get("PRESENCE_DEV_GAIN", "1.0"))   # 1.0 avoids clipping


def _wav_to_pcm16k(wav_path, gain):
    """Decode a WAV file -> (16 kHz mono s16le bytes, duration_s). Mirrors the
    robot's helper so converse_mode's playback worker is happy. Handles kokoro
    24 kHz -> 16 kHz."""
    w = wave.open(wav_path, "rb")
    sr, ch = w.getframerate(), w.getnchannels()
    a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    w.close()
    if ch == 2:
        a = a.reshape(-1, 2).mean(axis=1).astype(np.int16)
    if sr != TTS_RATE:
        n = int(len(a) * TTS_RATE / sr)
        a = np.interp(np.linspace(0, len(a), n, endpoint=False),
                      np.arange(len(a)), a).astype(np.int16)
    a = np.clip(a.astype(np.float32) * gain, -32768, 32767).astype(np.int16)
    return a.tobytes(), len(a) / TTS_RATE


def _pcm_to_wav(pcm, rate=TTS_RATE):
    bio = io.BytesIO()
    w = wave.open(bio, "wb")
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
    w.writeframes(pcm); w.close()
    return bio.getvalue()


class LocalAudio:
    """Stub for the G1 AudioClient: plays 16 kHz PCM on the laptop speakers."""

    def __init__(self):
        self.backend = None
        self._winsound = None
        self._sd = None
        try:
            import winsound
            self._winsound = winsound
            self.backend = "winsound"
        except Exception:
            try:
                import sounddevice as sd
                self._sd = sd
                self.backend = "sounddevice"
            except Exception:
                pass
        print(f"[dev] LocalAudio backend: {self.backend or 'none (silent)'}")

    def PlayStream(self, name, seq, pcm):
        try:
            if self.backend == "winsound":
                self._winsound.PlaySound(
                    _pcm_to_wav(pcm),
                    self._winsound.SND_MEMORY | self._winsound.SND_ASYNC)
            elif self.backend == "sounddevice":
                self._sd.play(np.frombuffer(pcm, dtype=np.int16), TTS_RATE)
        except Exception as e:
            print(f"[dev] play failed: {e}")

    # no-op robot controls
    def LedControl(self, r, g, b): pass
    def SetVolume(self, v): pass
    def SetTimeout(self, t): pass
    def Init(self): pass


def set_led(audio, r, g, b):
    """LED is surfaced in the UI via presence state, not a physical device."""
    pass


def main():
    audio = LocalAudio()
    busy = threading.Event()
    seq_ref = [0]
    controller = presence.PresenceController(
        audio, busy, set_led, _wav_to_pcm16k, DEV_GAIN, seq_ref,
        arm=None, mode_fn=lambda: 5)
    controller.start()
    control_center.start_control_center(controller.detector, controller, busy)
    port = os.environ.get("PRESENCE_WEB_PORT", "8080")
    pw = os.environ.get("PRESENCE_WEB_PASSWORD", "")
    print(f"[dev] open http://localhost:{port}  (password: {pw})")
    print("[dev] Arm + Hold-to-Talk from the page; native mic is disabled, "
          "browser mic is the input.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[dev] shutting down")
        controller.stop()


if __name__ == "__main__":
    main()
