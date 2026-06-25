"""Piper: CPU-first ONNX TTS, edge/onboard-robot friendly."""
from __future__ import annotations

import os

import numpy as np

from tts.engines.base import TTSEngine


class PiperEngine(TTSEngine):
    name = "piper"

    def __init__(self, model_path: str | None = None) -> None:
        # Voice .onnx is downloaded into the image; override via PIPER_MODEL.
        self.model_path = model_path or os.environ.get(
            "PIPER_MODEL", "/voices/en_US-amy-medium.onnx"
        )
        self._voice = None

    def load(self) -> None:
        from piper import PiperVoice

        self._voice = PiperVoice.load(self.model_path)
        self.sample_rate = self._voice.config.sample_rate

    def synthesize(self, text: str, voice: str | None = None) -> tuple[np.ndarray, int]:
        pcm = bytearray()
        for chunk in self._voice.synthesize_stream_raw(text):
            pcm.extend(chunk)  # int16 little-endian
        samples = np.frombuffer(bytes(pcm), dtype=np.int16).astype(np.float32) / 32768.0
        if samples.size == 0:
            samples = np.zeros(1, dtype=np.float32)
        return samples, self.sample_rate
