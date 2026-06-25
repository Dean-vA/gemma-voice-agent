"""TTS engine interface. Each engine returns mono float32 PCM + sample rate."""
from __future__ import annotations

import abc
import io

import numpy as np
import soundfile as sf


class TTSEngine(abc.ABC):
    name: str = "base"
    sample_rate: int = 24000

    @abc.abstractmethod
    def load(self) -> None: ...

    @abc.abstractmethod
    def synthesize(self, text: str, voice: str | None = None) -> tuple[np.ndarray, int]:
        """Return (float32 mono samples in [-1, 1], sample_rate)."""

    def to_wav(self, samples: np.ndarray, sample_rate: int) -> bytes:
        buf = io.BytesIO()
        sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue()
