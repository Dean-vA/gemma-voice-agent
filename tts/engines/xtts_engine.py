"""XTTS-v2 (Coqui): expressive + voice cloning. Heavier / higher latency."""
from __future__ import annotations

import os

import numpy as np

from tts.engines.base import TTSEngine


class XTTSEngine(TTSEngine):
    name = "xtts"
    sample_rate = 24000

    def __init__(self, language: str = "en", speaker_wav: str | None = None) -> None:
        self.language = language
        # Reference clip for cloning; if absent, fall back to a built-in speaker.
        self.speaker_wav = speaker_wav or os.environ.get("XTTS_SPEAKER_WAV")
        self.default_speaker = os.environ.get("XTTS_SPEAKER", "Claribel Dervla")
        self._tts = None

    def load(self) -> None:
        import torch
        from TTS.api import TTS

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    def synthesize(self, text: str, voice: str | None = None) -> tuple[np.ndarray, int]:
        kwargs = {"text": text, "language": self.language}
        if self.speaker_wav:
            kwargs["speaker_wav"] = self.speaker_wav
        else:
            kwargs["speaker"] = voice or self.default_speaker
        wav = self._tts.tts(**kwargs)
        samples = np.asarray(wav, dtype=np.float32)
        return samples, self.sample_rate
