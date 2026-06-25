"""Kokoro-82M: the low-latency default (~45ms first chunk on a 5090, 24 kHz)."""
from __future__ import annotations

import numpy as np

from tts.engines.base import TTSEngine


class KokoroEngine(TTSEngine):
    name = "kokoro"
    sample_rate = 24000

    def __init__(self, default_voice: str = "af_heart", lang_code: str = "a") -> None:
        self.default_voice = default_voice
        self.lang_code = lang_code  # 'a' = American English
        self._pipe = None

    def load(self) -> None:
        from kokoro import KPipeline

        self._pipe = KPipeline(lang_code=self.lang_code)

    def synthesize(self, text: str, voice: str | None = None) -> tuple[np.ndarray, int]:
        voice = voice or self.default_voice
        chunks: list[np.ndarray] = []
        for _, _, audio in self._pipe(text, voice=voice):
            arr = audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio)
            chunks.append(arr.astype(np.float32))
        if not chunks:
            return np.zeros(1, dtype=np.float32), self.sample_rate
        return np.concatenate(chunks), self.sample_rate
