"""Chatterbox: expressive cloning on a gaming GPU; middle ground vs XTTS."""
from __future__ import annotations

import os

import numpy as np

from tts.engines.base import TTSEngine


class ChatterboxEngine(TTSEngine):
    name = "chatterbox"
    sample_rate = 24000

    def __init__(self, audio_prompt: str | None = None) -> None:
        # Optional reference clip for voice cloning.
        self.audio_prompt = audio_prompt or os.environ.get("CHATTERBOX_PROMPT_WAV")
        self._model = None

    def load(self) -> None:
        import torch
        from chatterbox.tts import ChatterboxTTS

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = ChatterboxTTS.from_pretrained(device=device)
        self.sample_rate = getattr(self._model, "sr", self.sample_rate)

    def synthesize(self, text: str, voice: str | None = None) -> tuple[np.ndarray, int]:
        kwargs = {}
        if self.audio_prompt:
            kwargs["audio_prompt_path"] = self.audio_prompt
        wav = self._model.generate(text, **kwargs)
        arr = wav.detach().cpu().numpy() if hasattr(wav, "detach") else np.asarray(wav)
        return arr.astype(np.float32).reshape(-1), self.sample_rate
