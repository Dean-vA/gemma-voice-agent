"""Dedicated speech recognition, as an alternative to Gemma's own transcription.

Parakeet TDT through ``nano-parakeet`` — the default STT of
huggingface/speech-to-speech, the Reachy Mini backend. It only produces the
*transcript* of a turn (what is shown, and kept as the text of past turns);
Gemma still answers from the raw audio.
"""
from __future__ import annotations

import asyncio
import logging

import numpy as np

from app.config import Settings

log = logging.getLogger(__name__)


class ParakeetASR:
    name = "parakeet"

    def __init__(self, model_name: str, device: str = "auto") -> None:
        import torch
        from nano_parakeet import from_pretrained

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            log.warning("CUDA requested for Parakeet but unavailable; using CPU")
            device = "cpu"
        self.model_name = model_name
        self.device = device
        self.model = from_pretrained(model_name=model_name, device=device)
        self.model.transcribe(np.zeros(16000, dtype=np.float32))  # warmup
        # One utterance at a time: the model is not safe to call concurrently.
        self._lock = asyncio.Lock()

    async def transcribe(self, samples: np.ndarray) -> str:
        """Transcribe 16 kHz mono float32 audio."""
        async with self._lock:
            return (await asyncio.to_thread(self.model.transcribe, samples)).strip()


def load_parakeet(settings: Settings) -> ParakeetASR | None:
    if not settings.parakeet:
        return None
    return ParakeetASR(settings.parakeet_model, settings.parakeet_device)
