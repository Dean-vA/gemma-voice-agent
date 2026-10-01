"""Server-side voice activity + turn detection for the streamed voice loop.

Mirrors the approach the Reachy Mini conversation stack uses
(huggingface/speech-to-speech): Silero VAD finds speech/silence boundaries on
32 ms chunks, then Smart Turn judges whether the user actually finished their
turn. Both models run on CPU via onnxruntime, so they use no VRAM.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from app.config import Settings
from app.vad.segmenter import Segmenter, SpeechStarted, SpeechStopped, VADConfig

log = logging.getLogger(__name__)

__all__ = ["Segmenter", "SpeechStarted", "SpeechStopped", "VADConfig", "VADModels", "load_vad"]


@dataclass
class VADModels:
    """Process-wide model handles; per-connection state lives in SileroStream."""

    silero: object                 # app.vad.silero.SileroVAD
    smart_turn: object | None      # app.vad.smart_turn.SmartTurn, None if disabled/unavailable


def load_vad(settings: Settings) -> VADModels:
    from app.vad.silero import SileroVAD

    silero = SileroVAD()
    smart_turn = None
    if settings.smart_turn:
        try:
            from app.vad.smart_turn import SmartTurn

            smart_turn = SmartTurn()
        except Exception as exc:  # noqa: BLE001
            # Turn detection still works without it (fixed-delay endpointing).
            log.warning("Smart Turn unavailable, falling back to fixed-delay endpointing: %s", exc)
    return VADModels(silero=silero, smart_turn=smart_turn)
