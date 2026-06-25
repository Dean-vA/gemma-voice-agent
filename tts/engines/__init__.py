"""Engine registry. The active engine is chosen by the TTS_ENGINE env var,
matching the engine installed in the image (see tts.Dockerfile build arg)."""
from __future__ import annotations

from tts.engines.base import TTSEngine


def make_engine(name: str) -> TTSEngine:
    name = (name or "kokoro").lower()
    if name == "kokoro":
        from tts.engines.kokoro_engine import KokoroEngine
        return KokoroEngine()
    if name == "piper":
        from tts.engines.piper_engine import PiperEngine
        return PiperEngine()
    if name == "xtts":
        from tts.engines.xtts_engine import XTTSEngine
        return XTTSEngine()
    if name == "chatterbox":
        from tts.engines.chatterbox_engine import ChatterboxEngine
        return ChatterboxEngine()
    raise ValueError(f"unknown TTS engine: {name}")
