"""Pluggable chat backends for the voice-agent gateway."""
from __future__ import annotations

from app.config import Settings
from app.backends.base import ChatBackend


def make_backend(settings: Settings) -> ChatBackend:
    if settings.is_vllm:
        from app.backends.vllm_backend import VLLMBackend

        return VLLMBackend(settings)
    from app.backends.transformers_backend import TransformersBackend

    return TransformersBackend(settings)
