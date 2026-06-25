"""Pydantic response models for the gateway."""
from __future__ import annotations

from pydantic import BaseModel


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    transcript: str | None = None
    metrics: dict


class HealthResponse(BaseModel):
    status: str
    backend: str
    quant_mode: str
    settings: dict
    gpu: dict
    backend_info: dict
    tts: dict = {}
