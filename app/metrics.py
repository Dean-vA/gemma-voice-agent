"""Latency / throughput instrumentation for a single audio->text turn.

TTFT (time-to-first-token) is the headline metric for spoken interaction: for
the audio-native path it captures audio-encode + prefill, i.e. how long the
robot would wait before it could start speaking.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

try:
    import torch
except Exception:  # torch is absent in the pure-vLLM gateway image variant
    torch = None  # type: ignore


def _now() -> float:
    return time.perf_counter()


@dataclass
class TurnMetrics:
    backend: str = ""
    audio_seconds: float = 0.0
    preprocess_ms: float = 0.0
    ttft_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0
    output_tokens: int = 0
    tokens_per_sec: float = 0.0
    peak_vram_mb: float | None = None
    prefix_cache_hit: bool | None = None
    # Voice loop: wall time from request to first spoken audio chunk (TTS).
    time_to_first_audio_ms: float | None = None
    tts_engine: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


class TurnTimer:
    """Accumulates timestamps across the phases of one turn.

    Usage:
        timer = TurnTimer(backend)
        timer.mark_preprocess_done()      # after audio decode
        # ... start generation ...
        timer.mark_first_token()          # on first streamed token
        timer.add_token() per token
        timer.finish()
    """

    def __init__(self, backend: str, audio_seconds: float = 0.0) -> None:
        self.m = TurnMetrics(backend=backend, audio_seconds=audio_seconds)
        self._t_start = _now()
        self._t_preprocess_done: float | None = None
        self._t_first_token: float | None = None
        self._tokens = 0
        if torch is not None and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def mark_preprocess_done(self) -> None:
        self._t_preprocess_done = _now()
        self.m.preprocess_ms = (self._t_preprocess_done - self._t_start) * 1000.0

    def mark_first_token(self) -> None:
        if self._t_first_token is None:
            self._t_first_token = _now()
            self.m.ttft_ms = (self._t_first_token - self._t_start) * 1000.0

    def add_token(self, n: int = 1) -> None:
        self._tokens += n

    def finish(self, output_tokens: int | None = None, prefix_cache_hit: bool | None = None) -> TurnMetrics:
        t_end = _now()
        self.m.total_ms = (t_end - self._t_start) * 1000.0
        self.m.output_tokens = output_tokens if output_tokens is not None else self._tokens
        if self._t_first_token is not None:
            gen_s = max(t_end - self._t_first_token, 1e-9)
            self.m.generation_ms = gen_s * 1000.0
            self.m.tokens_per_sec = self.m.output_tokens / gen_s
        if prefix_cache_hit is not None:
            self.m.prefix_cache_hit = prefix_cache_hit
        if torch is not None and torch.cuda.is_available():
            self.m.peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        return self.m


def gpu_info() -> dict:
    if torch is None or not torch.cuda.is_available():
        return {"cuda": False}
    free, total = torch.cuda.mem_get_info()
    return {
        "cuda": True,
        "device_name": torch.cuda.get_device_name(0),
        "vram_total_mb": total / (1024 * 1024),
        "vram_free_mb": free / (1024 * 1024),
    }
