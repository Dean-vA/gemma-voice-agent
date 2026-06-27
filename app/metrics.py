"""Latency / throughput instrumentation for a single audio->text turn.

TTFT (time-to-first-token) is the headline metric for spoken interaction: for
the audio-native path it captures audio-encode + prefill, i.e. how long the
robot would wait before it could start speaking.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager, contextmanager
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
    # Per-component / per-API-call latency breakdown for the turn.
    asr_ms: float | None = None              # optional speech-transcription pass
    tts_total_ms: float | None = None        # summed client round-trip of all TTS calls
    tts_server_ms: float | None = None       # summed server-side synth time (X-Synth-Ms)
    tts_calls: int = 0
    tts_audio_seconds: float = 0.0
    tts_segments: list[dict] = field(default_factory=list)
    # Ordered pipeline breakdown: [{name, ms, ...}] for audio_decode, asr,
    # llm, tts. Each entry sums to roughly total_ms (modulo overlap/streaming).
    components: list[dict] = field(default_factory=list)

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
        # Generic named spans (e.g. "asr") recorded via record()/record_async,
        # in insertion order, plus per-call TTS segment timings.
        self._spans: list[tuple[str, float, dict]] = []
        self._tts_segments: list[dict] = []
        if torch is not None and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    @contextmanager
    def record(self, name: str, **extra):
        """Time a synchronous block as a named pipeline component."""
        t0 = _now()
        try:
            yield
        finally:
            self._add_span(name, (_now() - t0) * 1000.0, extra)

    @asynccontextmanager
    async def record_async(self, name: str, **extra):
        """Time an async block (e.g. an API call) as a named component."""
        t0 = _now()
        try:
            yield
        finally:
            self._add_span(name, (_now() - t0) * 1000.0, extra)

    def _add_span(self, name: str, ms: float, extra: dict) -> None:
        self._spans.append((name, ms, extra))
        if name == "asr":
            self.m.asr_ms = ms

    def add_tts_segment(self, *, text: str, client_ms: float, server_ms: float | None = None,
                        audio_seconds: float | None = None, engine: str | None = None) -> None:
        """Record one TTS synth call (one sentence -> one wav)."""
        seg: dict = {"index": len(self._tts_segments), "chars": len(text),
                     "client_ms": round(client_ms, 1)}
        if server_ms is not None:
            seg["server_ms"] = round(server_ms, 1)
        if audio_seconds is not None:
            seg["audio_seconds"] = round(audio_seconds, 3)
        if engine:
            seg["engine"] = engine
        self._tts_segments.append(seg)

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

        if self._tts_segments:
            self.m.tts_segments = self._tts_segments
            self.m.tts_calls = len(self._tts_segments)
            self.m.tts_total_ms = sum(s["client_ms"] for s in self._tts_segments)
            server = [s["server_ms"] for s in self._tts_segments if "server_ms" in s]
            self.m.tts_server_ms = sum(server) if server else None
            self.m.tts_audio_seconds = sum(s.get("audio_seconds", 0.0) for s in self._tts_segments)
            if self.m.tts_engine is None:
                self.m.tts_engine = next((s["engine"] for s in reversed(self._tts_segments)
                                          if s.get("engine")), None)

        self.m.components = self._build_components()
        return self.m

    def _build_components(self) -> list[dict]:
        comps: list[dict] = []
        if self.m.preprocess_ms:
            comps.append({"name": "audio_decode", "ms": round(self.m.preprocess_ms, 2)})
        for name, ms, extra in self._spans:
            comps.append({"name": name, "ms": round(ms, 2), **extra})
        if self.m.ttft_ms or self.m.generation_ms:
            comps.append({
                "name": "llm",
                "ms": round(self.m.ttft_ms + self.m.generation_ms, 2),
                "ttft_ms": round(self.m.ttft_ms, 2),
                "generation_ms": round(self.m.generation_ms, 2),
                "output_tokens": self.m.output_tokens,
            })
        if self.m.tts_total_ms:
            comps.append({
                "name": "tts",
                "ms": round(self.m.tts_total_ms, 2),
                "server_ms": round(self.m.tts_server_ms, 2) if self.m.tts_server_ms is not None else None,
                "calls": self.m.tts_calls,
                "audio_seconds": round(self.m.tts_audio_seconds, 3),
                "engine": self.m.tts_engine,
            })
        return comps


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
