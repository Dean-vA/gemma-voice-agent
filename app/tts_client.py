"""Async client for the TTS microservice, plus sentence chunking so we can
synthesize speech sentence-by-sentence as the LLM streams (low time-to-audio)."""
from __future__ import annotations

import re
import time

import httpx

_SENTENCE_END = re.compile(r"(.+?[.!?…]+[\"')\]]?\s+)", re.DOTALL)


class TTSClient:
    def __init__(self, base_url: str, voice: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.voice = voice
        self._client = httpx.AsyncClient(timeout=120)

    async def health(self) -> dict:
        try:
            r = await self._client.get(f"{self.base_url}/health")
            return r.json() | {"reachable": True}
        except Exception as exc:  # noqa: BLE001
            return {"reachable": False, "error": str(exc)}

    async def synthesize(self, text: str) -> tuple[bytes, float]:
        """Return (wav_bytes, synth_ms)."""
        t0 = time.perf_counter()
        r = await self._client.post(
            f"{self.base_url}/tts",
            json={"text": text, "voice": self.voice, "encoding": "wav"},
        )
        r.raise_for_status()
        return r.content, (time.perf_counter() - t0) * 1000.0


def split_sentences(buffer: str) -> tuple[list[str], str]:
    """Pull complete sentences out of a streaming buffer.

    Returns (complete_sentences, remainder). Keeps the trailing partial sentence
    in the remainder until more text arrives or the stream ends.
    """
    sentences: list[str] = []
    last = 0
    for m in _SENTENCE_END.finditer(buffer):
        sentences.append(m.group(1).strip())
        last = m.end()
    return sentences, buffer[last:]
