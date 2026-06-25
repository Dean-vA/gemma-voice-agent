"""vLLM backend: proxy to a vLLM OpenAI-compatible server with audio input.

Audio is sent as a base64 data payload using the OpenAI ``input_audio`` content
part, which vLLM maps onto Gemma 4's conformer audio encoder. Output is streamed
so the gateway can measure TTFT from the first chunk.
"""
from __future__ import annotations

import base64
from typing import AsyncIterator

import numpy as np
from openai import AsyncOpenAI

from app.audio import to_wav_bytes
from app.backends.base import ChatBackend, Turn
from app.config import Settings


def _audio_part(samples: np.ndarray, sample_rate: int) -> dict:
    wav = to_wav_bytes(samples, sample_rate)
    b64 = base64.b64encode(wav).decode("ascii")
    return {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}}


class VLLMBackend(ChatBackend):
    name = "vllm"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        # API key is unused by a local vLLM server but the client requires one.
        self.client = AsyncOpenAI(base_url=settings.vllm_base_url, api_key="EMPTY")
        self.model = settings.qat_model_id

    async def load(self) -> None:
        # Discover the served model id (vLLM serves under the path it was given).
        try:
            models = await self.client.models.list()
            if models.data:
                self.model = models.data[0].id
        except Exception:
            # Leave the configured id; /health will surface connectivity issues.
            pass

    def _build_messages(
        self, system_prompt: str, history: list[Turn], user_audio: np.ndarray, instruction: str
    ) -> list[dict]:
        sr = self.settings.sample_rate
        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        for turn in history:
            if turn.role == "user":
                content: list[dict] = []
                if turn.text:
                    content.append({"type": "text", "text": turn.text})
                if turn.audio is not None:
                    content.append(_audio_part(turn.audio, sr))  # audio AFTER text
                messages.append({"role": "user", "content": content})
            else:
                messages.append({"role": "assistant", "content": turn.text})

        cur: list[dict] = []
        if instruction:
            cur.append({"type": "text", "text": instruction})
        cur.append(_audio_part(user_audio, sr))  # audio AFTER text (Gemma 4 rule)
        messages.append({"role": "user", "content": cur})
        return messages

    async def stream(
        self,
        system_prompt: str,
        history: list[Turn],
        user_audio: np.ndarray,
        instruction: str,
        max_new_tokens: int,
    ) -> AsyncIterator[str]:
        messages = self._build_messages(system_prompt, history, user_audio, instruction)
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=0.7,
            stream=True,
        )
        async for chunk in resp:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    async def health(self) -> dict:
        info = {"backend": self.name, "model": self.model, "vllm_base_url": self.settings.vllm_base_url}
        try:
            await self.client.models.list()
            info["reachable"] = True
        except Exception as exc:  # noqa: BLE001
            info["reachable"] = False
            info["error"] = str(exc)
        return info
