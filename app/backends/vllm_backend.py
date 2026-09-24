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


def _image_part(image_bytes: bytes) -> dict:
    mime = "image/png" if image_bytes[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


class VLLMBackend(ChatBackend):
    name = "vllm"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        # API key is unused by a local vLLM server but the client requires one.
        self.client = AsyncOpenAI(base_url=settings.vllm_base_url, api_key="EMPTY")
        self.model = settings.qat_model_id
        self._discovered = False

    async def load(self) -> None:
        # Discover the served model id (vLLM serves under --served-model-name).
        try:
            models = await self.client.models.list()
            if models.data:
                self.model = models.data[0].id
                self._discovered = True
        except Exception:
            # Leave the configured id; /health will surface connectivity issues.
            pass

    async def _ensure_model(self) -> None:
        # If the gateway came up before vLLM, startup discovery failed and every
        # request 404s on the fallback id -- retry discovery until it succeeds.
        if not self._discovered:
            await self.load()

    def _build_messages(
        self, system_prompt: str, history: list[Turn], user_audio: np.ndarray,
        instruction: str, user_image: bytes | None = None,
    ) -> list[dict]:
        sr = self.settings.sample_rate
        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        # Past user turns go in as TEXT only. Re-sending every earlier clip puts
        # several audio items in one request, which crashes vLLM 0.26's Gemma-4
        # audio encoder ('list' object has no attribute 'squeeze') and kills the
        # engine. Only the current turn carries audio/image.
        for turn in history:
            if turn.role == "user":
                text = turn.text or "(the user spoke; audio not retained)"
                messages.append({"role": "user", "content": text})
            else:
                messages.append({"role": "assistant", "content": turn.text})

        cur: list[dict] = []
        if instruction:
            cur.append({"type": "text", "text": instruction})
        if user_image is not None:
            cur.append(_image_part(user_image))  # image after text, before audio
        cur.append(_audio_part(user_audio, sr))  # audio last (Gemma 4 rule)
        messages.append({"role": "user", "content": cur})
        return messages

    async def stream(
        self,
        system_prompt: str,
        history: list[Turn],
        user_audio: np.ndarray,
        instruction: str,
        max_new_tokens: int,
        user_image: bytes | None = None,
    ) -> AsyncIterator[str]:
        await self._ensure_model()
        messages = self._build_messages(system_prompt, history, user_audio, instruction, user_image)
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
