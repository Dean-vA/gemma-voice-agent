"""Transformers backend: load Gemma 4 E4B in-process and stream audio->text.

This is the reference / fallback path. It is slower than vLLM (eager/SDPA
attention — FlashAttention-2 is unavailable due to Gemma 4's mixed head dims),
but it is the most thoroughly documented audio route and de-risks the vLLM path.
"""
from __future__ import annotations

import asyncio
import os
import tempfile
import threading
from typing import AsyncIterator

import numpy as np

from app.audio import to_wav_bytes
from app.backends.base import ChatBackend, Turn
from app.config import Settings


def _load_model_class():
    """Resolve the multimodal model class across transformers versions."""
    import transformers

    for name in ("AutoModelForMultimodalLM", "Gemma4ForConditionalGeneration", "AutoModelForImageTextToText"):
        cls = getattr(transformers, name, None)
        if cls is not None:
            return cls
    raise ImportError("No suitable Gemma 4 multimodal model class found in transformers")


class TransformersBackend(ChatBackend):
    name = "transformers"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.processor = None
        self.model = None
        self._device = "cuda"

    async def load(self) -> None:
        await asyncio.to_thread(self._load_sync)
        # Warm up: a tiny silent clip to pay one-time compile/alloc costs.
        warm = np.zeros(int(0.5 * self.settings.sample_rate), dtype=np.float32)
        async for _ in self.stream(self.settings.system_prompt, [], warm, "Say hi.", 8):
            pass

    def _load_sync(self) -> None:
        import torch
        from transformers import AutoProcessor

        model_cls = _load_model_class()
        quant = self.settings.quant_mode.lower()

        model_id = self.settings.model_id
        kwargs: dict = {"torch_dtype": torch.bfloat16, "device_map": "cuda", "attn_implementation": "sdpa"}

        if quant == "qat":
            # Compressed-tensors checkpoint carries its own quantization config.
            model_id = self.settings.qat_model_id
        elif quant == "bnb4":
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        # else bf16: full precision, no quant config

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = model_cls.from_pretrained(model_id, **kwargs)
        self.model.eval()

    def _build_messages(
        self, system_prompt: str, history: list[Turn], audio_paths: dict[int, str], user_audio_path: str, instruction: str
    ) -> list[dict]:
        messages: list[dict] = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
        ]
        for idx, turn in enumerate(history):
            if turn.role == "user":
                content: list[dict] = []
                if turn.text:
                    content.append({"type": "text", "text": turn.text})
                if idx in audio_paths:
                    content.append({"type": "audio", "audio": audio_paths[idx]})  # audio AFTER text
                messages.append({"role": "user", "content": content})
            else:
                messages.append({"role": "assistant", "content": [{"type": "text", "text": turn.text}]})

        cur: list[dict] = [{"type": "text", "text": instruction or "Respond to the audio."}]
        cur.append({"type": "audio", "audio": user_audio_path})  # audio AFTER text
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
        import torch
        from transformers import TextIteratorStreamer

        sr = self.settings.sample_rate
        tmp_paths: list[str] = []

        def _write(samples: np.ndarray) -> str:
            fd, path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            with open(path, "wb") as fh:
                fh.write(to_wav_bytes(samples, sr))
            tmp_paths.append(path)
            return path

        try:
            audio_paths = {i: _write(t.audio) for i, t in enumerate(history) if t.audio is not None}
            user_path = _write(user_audio)
            messages = self._build_messages(system_prompt, history, audio_paths, user_path, instruction)

            inputs = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
            ).to(self.model.device)

            tok = getattr(self.processor, "tokenizer", self.processor)
            streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
            gen_kwargs = dict(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, streamer=streamer)

            loop = asyncio.get_running_loop()
            queue: asyncio.Queue[str | None] = asyncio.Queue()

            def _pump() -> None:
                for text in streamer:
                    loop.call_soon_threadsafe(queue.put_nowait, text)
                loop.call_soon_threadsafe(queue.put_nowait, None)

            def _generate() -> None:
                with torch.inference_mode():
                    self.model.generate(**gen_kwargs)

            gen_thread = threading.Thread(target=_generate, daemon=True)
            pump_thread = threading.Thread(target=_pump, daemon=True)
            gen_thread.start()
            pump_thread.start()

            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item
        finally:
            for p in tmp_paths:
                try:
                    os.remove(p)
                except OSError:
                    pass

    async def health(self) -> dict:
        return {
            "backend": self.name,
            "model": self.settings.qat_model_id if self.settings.quant_mode == "qat" else self.settings.model_id,
            "quant_mode": self.settings.quant_mode,
            "loaded": self.model is not None,
        }
