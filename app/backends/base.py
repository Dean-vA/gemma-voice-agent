"""ChatBackend interface shared by the vLLM and Transformers implementations."""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import AsyncIterator

import numpy as np


@dataclass
class Turn:
    """One conversation turn.

    ``audio`` carries the user's spoken input (mono float32 at the configured
    sample rate). ``image`` carries an optional encoded still (JPEG/PNG bytes)
    the user showed the robot. Assistant turns carry ``text`` only.
    """

    role: str  # "user" | "assistant"
    text: str = ""
    audio: np.ndarray | None = None
    image: bytes | None = None


@dataclass
class TurnState:
    """Progress of an in-flight turn, readable by whoever may cancel it."""

    user_text: str = ""
    parts: list[str] = field(default_factory=list)
    # When not None, filled with a text-only copy of the prompt sent to the LLM
    # (debug: lets a client see exactly what the model answered from).
    prompt: list[dict] | None = None

    @property
    def reply(self) -> str:
        return "".join(self.parts)


class ChatBackend(abc.ABC):
    """Streams an audio-native (audio-in, text-out) response.

    Implementations must place audio *after* text in the prompt (Gemma 4
    requirement) and stream output tokens so the gateway can measure TTFT.
    """

    name: str = "base"

    @abc.abstractmethod
    async def load(self) -> None:
        """Initialise the backend (load model / open client) and warm up."""

    @abc.abstractmethod
    def describe_prompt(
        self,
        system_prompt: str,
        history: list[Turn],
        user_audio: np.ndarray,
        instruction: str,
        user_image: bytes | None = None,
        user_text: str | None = None,
    ) -> list[dict]:
        """Text-only view of the prompt ``stream`` sends, for debugging: one
        {role, content} per message, audio/images as placeholders. Backends
        that build their own messages should override this to use them."""
        msgs = [{"role": "system", "content": system_prompt}]
        for t in history:
            msgs.append({"role": t.role, "content": t.text})
        cur = [instruction] if instruction else []
        if user_image is not None:
            cur.append("[image]")
        cur.append(user_text if user_text else f"[audio {len(user_audio) / 16000:.1f}s]")
        msgs.append({"role": "user", "content": " ".join(cur)})
        return msgs

    async def stream(
        self,
        system_prompt: str,
        history: list[Turn],
        user_audio: np.ndarray,
        instruction: str,
        max_new_tokens: int,
        user_image: bytes | None = None,
        user_text: str | None = None,
    ) -> AsyncIterator[str]:
        """Yield response text chunks as they are generated.

        ``user_image`` is optional encoded image bytes (JPEG/PNG) shown with the
        current turn; placed before audio in the prompt. ``user_text``, when
        given, is a transcript of the turn and is sent *instead of* the audio
        (cascaded ASR -> LLM mode).
        """
        raise NotImplementedError
        yield ""  # pragma: no cover  (makes this an async generator)

    async def health(self) -> dict:
        return {"backend": self.name}
