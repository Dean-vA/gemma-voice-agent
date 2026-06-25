"""ChatBackend interface shared by the vLLM and Transformers implementations."""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import AsyncIterator

import numpy as np


@dataclass
class Turn:
    """One conversation turn.

    ``audio`` carries the user's spoken input (mono float32 at the configured
    sample rate). Assistant turns carry ``text`` only.
    """

    role: str  # "user" | "assistant"
    text: str = ""
    audio: np.ndarray | None = None


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
    async def stream(
        self,
        system_prompt: str,
        history: list[Turn],
        user_audio: np.ndarray,
        instruction: str,
        max_new_tokens: int,
    ) -> AsyncIterator[str]:
        """Yield response text chunks as they are generated."""
        raise NotImplementedError
        yield ""  # pragma: no cover  (makes this an async generator)

    async def health(self) -> dict:
        return {"backend": self.name}
