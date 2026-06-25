"""Environment-driven settings for the Gemma 4 E4B voice-agent gateway."""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Backend selection: "vllm" or "transformers"
    backend: str = "vllm"

    # Model ids
    model_id: str = "google/gemma-4-E4B-it"
    qat_model_id: str = "google/gemma-4-E4B-it-qat-w4a16"

    # Transformers quantization: qat | bnb4 | bf16
    quant_mode: str = "bnb4"

    # vLLM OpenAI-compatible endpoint
    vllm_base_url: str = "http://vllm:8001/v1"

    # TTS engines for the /converse voice loop. Each engine is its own container
    # (avoids dependency conflicts); the gateway probes which are reachable and
    # the UI lets you pick at runtime. All listen on 8200 inside the compose net.
    tts_engine: str = "kokoro"  # preferred default when the client sends none
    tts_engines: dict[str, str] = {
        "kokoro": "http://tts-kokoro:8200",
        "piper": "http://tts-piper:8200",
        "xtts": "http://tts-xtts:8200",
        "chatterbox": "http://tts-chatterbox:8200",
    }
    tts_voice: str | None = None

    # Audio
    sample_rate: int = 16000
    max_audio_seconds: float = 30.0

    # Conversation
    max_history_turns: int = 8
    max_new_tokens: int = 256
    system_prompt: str = (
        "You are a friendly humanoid robot assistant. Reply in short, natural "
        "spoken sentences. Be concise and conversational."
    )

    # Server
    port: int = 8000

    # Hugging Face
    hf_token: str | None = None

    @property
    def is_vllm(self) -> bool:
        return self.backend.lower() == "vllm"


@lru_cache
def get_settings() -> Settings:
    return Settings()
