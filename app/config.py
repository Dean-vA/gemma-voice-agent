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

    # Streamed voice loop (/ws/converse): server-side Silero VAD + Smart Turn.
    # Defaults follow huggingface/speech-to-speech, the Reachy Mini backend.
    vad_threshold: float = 0.6               # speech trigger; stays active to threshold-0.15
    vad_min_silence_ms: int = 64             # silence that closes a speech segment
    vad_min_speech_ms: int = 384             # active speech before "speech started" (and barge-in)
    vad_min_speech_continuation_ms: int = 192  # same, for speech resuming inside a reopen window
    vad_speech_pad_ms: int = 500             # pre-roll kept before the trigger
    # After a segment closes the reply is prepared speculatively but held back
    # for a grace period; speech resuming before it is released reopens the
    # same turn. Smart Turn picks the grace: reopen_ms when the turn sounds
    # complete (or Smart Turn is off), smart_turn_max_wait_ms when it doesn't.
    smart_turn: bool = True
    smart_turn_threshold: float = 0.5
    smart_turn_max_wait_ms: int = 2000         # grace for an incomplete-sounding turn
    smart_turn_incomplete_delay_ms: int = 600  # ...and delay before working on its reply
    reopen_ms: int = 800                     # grace for a complete-sounding turn
    unanswered_reopen_ms: int = 7000         # a turn with no reply released yet can reopen this long
    # Let the user talk over the robot. Relies on echo cancellation at the
    # client; set false to ignore the mic while the robot is replying/speaking.
    vad_barge_in: bool = True

    # Transcription. "gemma" asks the LLM itself for a transcript (an extra LLM
    # call); "parakeet" uses Parakeet TDT via nano-parakeet, loaded in the
    # gateway. Clients can pick per request; this is the default.
    asr_engine: str = "gemma"
    # What the LLM answers from: "audio" (Gemma hears the speech itself) or
    # "transcript" (cascade: the ASR transcript is sent as text, no audio;
    # Parakeet unless another asr_engine is requested).
    llm_input: str = "audio"
    parakeet: bool = True                    # load Parakeet at startup (uses some VRAM on cuda)
    parakeet_model: str = "nvidia/parakeet-tdt-0.6b-v3"
    parakeet_device: str = "auto"            # auto | cuda | cpu

    # Read-aloud clips recorded on the ASR eval page (/web/asr-eval.html).
    asr_eval_dir: str = "samples/asr"

    # Conversation
    max_history_turns: int = 8
    # Replayed user audio dominates context in long chats. If >= 0, keep raw
    # audio only for the most recent N user turns *that have a transcript*;
    # older ones fall back to their transcribed text. Turns without text
    # (transcribe=false) always keep audio. -1 (default) = keep all audio,
    # preserving existing behavior.
    history_keep_audio_turns: int = -1
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
