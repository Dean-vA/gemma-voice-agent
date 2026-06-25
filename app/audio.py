"""Audio decoding helpers: bytes -> 16 kHz mono float32, with a duration cap."""
from __future__ import annotations

import io
from dataclasses import dataclass

import librosa
import numpy as np
import soundfile as sf


@dataclass
class DecodedAudio:
    samples: np.ndarray  # float32, mono, at target sample_rate
    sample_rate: int
    duration_s: float
    trimmed: bool


def decode_audio(
    data: bytes,
    target_sr: int = 16000,
    max_seconds: float = 30.0,
) -> DecodedAudio:
    """Decode arbitrary audio bytes to mono float32 at ``target_sr``.

    Gemma 4's audio encoder expects 16 kHz mono. We resample here so the
    server never depends on the client sending a particular rate, and we cap
    the clip to ``max_seconds`` (the model's per-clip limit).
    """
    samples, sr = librosa.load(io.BytesIO(data), sr=target_sr, mono=True)
    samples = samples.astype(np.float32, copy=False)

    trimmed = False
    max_len = int(max_seconds * target_sr)
    if samples.shape[0] > max_len:
        samples = samples[:max_len]
        trimmed = True

    duration = float(samples.shape[0]) / target_sr
    return DecodedAudio(samples=samples, sample_rate=target_sr, duration_s=duration, trimmed=trimmed)


def to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """Encode float32 mono samples to an in-memory 16-bit PCM WAV.

    Used by the vLLM backend, which sends audio as a base64 data-URL.
    """
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()
