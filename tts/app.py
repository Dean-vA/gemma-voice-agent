"""TTS microservice: text -> speech (WAV). One engine per container, chosen by
TTS_ENGINE. Kept separate from the gateway so heavy/conflicting TTS deps don't
pollute the inference image and engines can be swapped by rebuild."""
from __future__ import annotations

import base64
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel

from tts.engines import make_engine

ENGINE_NAME = os.environ.get("TTS_ENGINE", "kokoro")


class TTSRequest(BaseModel):
    text: str
    voice: str | None = None
    encoding: str = "wav"  # "wav" -> raw bytes; "base64" -> JSON


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = make_engine(ENGINE_NAME)
    engine.load()
    # warm up so the first real request isn't penalised
    engine.synthesize("Ready.")
    app.state.engine = engine
    yield


app = FastAPI(title=f"TTS · {ENGINE_NAME}", lifespan=lifespan)


@app.get("/health")
async def health():
    eng = app.state.engine
    return {"status": "ok", "engine": eng.name, "sample_rate": eng.sample_rate}


@app.post("/tts")
async def tts(req: TTSRequest):
    eng = app.state.engine
    t0 = time.perf_counter()
    samples, sr = eng.synthesize(req.text, req.voice)
    synth_ms = (time.perf_counter() - t0) * 1000.0
    wav = eng.to_wav(samples, sr)
    audio_seconds = len(samples) / sr
    headers = {
        "X-Engine": eng.name,
        "X-Synth-Ms": f"{synth_ms:.1f}",
        "X-Audio-Seconds": f"{audio_seconds:.3f}",
        "X-Sample-Rate": str(sr),
    }
    if req.encoding == "base64":
        return {
            "engine": eng.name, "sample_rate": sr, "audio_seconds": audio_seconds,
            "synth_ms": synth_ms, "wav_base64": base64.b64encode(wav).decode("ascii"),
        }
    return Response(content=wav, media_type="audio/wav", headers=headers)
