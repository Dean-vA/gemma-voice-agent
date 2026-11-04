"""Speaker-aware transcription for Ruxithid's self-hosted speech backend.

Speaks the OpenAI transcription API (`POST /v1/audio/transcriptions`) so
`speech-to-speech --stt openai` can use it. Each utterance is transcribed with
Parakeet and matched against enrolled voiceprints with TitaNet; a recognised
speaker comes back as a prefix:  [Peregrine] Give us back the townsfolk.

Needs torch + NeMo. The DnD-Session-Summaries environment already has both
models cached, so run it with that interpreter:

    <DnD-Session-Summaries>\\.venv\\Scripts\\python speaker_id\\server.py serve
"""

from __future__ import annotations

import argparse
import io
import json
import os
import threading
import time
import tomllib
import urllib.request
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse

HERE = Path(__file__).parent
# Where the enrolled voices live; a container points this at a volume (SPEAKER_ID_DATA).
DATA = Path(os.environ.get("SPEAKER_ID_DATA") or HERE)
VOICEPRINTS = DATA / "voiceprints.npz"
ROSTER = DATA / "roster.toml"

ASR_MODEL = "nvidia/parakeet-tdt-0.6b-v3"
SPEAKER_MODEL = "nvidia/speakerverification_en_titanet_large"
SAMPLE_RATE = 16000
DEFAULT_PORT = 8790

MIN_SECONDS = 0.5  # below this a clip is mostly padding, with too little voice to score at all
# Voices heard through the robot's mic score well below the same voice on its enrolment mic
# (0.24-0.59 seen, median 0.41), so these are set low. On the stored clips no enrolled player
# is mistaken for another at these values; the cost is that about one in five clips from
# someone who was never enrolled gets a player's name. Raise them if guests join the table.
MATCH_THRESHOLD = 0.30  # cosine similarity to a player's mean voiceprint
MATCH_MARGIN = 0.08  # and this far ahead of the runner-up, or it stays unlabelled
# A weaker match still counts when it is the clear winner.
CLEAR_WINNER_THRESHOLD = 0.25
CLEAR_WINNER_MARGIN = 0.10
# One-word remarks carry little voice, so they have to match more convincingly to earn a name,
# and they are never used to learn a voice.
SHORT_SECONDS = 1.0
SHORT_THRESHOLD = 0.40
SHORT_MARGIN = 0.15


class Voiceprints:
    """Per-player TitaNet embeddings, one row per enrolled sample."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self.samples: dict[str, np.ndarray] = {}
        if path.exists():
            with np.load(path) as data:
                self.samples = {player: data[player] for player in data.files}

    def add(self, player: str, embedding: np.ndarray) -> None:
        existing = self.samples.get(player)
        row = embedding[None, :]
        self.samples[player] = row if existing is None else np.vstack([existing, row])
        self._save()

    def forget(self, player: str) -> bool:
        removed = self.samples.pop(player, None) is not None
        self._save()
        return removed

    def _save(self) -> None:
        np.savez(self._path, **self.samples)

    def identify(self, embedding: np.ndarray, short: bool = False) -> tuple[str | None, dict[str, float]]:
        """Return the matching player (or None) and every player's score."""
        scores = {}
        for player, rows in self.samples.items():
            mean = rows.mean(axis=0)
            scores[player] = float(np.dot(embedding, mean / np.linalg.norm(mean)))
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if not ranked:
            return None, scores
        best = ranked[0][1]
        lead = best - ranked[1][1] if len(ranked) > 1 else best
        if short:
            matched = best >= SHORT_THRESHOLD and lead >= SHORT_MARGIN
        else:
            strong = best >= MATCH_THRESHOLD and lead >= MATCH_MARGIN
            clear_winner = best >= CLEAR_WINNER_THRESHOLD and lead >= CLEAR_WINNER_MARGIN
            matched = strong or clear_winner
        return (ranked[0][0] if matched else None), scores


class Models:
    """Parakeet for the words, TitaNet for the voice. One utterance at a time on the GPU."""

    def __init__(self) -> None:
        import nemo.collections.asr as nemo_asr
        import torch

        self._torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._lock = threading.Lock()

        self.asr = nemo_asr.models.ASRModel.from_pretrained(model_name=ASR_MODEL).to(self.device)
        self.asr.freeze()
        self.asr.eval()

        # TitaNet's checkpoint is an old-style pickle that torch >= 2.6 refuses by default.
        original_load = torch.load
        torch.load = lambda *a, **kw: original_load(*a, **{**kw, "weights_only": False})
        try:
            self.speaker = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(SPEAKER_MODEL)
        finally:
            torch.load = original_load
        self.speaker = self.speaker.to(self.device)
        self.speaker.eval()

    def transcribe(self, audio: np.ndarray) -> str:
        with self._lock, self._torch.no_grad():
            return self.asr.transcribe([audio], batch_size=1, verbose=False)[0].text.strip()

    def embed(self, audio: np.ndarray) -> np.ndarray:
        signal = self._torch.from_numpy(audio).unsqueeze(0).to(self.device)
        length = self._torch.tensor([signal.shape[1]], device=self.device)
        with self._lock, self._torch.no_grad():
            _, embedding = self.speaker.forward(input_signal=signal, input_signal_length=length)
        vector = embedding.squeeze(0).float().cpu().numpy()
        return vector / np.linalg.norm(vector)


def decode(data: bytes | Path) -> np.ndarray:
    """Decode an audio file to 16 kHz mono float32."""
    import soundfile as sf

    audio, rate = sf.read(io.BytesIO(data) if isinstance(data, bytes) else str(data), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != SAMPLE_RATE:
        import librosa

        audio = librosa.resample(audio, orig_sr=rate, target_sr=SAMPLE_RATE)
    return np.ascontiguousarray(audio, dtype=np.float32)


def load_roster() -> dict[str, str]:
    """Player name -> the name Ruxithid should hear (their character)."""
    if not ROSTER.exists():
        return {}
    return tomllib.loads(ROSTER.read_text(encoding="utf-8"))


def create_app() -> FastAPI:
    app = FastAPI(title="Ruxithid speaker-aware STT")
    models = Models()
    voices = Voiceprints(VOICEPRINTS)
    state: dict[str, str | None] = {"enrolling": None}
    recent: deque[str] = deque(maxlen=20)

    def note(message: str) -> None:
        """Record who an utterance sounded like (never what was said)."""
        recent.append(f"{time.strftime('%H:%M:%S')}  {message}")
        print(f"speaker-id: {message}", flush=True)

    @app.post("/v1/audio/transcriptions")
    def transcriptions(file: UploadFile, response_format: str = Form("json")) -> Any:
        audio = decode(file.file.read())
        text = models.transcribe(audio) if audio.size else ""

        seconds = audio.size / SAMPLE_RATE
        short = seconds < SHORT_SECONDS
        if text and seconds >= MIN_SECONDS:
            embedding = models.embed(audio)
            if state["enrolling"]:
                if short:
                    note(f"{seconds:.2f}s is too short to learn from (needs {SHORT_SECONDS}s)")
                else:
                    voices.add(state["enrolling"], embedding)
                    note(f"{seconds:.1f}s added to {state['enrolling']}'s voiceprint")
            else:
                player, scores = voices.identify(embedding, short=short)
                if player:
                    text = f"[{load_roster().get(player, player)}] {text}"
                ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:3]
                note(
                    f"{seconds:.2f}s{' (short)' if short else ''} -> {player or 'nobody (unlabelled)'} | "
                    + ", ".join(f"{name} {score:.2f}" for name, score in ranked)
                )
        elif text:
            note(f"{seconds:.2f}s is too short to identify (needs {MIN_SECONDS}s)")

        if response_format == "text":
            return PlainTextResponse(text)
        return {"text": text}

    @app.post("/identify")
    def identify(file: UploadFile) -> dict[str, Any]:
        """Debugging aid: scores against every enrolled player, without transcribing."""
        audio = decode(file.file.read())
        player, scores = voices.identify(models.embed(audio))
        return {"player": player, "seconds": round(audio.size / SAMPLE_RATE, 2), "scores": scores}

    @app.post("/enroll/{player}")
    def enroll_start(player: str) -> dict[str, Any]:
        """Until /enroll/stop, every utterance heard is added to this player's voiceprint."""
        state["enrolling"] = player
        return {"enrolling": player}

    @app.post("/enroll-stop")
    def enroll_stop() -> dict[str, Any]:
        player, state["enrolling"] = state["enrolling"], None
        return {"stopped": player, "samples": len(voices.samples.get(player, [])) if player else 0}

    @app.get("/speakers")
    def speakers() -> dict[str, Any]:
        roster = load_roster()
        return {
            "enrolling": state["enrolling"],
            "speakers": {p: {"samples": len(rows), "heard_as": roster.get(p, p)} for p, rows in voices.samples.items()},
            "recent": list(recent),
        }

    @app.delete("/speakers/{player}")
    def forget(player: str) -> dict[str, Any]:
        if not voices.forget(player):
            raise HTTPException(404, f"No voiceprint for {player!r}")
        return {"forgot": player}

    return app


def call(method: str, path: str, port: int) -> Any:
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method=method)
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("serve", help="run the transcription service")
    p.add_argument("--host", default="127.0.0.1", help="0.0.0.0 in a container, so the speech server can reach it")

    p = sub.add_parser("import", help="enrol a player from existing recordings (no server needed)")
    p.add_argument("player")
    p.add_argument("files", nargs="+", type=Path)

    p = sub.add_parser("enroll", help="have the running service learn a player from what it hears next")
    p.add_argument("player", nargs="?", help="omit to stop enrolling")

    sub.add_parser("list", help="show who the running service knows")

    p = sub.add_parser("forget", help="delete a player's voiceprint from the running service")
    p.add_argument("player")

    args = parser.parse_args()
    if args.command == "serve":
        import uvicorn

        uvicorn.run(create_app(), host=args.host, port=args.port, log_level="warning")
    elif args.command == "import":
        models, voices = Models(), Voiceprints(VOICEPRINTS)
        for path in args.files:
            audio = decode(path)
            if audio.size < SHORT_SECONDS * SAMPLE_RATE:
                print(f"  skipped {path.name}: under {SHORT_SECONDS}s")
                continue
            voices.add(args.player, models.embed(audio))
            print(f"  {args.player} <- {path.name} ({audio.size / SAMPLE_RATE:.1f}s)")
    elif args.command == "enroll":
        print(call("POST", f"/enroll/{args.player}" if args.player else "/enroll-stop", args.port))
    elif args.command == "list":
        print(json.dumps(call("GET", "/speakers", args.port), indent=2))
    elif args.command == "forget":
        print(call("DELETE", f"/speakers/{args.player}", args.port))


if __name__ == "__main__":
    main()
