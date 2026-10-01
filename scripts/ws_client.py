#!/usr/bin/env python3
"""Stream audio files to /ws/converse in real time and print the server's events.

A headless stand-in for the browser's Continuous mode (and for the robot):
exercises server-side VAD, turn detection, reopening and barge-in without a
microphone. Files are played back to back, separated by --gap seconds of
silence; use --barge-in to start the next file while the reply is still coming.

Usage:
    pip install websockets soundfile librosa
    python scripts/ws_client.py samples/hello.wav
    python scripts/ws_client.py a.wav b.wav --gap 0.4        # pause inside the reopen window
    python scripts/ws_client.py a.wav b.wav --barge-in       # talk over the reply
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

try:
    import librosa
    import numpy as np
    import websockets
except ImportError:
    sys.exit("pip install websockets soundfile librosa")

SR = 16000
FRAME = 512  # 32 ms


def load(path: str) -> np.ndarray:
    audio, _ = librosa.load(path, sr=SR, mono=True)
    return (np.clip(audio, -1, 1) * 32767).astype("<i2")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--url", default="ws://localhost:8000/ws/converse")
    ap.add_argument("--gap", type=float, default=4.0, help="silence between files (s)")
    ap.add_argument("--tail", type=float, default=8.0, help="silence streamed after the last file (s)")
    ap.add_argument("--barge-in", action="store_true", help="send the next file as soon as a reply starts")
    ap.add_argument("--no-speak", action="store_true", help="text replies only (skip TTS)")
    ap.add_argument("--transcribe", action="store_true")
    ap.add_argument("--asr-engine", default="", help="gemma | parakeet (default: the gateway's)")
    args = ap.parse_args()

    t0 = time.perf_counter()
    reply_started = asyncio.Event()

    def stamp() -> str:
        return f"[{time.perf_counter() - t0:7.2f}s]"

    async with websockets.connect(args.url, max_size=None) as ws:
        await ws.send(json.dumps({"type": "config", "speak": not args.no_speak, "transcribe": args.transcribe,
                                  "asr_engine": args.asr_engine}))

        async def reader() -> None:
            async for raw in ws:
                ev = json.loads(raw)
                kind = ev.pop("type")
                if kind == "token":
                    reply_started.set()
                    print(f"{stamp()} token          {ev['text']!r}")
                elif kind == "audio":
                    print(f"{stamp()} audio          #{ev['index']} {ev['sentence']!r}")
                elif kind == "done":
                    comps = ", ".join(f"{c['name']}={c['ms']:.0f}ms" for c in ev["metrics"]["components"])
                    print(f"{stamp()} done           {ev['reply']!r}\n{'':10} {comps}")
                else:
                    print(f"{stamp()} {kind:<14} {ev}")

        sent = 0  # frames so far; pace against the wall clock, not per-frame sleeps

        async def frame(data: bytes) -> None:
            nonlocal sent
            await ws.send(data)
            sent += 1
            await asyncio.sleep(max(0.0, t0 + sent * FRAME / SR - time.perf_counter()))

        async def send(pcm: np.ndarray) -> None:
            for i in range(0, len(pcm), FRAME):
                await frame(pcm[i:i + FRAME].tobytes())

        async def silence(seconds: float, until: asyncio.Event | None = None) -> None:
            for _ in range(int(seconds * SR / FRAME)):
                if until is not None and until.is_set():
                    return
                await frame(bytes(FRAME * 2))

        task = asyncio.create_task(reader())
        await silence(0.5)
        for n, path in enumerate(args.files):
            print(f"{stamp()} >>> sending {path}")
            reply_started.clear()
            await send(load(path))
            last = n == len(args.files) - 1
            if last:
                await silence(args.tail)
            elif args.barge_in:
                await silence(30.0, until=reply_started)
                await silence(1.0)  # let the reply get going before talking over it
            else:
                await silence(args.gap)
        task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
