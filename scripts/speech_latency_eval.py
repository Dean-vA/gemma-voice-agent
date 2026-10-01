#!/usr/bin/env python3
"""Time to first speech: Gemma hearing the audio vs. reading a Parakeet transcript.

Replays the clips recorded in the console's ASR eval mode (samples/asr/) through
POST /converse in both pipeline configurations and compares how long the gateway
takes to produce the first spoken sentence:

  audio       speech -> Gemma (audio-native) -> TTS
  transcript  speech -> Parakeet -> Gemma (text) -> TTS

Times are the gateway's own (request received -> first sentence synthesized), so
upload and playback are excluded. In live Continuous mode both configurations
additionally wait for the VAD's turn-end decision, which is identical for both.

Each clip is sent once per configuration by default: vLLM caches prompt
prefixes, so a repeated clip gets an unrealistically fast first token. Per-run
rows go to samples/asr/speech_latency.csv.

Usage:
    pip install requests
    python scripts/speech_latency_eval.py
    python scripts/speech_latency_eval.py --host http://localhost:8000 --engine kokoro
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("pip install requests")

CONFIGS = ["audio", "transcript"]


def pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    k = (len(values) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def converse(host: str, wav: Path, llm_input: str, engine: str) -> dict:
    with open(wav, "rb") as fh:
        r = requests.post(f"{host}/converse", files={"audio": (wav.name, fh, "audio/wav")},
                          data={"llm_input": llm_input, "transcribe": "false", "engine": engine}, timeout=300)
    r.raise_for_status()
    sid, metrics, reply = None, None, ""
    for block in r.text.split("\n\n"):
        lines = block.split("\n")
        event = next((ln[6:].strip() for ln in lines if ln.startswith("event:")), "")
        data = "".join(ln[5:].strip() for ln in lines if ln.startswith("data:"))
        if event == "session":
            sid = json.loads(data)["session_id"]
        elif event == "done":
            done = json.loads(data)
            metrics, reply = done["metrics"], done["reply"]
    if sid:
        requests.post(f"{host}/reset", data={"session_id": sid}, timeout=30)  # each turn starts from an empty history
    if not metrics or metrics.get("time_to_first_audio_ms") is None:
        raise RuntimeError("the reply contained no audio")
    asr = metrics.get("asr_ms") or 0.0
    return {"first_speech_ms": metrics["time_to_first_audio_ms"], "asr_ms": asr,
            "llm_first_token_ms": metrics["ttft_ms"] - asr,  # ttft is measured from the start of the request
            "first_tts_ms": metrics["tts_segments"][0]["client_ms"], "reply": reply}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="http://localhost:8000")
    ap.add_argument("--dir", default="samples/asr", help="folder with manifest.json and the WAVs")
    ap.add_argument("--engine", default="", help="TTS engine (default: the gateway's)")
    ap.add_argument("--runs", type=int, default=1, help="runs per clip per configuration (repeats hit the prompt cache)")
    ap.add_argument("--out", help="CSV path (default: <dir>/speech_latency.csv)")
    args = ap.parse_args()

    folder = Path(args.dir)
    manifest_path = folder / "manifest.json"
    if not manifest_path.exists():
        sys.exit(f"No {manifest_path}. Record clips first in the console's ASR eval mode.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    clips = [(cid, c) for cid, c in sorted(manifest.items()) if (folder / c["file"]).exists()]
    if not clips:
        sys.exit(f"{manifest_path} lists no clips that exist in {folder}")

    print(f"{len(clips)} clips x {args.runs} run(s) x {len(CONFIGS)} configurations\n")
    for cfg in CONFIGS:  # warm-up, discarded
        converse(args.host, folder / clips[0][1]["file"], cfg, args.engine)

    rows: list[dict] = []
    for i, (cid, clip) in enumerate(clips):
        for run in range(args.runs):
            # Alternate which configuration goes first so neither always follows a warm one.
            for cfg in (CONFIGS if (i + run) % 2 == 0 else CONFIGS[::-1]):
                res = converse(args.host, folder / clip["file"], cfg, args.engine)
                rows.append({"config": cfg, "clip": cid, "run": run + 1,
                             "audio_seconds": clip["audio_seconds"], **res})
        mine = {cfg: [r["first_speech_ms"] for r in rows if r["clip"] == cid and r["config"] == cfg] for cfg in CONFIGS}
        print(f"  {cid:<8} audio {pct(mine['audio'], .5):6.0f} ms   transcript {pct(mine['transcript'], .5):6.0f} ms")

    out = Path(args.out) if args.out else folder / "speech_latency.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== Time to first speech (ms) ===")
    print(f"  {'config':<11} {'p50':>6} {'p95':>6} {'mean':>6}   {'asr':>5} {'llm 1st token':>14} {'1st sentence tts':>17}")
    for cfg in CONFIGS:
        mine = [r for r in rows if r["config"] == cfg]
        first = [r["first_speech_ms"] for r in mine]
        print(f"  {cfg:<11} {pct(first, .5):6.0f} {pct(first, .95):6.0f} {sum(first) / len(first):6.0f}   "
              f"{pct([r['asr_ms'] for r in mine], .5):5.0f} {pct([r['llm_first_token_ms'] for r in mine], .5):14.0f} "
              f"{pct([r['first_tts_ms'] for r in mine], .5):17.0f}")
    print(f"\nPer-run rows written to {out}")


if __name__ == "__main__":
    main()
