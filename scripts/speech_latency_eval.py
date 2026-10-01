#!/usr/bin/env python3
"""Time to first speech: Gemma hearing the audio vs. reading a Parakeet transcript.

Replays the clips recorded in the console's ASR eval mode (samples/asr/) through
POST /converse in both pipeline configurations and compares how long the gateway
takes to produce the first spoken sentence:

  audio                 speech -> Gemma (audio-native) -> TTS
  transcript-parakeet   speech -> Parakeet -> Gemma (text) -> TTS
  transcript-gemma      speech -> Gemma transcription -> Gemma (text) -> TTS

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
    python scripts/speech_latency_eval.py --asr-engines parakeet     # cascade with Parakeet only
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


def pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    k = (len(values) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def converse(host: str, wav: Path, config: str, engine: str) -> dict:
    """Run one clip through /converse. ``config`` is "audio" or "transcript-<asr engine>"."""
    llm_input, _, asr_engine = config.partition("-")
    data = {"llm_input": llm_input, "transcribe": "false", "engine": engine}
    if asr_engine:
        data["asr_engine"] = asr_engine
    with open(wav, "rb") as fh:
        r = requests.post(f"{host}/converse", files={"audio": (wav.name, fh, "audio/wav")}, data=data, timeout=300)
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
    # Make sure the gateway really ran what this configuration claims.
    ran = next((c.get("engine") for c in metrics.get("components", []) if c["name"] == "asr"), None)
    if asr_engine and ran != asr_engine:
        raise RuntimeError(f"{asr_engine} did not run (got {ran or 'no transcript'})")
    if llm_input == "transcript" and metrics.get("llm_input") != "transcript":
        raise RuntimeError("empty transcript; Gemma heard the audio instead")
    asr = metrics.get("asr_ms") or 0.0
    return {"first_speech_ms": metrics["time_to_first_audio_ms"], "asr_ms": asr,
            "llm_first_token_ms": metrics["ttft_ms"] - asr,  # ttft is measured from the start of the request
            "first_tts_ms": metrics["tts_segments"][0]["client_ms"], "reply": reply}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="http://localhost:8000")
    ap.add_argument("--dir", default="samples/asr", help="folder with manifest.json and the WAVs")
    ap.add_argument("--engine", default="", help="TTS engine (default: the gateway's)")
    ap.add_argument("--asr-engines", nargs="+", metavar="ENGINE",
                    help="transcript engines for the cascade: parakeet and/or gemma (default: every loaded one)")
    ap.add_argument("--order", choices=["grouped", "interleaved"], default="grouped",
                    help="grouped: every clip through one configuration, then the next (default); "
                         "interleaved: each clip through all configurations before moving on")
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

    loaded = [e["name"] for e in requests.get(f"{args.host}/asr/engines", timeout=10).json()["engines"] if e["available"]]
    engines = args.asr_engines or [e for e in ("parakeet", "gemma") if e in loaded]
    missing = [e for e in engines if e not in loaded]
    if missing:
        sys.exit(f"ASR engine(s) not loaded on the gateway: {', '.join(missing)} (loaded: {', '.join(loaded)})")
    CONFIGS = ["audio"] + [f"transcript-{e}" for e in engines]

    print(f"{len(clips)} clips x {args.runs} run(s) x {len(CONFIGS)} configurations: {', '.join(CONFIGS)}\n")
    for cfg in CONFIGS:  # warm-up, discarded
        converse(args.host, folder / clips[0][1]["file"], cfg, args.engine)

    rows: list[dict] = []

    def one(cfg: str, cid: str, clip: dict, run: int) -> None:
        res = converse(args.host, folder / clip["file"], cfg, args.engine)
        rows.append({"config": cfg, "clip": cid, "run": run + 1, "audio_seconds": clip["audio_seconds"], **res})

    if args.order == "grouped":
        # One configuration at a time, like a live session that stays in one mode.
        for cfg in CONFIGS:
            for cid, clip in clips:
                for run in range(args.runs):
                    one(cfg, cid, clip, run)
            mine = [r["first_speech_ms"] for r in rows if r["config"] == cfg]
            print(f"  {cfg:<20} done: p50 {pct(mine, .5):5.0f} ms over {len(mine)} runs")
    else:
        for i, (cid, clip) in enumerate(clips):
            for run in range(args.runs):
                # Rotate which configuration goes first so none always follows a warm one.
                k = (i + run) % len(CONFIGS)
                for cfg in CONFIGS[k:] + CONFIGS[:k]:
                    one(cfg, cid, clip, run)
            mine = {cfg: [r["first_speech_ms"] for r in rows if r["clip"] == cid and r["config"] == cfg] for cfg in CONFIGS}
            print(f"  {cid:<8} " + "   ".join(f"{cfg} {pct(mine[cfg], .5):5.0f} ms" for cfg in CONFIGS))

    out = Path(args.out) if args.out else folder / "speech_latency.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== Time to first speech (ms) ===")
    print(f"  {'config':<20} {'p50':>6} {'p95':>6} {'mean':>6}   {'asr':>5} {'llm 1st token':>14} {'1st sentence tts':>17}")
    for cfg in CONFIGS:
        mine = [r for r in rows if r["config"] == cfg]
        first = [r["first_speech_ms"] for r in mine]
        print(f"  {cfg:<20} {pct(first, .5):6.0f} {pct(first, .95):6.0f} {sum(first) / len(first):6.0f}   "
              f"{pct([r['asr_ms'] for r in mine], .5):5.0f} {pct([r['llm_first_token_ms'] for r in mine], .5):14.0f} "
              f"{pct([r['first_tts_ms'] for r in mine], .5):17.0f}")
    print(f"\nPer-run rows written to {out}")


if __name__ == "__main__":
    main()
