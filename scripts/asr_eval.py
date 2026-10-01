#!/usr/bin/env python3
"""Compare the gateway's transcription engines (Gemma vs Parakeet) on your own voice.

Reads the clips recorded on the ASR eval page (http://localhost:8000/web/asr-eval.html),
which land in samples/asr/ with a manifest of reference texts, sends each one to
POST /transcribe for every engine, and reports:

  quality  word error rate (case, punctuation and digits-vs-words ignored)
  latency  p50 / p95 per clip, and the real-time factor (latency / audio length)

Each clip is transcribed --runs times per engine; one extra warm-up pass is
discarded. Per-run rows go to samples/asr/results.csv.

Usage:
    pip install requests
    python scripts/asr_eval.py
    python scripts/asr_eval.py --host http://localhost:8000 --runs 5 --engines gemma parakeet
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.wer import word_errors  # noqa: E402  (pure Python, no gateway deps)


def pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    k = (len(values) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def transcribe(host: str, wav: Path, engine: str) -> dict:
    with open(wav, "rb") as fh:
        r = requests.post(f"{host}/transcribe", files={"audio": (wav.name, fh, "audio/wav")},
                          data={"engine": engine}, timeout=300)
    r.raise_for_status()
    return r.json()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="http://localhost:8000")
    ap.add_argument("--dir", default="samples/asr", help="folder with manifest.json and the WAVs")
    ap.add_argument("--engines", nargs="+", help="default: every engine the gateway has loaded")
    ap.add_argument("--runs", type=int, default=3, help="timed runs per clip per engine")
    ap.add_argument("--out", help="CSV path (default: <dir>/results.csv)")
    args = ap.parse_args()

    folder = Path(args.dir)
    manifest_path = folder / "manifest.json"
    if not manifest_path.exists():
        sys.exit(f"No {manifest_path}. Record clips first at {args.host}/web/asr-eval.html")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    clips = [(cid, c) for cid, c in sorted(manifest.items()) if (folder / c["file"]).exists()]
    if not clips:
        sys.exit(f"{manifest_path} lists no clips that exist in {folder}")

    available = [e["name"] for e in requests.get(f"{args.host}/asr/engines", timeout=10).json()["engines"]
                 if e["available"]]
    engines = args.engines or available
    missing = [e for e in engines if e not in available]
    if missing:
        sys.exit(f"Engine(s) not loaded on the gateway: {', '.join(missing)} (loaded: {', '.join(available)})")

    total_audio = sum(c["audio_seconds"] for _, c in clips)
    print(f"{len(clips)} clips, {total_audio:.0f} s of audio | engines: {', '.join(engines)} | "
          f"{args.runs} timed run(s) each + 1 warm-up\n")

    rows: list[dict] = []
    for engine in engines:
        transcribe(args.host, folder / clips[0][1]["file"], engine)  # warm-up, discarded
        for cid, clip in clips:
            for run in range(args.runs):
                res = transcribe(args.host, folder / clip["file"], engine)
                score = word_errors(clip["text"], res["text"])
                rows.append({"engine": engine, "clip": cid, "run": run + 1, "ms": res["ms"],
                             "audio_seconds": res["audio_seconds"], **score,
                             "reference": clip["text"], "hypothesis": res["text"]})
            last = rows[-1]
            print(f"  {engine:<9} {cid:<8} wer={last['wer'] * 100:5.1f}%  {last['ms']:7.0f} ms  {last['hypothesis']}")
        print()

    out = Path(args.out) if args.out else folder / "results.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("=== Summary ===")
    print(f"  {'engine':<9} {'WER':>7} {'perfect':>9} {'p50 ms':>8} {'p95 ms':>8} {'RTF':>7}")
    for engine in engines:
        mine = [r for r in rows if r["engine"] == engine]
        first = [r for r in mine if r["run"] == 1]  # score each clip once; runs differ only if sampling does
        wer = sum(r["errors"] for r in first) / max(1, sum(r["ref_words"] for r in first))
        perfect = sum(r["errors"] == 0 for r in first)
        lat = [r["ms"] for r in mine]
        rtf = sum(r["ms"] for r in mine) / 1000.0 / max(1e-9, sum(r["audio_seconds"] for r in mine))
        print(f"  {engine:<9} {wer * 100:6.1f}% {perfect:>4}/{len(first):<4} {pct(lat, .5):8.0f} {pct(lat, .95):8.0f} {rtf:7.3f}")

    if len(engines) == 2:
        a, b = engines
        by = {(r["engine"], r["clip"]): r for r in rows if r["run"] == 1}
        differ = [(cid, by[a, cid], by[b, cid]) for cid, _ in clips if by[a, cid]["errors"] != by[b, cid]["errors"]]
        print(f"\n=== Clips where {a} and {b} differ in errors ({len(differ)}) ===")
        for cid, ra, rb in differ:
            print(f"  {cid}  ref: {ra['reference']}")
            print(f"    {a:<9} ({ra['errors']} err) {ra['hypothesis']}")
            print(f"    {b:<9} ({rb['errors']} err) {rb['hypothesis']}")
    print(f"\nPer-run rows written to {out}")


if __name__ == "__main__":
    main()
