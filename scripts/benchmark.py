#!/usr/bin/env python3
"""Replay sample audio against the gateway and aggregate latency.

Hits POST /chat for each WAV in samples/ (or a given file) N times, discards a
warmup run, and prints p50/p95 TTFT and tokens/sec — the numbers that decide
whether the audio-native path is fast enough for robot conversation.

Usage:
    python scripts/benchmark.py --iterations 20
    python scripts/benchmark.py --host http://localhost:8000 --file samples/hello.wav
"""
from __future__ import annotations

import argparse
import glob
import statistics
import sys
import time
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


def run(host: str, files: list[str], iterations: int, instruction: str) -> None:
    sid = f"bench-{int(time.time())}"
    ttft, tps, total = [], [], []
    print(f"Benchmarking {host}  |  {len(files)} clip(s) x {iterations} iters (1 warmup discarded)\n")

    for it in range(iterations + 1):
        for f in files:
            with open(f, "rb") as fh:
                t0 = time.perf_counter()
                r = requests.post(
                    f"{host}/chat",
                    files={"audio": (Path(f).name, fh, "audio/wav")},
                    data={"session_id": sid, "instruction": instruction},
                    timeout=300,
                )
            wall = (time.perf_counter() - t0) * 1000
            r.raise_for_status()
            m = r.json()["metrics"]
            tag = "warmup" if it == 0 else f"iter {it}"
            print(f"  [{tag:>7}] {Path(f).name:<20} ttft={m['ttft_ms']:7.0f}ms  "
                  f"tok/s={m['tokens_per_sec']:6.1f}  total={m['total_ms']:7.0f}ms  wall={wall:7.0f}ms")
            if it > 0:
                ttft.append(m["ttft_ms"]); tps.append(m["tokens_per_sec"]); total.append(m["total_ms"])
        # reset history between iterations to keep prefill comparable
        requests.post(f"{host}/reset", data={"session_id": sid}, timeout=30)

    print("\n=== Summary ===")
    print(f"  TTFT      p50={pct(ttft,.5):7.0f}ms   p95={pct(ttft,.95):7.0f}ms")
    print(f"  tokens/s  p50={pct(tps,.5):7.1f}     p95={pct(tps,.95):7.1f}")
    print(f"  total     p50={pct(total,.5):7.0f}ms   p95={pct(total,.95):7.0f}ms")
    print(f"  samples   n={len(ttft)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost:8000")
    ap.add_argument("--file", help="single WAV; default = all of samples/*.wav")
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--instruction", default="Answer briefly.")
    args = ap.parse_args()

    files = [args.file] if args.file else sorted(glob.glob("samples/*.wav"))
    if not files:
        sys.exit("No audio found. Put 16 kHz mono WAVs in samples/ or pass --file.")

    health = requests.get(f"{args.host}/health", timeout=10).json()
    print(f"Backend: {health['backend']}  quant: {health['quant_mode']}  "
          f"gpu: {health.get('gpu', {}).get('device_name', 'n/a')}\n")
    run(args.host, files, args.iterations, args.instruction)


if __name__ == "__main__":
    main()
