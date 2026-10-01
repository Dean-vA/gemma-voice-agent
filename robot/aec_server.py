#!/usr/bin/env python3
"""Echo-cancellation sidecar: runs aec.EchoCanceller for a client on a socket.

The robot's client image is an older Ubuntu whose GStreamer carries the old,
delay-fragile WebRTC canceller (or none at all). This small service runs in its
own container with a current GStreamer (Dockerfile.aec), so the client gets the
modern canceller without its image being rebuilt. Wire format: see aec.py.

    python3 aec_server.py            # listens on 127.0.0.1:5005
    AEC_BIND=0.0.0.0:5005 python3 aec_server.py
"""
from __future__ import annotations

import json
import os
import socket
import struct
import threading

import numpy as np

import aec


def _recv_exact(sock, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("client closed")
        buf += chunk
    return buf


def _serve(conn: socket.socket, peer) -> None:
    canceller = None
    try:
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        line = b""
        while not line.endswith(b"\n"):
            line += _recv_exact(conn, 1)
        cfg = json.loads(line)
        try:
            canceller = aec.EchoCanceller(float(cfg.get("delay_ms", 0.0)), bool(cfg.get("noise_suppression", True)),
                                          bool(cfg.get("gain_control", True)))
        except Exception as e:
            conn.sendall((json.dumps({"ok": False, "error": str(e)}) + "\n").encode())
            return
        conn.sendall((json.dumps({"ok": True, "legacy": canceller.legacy}) + "\n").encode())
        print(f"[aec] session from {peer} (legacy={canceller.legacy})", flush=True)
        while True:
            kind, n = struct.unpack("<cI", _recv_exact(conn, 5))
            payload = _recv_exact(conn, n) if n else b""
            if kind == b"M":
                out = canceller.process(np.frombuffer(payload, dtype="<i2"))
                data = out.astype("<i2").tobytes()
                conn.sendall(b"O" + struct.pack("<I", len(data)) + data)
            elif kind == b"F":
                at = struct.unpack("<q", payload[:8])[0]
                canceller.feed_far(payload[8:], None if at < 0 else at)
            elif kind == b"X":
                canceller.flush_far()
            elif kind == b"D":
                canceller.delay_ms = struct.unpack("<d", payload)[0]
    except (ConnectionError, OSError):
        pass
    finally:
        if canceller is not None:
            canceller.close()
        conn.close()
        print(f"[aec] session from {peer} ended", flush=True)


def main() -> None:
    host, _, port = os.environ.get("AEC_BIND", "127.0.0.1:5005").rpartition(":")
    if not aec.EchoCanceller.available():
        raise SystemExit("GStreamer webrtcdsp unavailable: " + aec.EchoCanceller.unavailable_reason())
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host or "127.0.0.1", int(port)))
    srv.listen(4)
    probe = aec.EchoCanceller()
    print(f"[aec] listening on {host or '127.0.0.1'}:{port} "
          f"({'legacy canceller: needs delay calibration' if probe.legacy else 'AEC3'})", flush=True)
    probe.close()
    while True:
        conn, peer = srv.accept()
        threading.Thread(target=_serve, args=(conn, peer), daemon=True).start()


if __name__ == "__main__":
    main()
