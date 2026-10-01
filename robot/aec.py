"""Acoustic echo cancellation for the open-mic (WebSocket, barge-in) mode.

Same software stack as the Reachy Mini SDK's AEC fallback
(pollen-robotics/reachy_mini, media/audio_gstreamer.py): GStreamer's
``webrtcdsp`` (the WebRTC audio-processing module) paired with a
``webrtcechoprobe`` that carries the far-end reference, i.e. what the robot is
saying. Like Reachy, every ``webrtcdsp`` property is left at its default unless
overridden here.

One difference is forced by the G1: Reachy plays through a local sound card, so
its probe sits in the playback pipeline and GStreamer's clock aligns reference
and microphone. George's speaker is driven over DDS (AudioClient.PlayStream),
so there is no local playback to tap. Instead the player tells us what it sent
and when (``feed_far``), we lay that out on the microphone's own sample clock,
and push reference and mic frames through the two elements with matching
timestamps. ``delay_ms`` is where the reference is placed relative to the moment
the clip was sent.

How forgiving that placement is depends on the GStreamer version (measured in a
simulation, 40+ dB of echo removed when it works, none when it doesn't):

  GStreamer >= 1.24   WebRTC "AEC3", which finds the delay itself. The reference
                      only has to be *early*: anything from ~20 to ~400 ms ahead
                      of the real echo works, so delay_ms = 0 (at send time) is
                      right for any plausible speaker path. This is what Reachy
                      runs.
  GStreamer <  1.24   the older canceller (Ubuntu 20.04/22.04's packages). The
                      reference must lead the echo by 20-60 ms, no more and no
                      less, so the speaker delay has to be measured
                      (``estimate_delay_ms``) and tracked. Fragile on a speaker
                      path with jitter; ``legacy`` is True for it.

Where it runs: in this process if GStreamer + PyGObject are importable here,
otherwise in the ``aec`` sidecar container (aec_server.py, a recent Debian with
a current GStreamer) reached over a local socket. ``create()`` picks.
"""
from __future__ import annotations

import itertools
import json
import os
import socket
import struct
import threading

import numpy as np

RATE = 16000
_CAPS = f"audio/x-raw,format=S16LE,rate={RATE},channels=1,layout=interleaved"
_ids = itertools.count()

# Legacy canceller only: place the reference this far ahead of the measured speaker delay.
DELAY_MARGIN_MS = 40.0
MAX_DELAY_MS = 1000.0           # longest speaker delay we look for
MIN_DELAY_CONFIDENCE = 8.0      # correlation peak vs. background, below which a measurement is ignored

# The sidecar's address: "host:port". Empty = don't try a sidecar.
AEC_URL = os.environ.get("PRESENCE_AEC_URL", "127.0.0.1:5005")

_gst = None
_gst_error = None


def estimate_delay_ms(mic: np.ndarray, ref: np.ndarray) -> tuple:
    """How late ``ref`` (what was sent to the speaker) shows up in ``mic``.

    ``mic`` must start at the moment ``ref`` was sent. Returns
    ``(delay_ms, confidence)``; confidence is the correlation peak relative to
    the typical correlation, so speech from someone else lowers it rather than
    moving the answer."""
    mic = np.asarray(mic, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    max_lag = min(int(RATE * MAX_DELAY_MS / 1000), len(mic) - 1)
    if len(ref) < RATE // 10 or max_lag <= 0 or not ref.any() or not mic.any():
        return 0.0, 0.0
    n = 1 << int(np.ceil(np.log2(len(mic) + len(ref))))
    corr = np.fft.irfft(np.fft.rfft(mic, n) * np.conj(np.fft.rfft(ref, n)), n)[: max_lag + 1]
    mag = np.abs(corr)
    peak = int(np.argmax(mag))
    return peak * 1000.0 / RATE, float(mag[peak] / (np.median(mag) + 1e-9))


def _load_gst():
    """Import and initialise GStreamer once; remember why it failed if it did."""
    global _gst, _gst_error
    if _gst is not None or _gst_error is not None:
        return _gst
    try:
        import gi
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
        Gst.init(None)
        for element in ("webrtcdsp", "webrtcechoprobe"):
            if Gst.ElementFactory.find(element) is None:
                raise RuntimeError(f"GStreamer element '{element}' not found (install gstreamer1.0-plugins-bad)")
        _gst = Gst
    except Exception as e:  # ImportError, ValueError (no Gst typelib), RuntimeError
        _gst_error = f"{type(e).__name__}: {e}"
    return _gst


class EchoCanceller:
    """Removes the robot's own voice from 16 kHz mono int16 mic frames, in-process."""

    where = "in-process"

    @staticmethod
    def available() -> bool:
        return _load_gst() is not None

    @staticmethod
    def unavailable_reason() -> str:
        _load_gst()
        return _gst_error or ""

    def __init__(self, delay_ms: float = 0.0, noise_suppression: bool = True, gain_control: bool = True) -> None:
        Gst = _load_gst()
        if Gst is None:
            raise RuntimeError(f"echo cancellation unavailable: {_gst_error}")
        self._Gst = Gst
        self.delay_ms = float(delay_ms)
        self._lock = threading.Lock()
        self._mic_samples = 0          # mic samples processed so far = our clock
        self._far_pushed = 0           # reference pushed to the probe up to this sample
        self._clips = []               # scheduled playback: [start_sample, int16 array]
        self._out = np.zeros(0, dtype=np.int16)

        probe = f"g1_aec_probe_{next(_ids)}"
        self._pipeline = Gst.parse_launch(
            f"appsrc name=far format=time is-live=false caps={_CAPS} "
            f"! webrtcechoprobe name={probe} ! fakesink sync=false async=false "
            f"appsrc name=mic format=time is-live=false caps={_CAPS} "
            f"! webrtcdsp name=dsp probe={probe} "
            f"! appsink name=out sync=false async=false max-buffers=0 drop=false")
        dsp = self._pipeline.get_by_name("dsp")
        dsp.set_property("noise-suppression", bool(noise_suppression))
        dsp.set_property("gain-control", bool(gain_control))
        # Which canceller we got (see the module docstring): GStreamer moved
        # webrtcdsp to WebRTC's AEC3 in 1.24.
        self.legacy = tuple(Gst.version()[:2]) < (1, 24)
        self._far = self._pipeline.get_by_name("far")
        self._mic = self._pipeline.get_by_name("mic")
        self._sink = self._pipeline.get_by_name("out")
        self._pipeline.set_state(Gst.State.PLAYING)

    # ---- far end: what the robot is saying -----------------------------------
    def feed_far(self, pcm, at_sample=None) -> None:
        """Call at the moment a clip is handed to the speaker (16 kHz mono int16).

        ``at_sample`` is that moment on the mic sample clock; pass it when the
        caller keeps a finer clock than "frames processed so far" (mic audio
        arrives in bursts). Clips queue one after another, as the speaker plays
        them."""
        clip = np.frombuffer(pcm, dtype=np.int16) if isinstance(pcm, (bytes, bytearray)) else np.asarray(pcm, np.int16)
        if clip.size == 0:
            return
        with self._lock:
            now = self._mic_samples if at_sample is None else int(at_sample)
            start = now + int(self.delay_ms * RATE / 1000)
            if self._clips:
                last_start, last = self._clips[-1]
                start = max(start, last_start + len(last))
            self._clips.append([start, clip])

    def flush_far(self) -> None:
        """Playback was cut short (barge-in): drop everything not yet played."""
        with self._lock:
            now, kept = self._mic_samples, []
            for start, clip in self._clips:
                if start < now:
                    kept.append([start, clip[: max(0, now - start)]])
            self._clips = kept

    def _far_slice(self, start: int, n: int) -> np.ndarray:
        """Reference audio for mic samples [start, start+n); silence where nothing plays."""
        out = np.zeros(n, dtype=np.int16)
        for clip_start, clip in self._clips:
            lo, hi = max(start, clip_start), min(start + n, clip_start + len(clip))
            if hi > lo:
                out[lo - start: hi - start] = clip[lo - clip_start: hi - clip_start]
        self._clips = [c for c in self._clips if c[0] + len(c[1]) > start]   # forget what is past
        return out

    # ---- near end: the microphone --------------------------------------------
    def _push(self, src, samples: np.ndarray, start: int) -> None:
        Gst = self._Gst
        buf = Gst.Buffer.new_wrapped(samples.astype("<i2").tobytes())
        buf.pts = start * Gst.SECOND // RATE
        buf.duration = len(samples) * Gst.SECOND // RATE
        src.emit("push-buffer", buf)

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Echo-cancel one mic frame (int16). Returns a frame of the same length;
        the first few are silence while the pipeline fills."""
        frame = np.asarray(frame, dtype=np.int16)
        n = len(frame)
        with self._lock:
            start = self._mic_samples
            # Keep the reference one frame ahead of the mic so it is always in
            # the probe before the matching mic audio reaches the canceller.
            while self._far_pushed < start + 2 * n:
                self._push(self._far, self._far_slice(self._far_pushed, n), self._far_pushed)
                self._far_pushed += n
            self._push(self._mic, frame, start)
            self._mic_samples += n

        # Collect whatever the canceller has produced (it works in 10 ms blocks).
        Gst = self._Gst
        while len(self._out) < n:
            sample = self._sink.emit("try-pull-sample", 20 * Gst.MSECOND)
            if sample is None:
                break
            buf = sample.get_buffer()
            ok, info = buf.map(Gst.MapFlags.READ)
            if ok:
                self._out = np.concatenate([self._out, np.frombuffer(bytes(info.data), dtype="<i2")])
                buf.unmap(info)
        if len(self._out) < n:
            return np.zeros(n, dtype=np.int16)
        out, self._out = self._out[:n], self._out[n:]
        return out

    def close(self) -> None:
        try:
            self._pipeline.set_state(self._Gst.State.NULL)
        except Exception:
            pass


# ------------------------------- sidecar client --------------------------------
# Wire format (see aec_server.py): one JSON line each way to open, then messages
# of 1 type byte + uint32 length + payload.
#   'M' mic frame (int16)                      -> 'O' cleaned frame (int16)
#   'F' int64 at_sample (-1 = now) + clip      (no reply)
#   'X' flush the reference                    (no reply)
#   'D' float64 delay_ms                       (no reply)
def _recv_exact(sock, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("echo-cancellation sidecar closed the connection")
        buf += chunk
    return buf


class RemoteEchoCanceller:
    """The same canceller, running in the ``aec`` sidecar container."""

    where = "sidecar"

    def __init__(self, url: str, delay_ms: float = 0.0, noise_suppression: bool = True,
                 gain_control: bool = True) -> None:
        host, _, port = url.rpartition(":")
        self._sock = socket.create_connection((host or "127.0.0.1", int(port)), timeout=2.0)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._lock = threading.Lock()
        self._delay_ms = float(delay_ms)
        self._sock.sendall((json.dumps({"delay_ms": delay_ms, "noise_suppression": noise_suppression,
                                        "gain_control": gain_control}) + "\n").encode())
        line = b""
        while not line.endswith(b"\n"):
            line += _recv_exact(self._sock, 1)
        hello = json.loads(line)
        if not hello.get("ok"):
            raise RuntimeError(hello.get("error", "sidecar refused"))
        self.legacy = bool(hello.get("legacy"))

    def _send(self, kind: bytes, payload: bytes) -> None:
        self._sock.sendall(kind + struct.pack("<I", len(payload)) + payload)

    @property
    def delay_ms(self) -> float:
        return self._delay_ms

    @delay_ms.setter
    def delay_ms(self, value: float) -> None:
        if float(value) != self._delay_ms:
            self._delay_ms = float(value)
            with self._lock:
                self._send(b"D", struct.pack("<d", self._delay_ms))

    def feed_far(self, pcm, at_sample=None) -> None:
        clip = pcm if isinstance(pcm, (bytes, bytearray)) else np.asarray(pcm, "<i2").tobytes()
        with self._lock:
            self._send(b"F", struct.pack("<q", -1 if at_sample is None else int(at_sample)) + bytes(clip))

    def flush_far(self) -> None:
        with self._lock:
            self._send(b"X", b"")

    def process(self, frame: np.ndarray) -> np.ndarray:
        with self._lock:
            self._send(b"M", np.asarray(frame, "<i2").tobytes())
            kind, n = struct.unpack("<cI", _recv_exact(self._sock, 5))
            return np.frombuffer(_recv_exact(self._sock, n), dtype="<i2").astype(np.int16)

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass


def create(delay_ms: float = 0.0, noise_suppression: bool = True, gain_control: bool = True):
    """The best echo canceller on offer. Returns ``(canceller, "")`` or ``(None, why_not)``.

    In-process if this Python has a current GStreamer; otherwise the sidecar,
    if it is running; and as a last resort an old in-process GStreamer with its
    legacy canceller."""
    local, reasons = None, []
    if EchoCanceller.available():
        local = EchoCanceller(delay_ms, noise_suppression, gain_control)
        if not local.legacy:
            return local, ""
    else:
        reasons.append("in-process: " + EchoCanceller.unavailable_reason())
    if AEC_URL:
        try:
            remote = RemoteEchoCanceller(AEC_URL, delay_ms, noise_suppression, gain_control)
            if local is not None:
                if remote.legacy:           # no better than what we have here
                    remote.close()
                    return local, ""
                local.close()
            return remote, ""
        except (OSError, RuntimeError, ValueError) as e:
            reasons.append(f"sidecar at {AEC_URL}: {e}")
    if local is not None:
        return local, ""
    return None, "; ".join(reasons)
