#!/usr/bin/env python3
"""
In-process web control centre for the G1 presence-greeter.

Runs a Flask server in a daemon thread inside the client process (the camera is
a single-opener V4L2 handle owned by this process, so the UI must be in-process).
Everything is additive and guarded: if anything here fails, the greeter keeps
running. Enabled by PRESENCE_WEB=1.

Features
  - Prompt library: named presets (greet/converse/goodbye), applied live.
  - Live MJPEG camera feed with the CV detection overlay.
  - Streamed logs (a tee on stdout; docker logs still works).
  - Architecture diagram with live per-component latencies + the last
    image/audio/heard/reply flowing through the pipeline (gateway's real
    `metrics.components` split + client-measured hops).
  - Browser "Hold to Talk" mic: WebAudio -> 16 kHz WAV -> /api/utterance ->
    a real converse turn (works on the laptop where the native mic is blocked;
    doubles as a remote PTT on the robot when served over a secure context).

The presence/instrumentation code publishes into the singleton Hub via
`presence.HUB` (set by start_control_center); see presence.py `_pub`.
"""

import os
import io
import sys
import json
import time
import hmac
import wave
import base64
import threading
import collections

import cv2
import numpy as np
from flask import Flask, Response, request, jsonify

import usb_cam

PORT      = int(os.environ.get("PRESENCE_WEB_PORT", "8080"))
PASSWORD  = os.environ.get("PRESENCE_WEB_PASSWORD", "")
FEED_FPS  = float(os.environ.get("PRESENCE_WEB_FPS", "14"))
PROMPTS_PATH = os.environ.get("PRESENCE_PROMPTS_PATH", "prompts.json")
MAX_STREAM_CLIENTS = int(os.environ.get("PRESENCE_WEB_MAX_CLIENTS", "8"))


# ============================== Hub ==========================================
class Hub:
    """Thread-safe shared state between the greeter threads and the web server."""

    def __init__(self):
        self._lock = threading.Lock()
        # logs
        self.log_ring = collections.deque(maxlen=600)
        self._log_clients = []          # list of per-client deques (unhashable -> not a set)
        self._log_lock = threading.Lock()
        # video
        self._frame_jpeg = None
        self._frame_seq = 0
        self._frame_cond = threading.Condition()
        # presence + metrics
        self.presence = {"enabled": False, "greeted": False, "busy": False,
                         "face_frac": 0.0, "thr": 0.0, "nfaces": 0}
        self.current = None
        self.last_turn = None
        self.turns = collections.deque(maxlen=40)
        self.last_image_jpeg = None
        self.speaking = False
        self.vad = {"floor": None, "onset": None, "offset": None, "level": 0.0,
                    "voiced": False, "active": False, "phase": "idle", "samples": 0}
        self._stream_clients = 0

    # ---- logs ----
    def log(self, line):
        with self._log_lock:
            self.log_ring.append(line)
            for q in self._log_clients:
                try:
                    q.append(line)
                except Exception:
                    pass

    def add_log_client(self, q):
        with self._log_lock:
            self._log_clients.append(q)

    def remove_log_client(self, q):
        with self._log_lock:
            if q in self._log_clients:
                self._log_clients.remove(q)

    # ---- video ----
    def set_frame(self, jpeg):
        with self._frame_cond:
            self._frame_jpeg = jpeg
            self._frame_seq += 1
            self._frame_cond.notify_all()

    def wait_frame(self, last_seq, timeout=1.0):
        with self._frame_cond:
            if self._frame_seq == last_seq:
                self._frame_cond.wait(timeout)
            return self._frame_jpeg, self._frame_seq

    # ---- publish (called from presence via _pub) ----
    def publish(self, kind, data):
        try:
            if kind == "presence":
                with self._lock:
                    self.presence.update(data)
            elif kind == "turn":
                self._turn_event(dict(data))
            elif kind == "speaking":
                with self._lock:
                    self.speaking = bool(data.get("on"))
            elif kind == "vad":
                with self._lock:
                    self.vad.update(data)
        except Exception:
            pass

    def _turn_event(self, d):
        ev = d.pop("ev", None)
        with self._lock:
            if ev == "start":
                img = d.pop("image_jpeg", None)
                if img:
                    self.last_image_jpeg = img
                    d["image_bytes"] = len(img)
                self.current = {"hops": {}, "metrics": {}, "heard": "",
                                "reply": "", "t": time.time(), **d}
            elif ev in ("first_token", "first_audio"):
                if self.current is not None:
                    self.current["hops"][ev + "_ms"] = d.get("ms")
            elif ev == "heard":
                if self.current is not None:
                    self.current["heard"] = d.get("heard", "")
            elif ev == "reply":
                if self.current is not None:
                    self.current["reply"] = d.get("reply", "")
            elif ev == "metrics":
                if self.current is not None:
                    self.current["metrics"] = d.get("metrics") or {}
            elif ev == "done":
                cur = self.current or {"hops": {}}
                cur["metrics"] = d.get("metrics") or {}
                cur["heard"] = d.get("heard", "")
                cur["reply"] = d.get("reply", "")
                cur["prompt"] = d.get("prompt")
                cur["hops"]["total_ms"] = d.get("total_ms")
                self.turns.append(cur)
                self.last_turn = cur
                self.current = None

    def state_snapshot(self):
        with self._lock:
            src = self.current if self.current is not None else self.last_turn
            turn = None
            if src is not None:                     # detached copy (avoid mutation mid-serialize)
                turn = {**src, "hops": dict(src.get("hops", {})),
                        "metrics": src.get("metrics", {})}
            return {
                "presence": dict(self.presence),
                "turn": turn,
                "in_progress": self.current is not None,
                "speaking": self.speaking,
                "has_image": self.last_image_jpeg is not None,
                "vad": dict(self.vad),
                "frame_seq": self._frame_seq,
            }


# ============================ LogTee =========================================
class LogTee:
    """Mirror stdout writes into the Hub (and keep writing to the real stdout)."""

    def __init__(self, real, hub):
        self._real = real
        self._hub = hub
        self._buf = ""

    def write(self, s):
        try:
            self._real.write(s)
        except Exception:
            pass
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line:
                self._hub.log(line)
        return len(s)

    def flush(self):
        try:
            self._real.flush()
        except Exception:
            pass


# ========================== PromptStore ======================================
class PromptStore:
    """Named prompt presets persisted to prompts.json, mtime-cached."""

    def __init__(self, path, seed):
        self.path = path
        self._lock = threading.Lock()
        self._mtime = 0
        self._data = None
        self._seed = seed
        self._ensure()

    def _ensure(self):
        if not os.path.exists(self.path):
            default = {"active": "default", "presets": [
                {"name": "default", **self._seed}]}
            self._write(default)

    def _write(self, data):
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self.path)
        self._data = data
        try:
            self._mtime = os.path.getmtime(self.path)
        except OSError:
            self._mtime = 0

    def _load(self):
        try:
            mt = os.path.getmtime(self.path)
        except OSError:
            mt = 0
        if self._data is None or mt != self._mtime:
            try:
                with open(self.path, encoding="utf-8") as f:
                    self._data = json.load(f)
                self._mtime = mt
            except Exception:
                if self._data is None:
                    self._data = {"active": "default",
                                  "presets": [{"name": "default", **self._seed}]}
        return self._data

    def active(self):
        """Return the active preset's {greet, converse, goodbye} (mtime-cached)."""
        with self._lock:
            d = self._load()
            name = d.get("active")
            for p in d.get("presets", []):
                if p.get("name") == name:
                    return p
            return d.get("presets", [{}])[0] if d.get("presets") else dict(self._seed)

    def snapshot(self):
        with self._lock:
            return json.loads(json.dumps(self._load()))

    def upsert(self, preset):
        with self._lock:
            d = self._load()
            presets = d.setdefault("presets", [])
            for i, p in enumerate(presets):
                if p.get("name") == preset.get("name"):
                    presets[i] = preset
                    break
            else:
                presets.append(preset)
            self._write(d)
            return d

    def delete(self, name):
        with self._lock:
            d = self._load()
            d["presets"] = [p for p in d.get("presets", []) if p.get("name") != name]
            if d.get("active") == name and d["presets"]:
                d["active"] = d["presets"][0]["name"]
            self._write(d)
            return d

    def set_active(self, name):
        with self._lock:
            d = self._load()
            if any(p.get("name") == name for p in d.get("presets", [])):
                d["active"] = name
                self._write(d)
            return d


# =========================== Grabber =========================================
# Camera watchdog: a wedged USB cam can keep reporting isOpened()==True while
# read() returns nothing -> the feed "dies after a while". Reopen the handle
# after this many consecutive failures OR this long without a fresh frame.
CAM_FAIL_LIMIT      = int(os.environ.get("PRESENCE_CAM_FAIL_LIMIT", "20"))
CAM_STALE_SECS      = float(os.environ.get("PRESENCE_CAM_STALE_SECS", "3.0"))
CAM_REOPEN_COOLDOWN = float(os.environ.get("PRESENCE_CAM_REOPEN_COOLDOWN", "2.0"))


class Grabber(threading.Thread):
    """Sole camera reader: loops cap.read() at camera rate and keeps only the
    freshest frame. Everything else (detector, frame pump, turn snapshots) reads
    this snapshot instead of pulling from the camera, so the driver buffer never
    backlogs (that backlog is what makes the feed lag on Windows/MSMF).

    Self-heals: if reads stop yielding frames (wedged handle), it releases and
    reopens the capture so the feed recovers without a restart."""

    def __init__(self):
        super().__init__(daemon=True)
        self._latest = None
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def get(self):
        with self._lock:
            return self._latest

    def stop(self):
        self._stop.set()

    def run(self):
        fails = 0
        last_good = time.time()
        last_reopen = 0.0
        while not self._stop.is_set():
            frame = None
            with usb_cam._lock:
                cap = usb_cam._get_cap()
                if cap is not None:
                    ok, f = cap.read()       # blocks ~1/fps; returns the next frame
                    if ok and f is not None:
                        frame = f
            now = time.time()
            if frame is not None:
                fails = 0
                last_good = now
                with self._lock:
                    self._latest = frame
                continue
            # no frame this iteration -> watchdog (reopen a wedged-but-"open" cam)
            fails += 1
            if ((fails >= CAM_FAIL_LIMIT or (now - last_good) > CAM_STALE_SECS)
                    and (now - last_reopen) > CAM_REOPEN_COOLDOWN):
                print(f"[cam] feed stalled ({fails} fails, {now - last_good:.1f}s "
                      f"no frame) -> reopening camera")
                try:
                    usb_cam.reset()
                except Exception as e:
                    print(f"[cam] reopen failed: {e}")
                last_reopen = now
                fails = 0
                last_good = now              # grace window for the fresh handle
            time.sleep(0.05)


# =========================== FramePump =======================================
class FramePump(threading.Thread):
    """Produce annotated JPEG frames into the Hub at a low FPS. Reuses the
    detector's last_frame/last_faces under usb_cam._lock; light-grabs a preview
    when idle. Yields the camera to active turns (CONVERSATION_BUSY)."""

    def __init__(self, hub, detector, controller, busy, grabber):
        super().__init__(daemon=True)
        self.hub = hub
        self.detector = detector
        self.controller = controller
        self.busy = busy
        self.grabber = grabber
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        period = 1.0 / max(1.0, FEED_FPS)
        # import the shared overlay drawer lazily (presence may import after us)
        try:
            from presence import _draw_overlay
        except Exception:
            _draw_overlay = None
        while not self._stop.is_set():
            t0 = time.time()
            try:
                self._tick(_draw_overlay)
            except Exception:
                # never let the feed crash the process
                pass
            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)

    def _tick(self, draw):
        enabled = getattr(self.controller, "enabled", False)
        # Always render the FRESHEST frame from the grabber (low latency).
        frame = self.grabber.get()
        if frame is None:
            frame = getattr(self.detector, "last_frame", None)
        if frame is None:
            return
        # Overlay (face boxes + status line) ONLY when armed -- the controller is
        # detecting then (5 Hz) so its last_faces are fresh. When idle, show a
        # clean camera preview with no boxes/label.
        if enabled and draw is not None:
            ps = self.hub.presence
            faces = getattr(self.detector, "last_faces", None)
            img = draw(frame, faces, ps.get("face_frac", 0.0), ps)
        else:
            img = frame
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            self.hub.set_frame(jpg.tobytes())


# ============================ Flask app ======================================
def _build_app(hub, store, controller, busy):
    app = Flask(__name__)

    def _authed():
        a = request.authorization
        return a is not None and hmac.compare_digest(a.password or "", PASSWORD)

    @app.before_request
    def _auth():
        if not PASSWORD:          # empty password => open (handy for local dev)
            return
        if not _authed():
            return Response("auth required", 401,
                            {"WWW-Authenticate": 'Basic realm="G1 Control Centre"'})

    @app.route("/")
    def index():
        return Response(PAGE, mimetype="text/html")

    @app.route("/video.mjpg")
    def video():
        if hub._stream_clients >= MAX_STREAM_CLIENTS:
            return Response("too many clients", 503)
        def gen():
            hub._stream_clients += 1
            seq = -1
            try:
                while True:
                    jpeg, seq = hub.wait_frame(seq, timeout=2.0)
                    if jpeg is None:
                        time.sleep(0.1)
                        continue
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
                           b"Content-Length: " + str(len(jpeg)).encode() +
                           b"\r\n\r\n" + jpeg + b"\r\n")
            finally:
                hub._stream_clients -= 1
        return Response(gen(),
                        mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/last_image.jpg")
    def last_image():
        if hub.last_image_jpeg is None:
            return Response(b"", 404)
        return Response(hub.last_image_jpeg, mimetype="image/jpeg")

    @app.route("/logs")
    def logs():
        def gen():
            q = collections.deque(maxlen=500)
            for line in list(hub.log_ring):
                yield _sse(line)
            hub.add_log_client(q)
            try:
                while True:
                    if q:
                        yield _sse(q.popleft())
                    else:
                        time.sleep(0.2)
                        yield ": keepalive\n\n"
            finally:
                hub.remove_log_client(q)
        return Response(gen(), mimetype="text/event-stream")

    @app.route("/state")
    def state():
        def gen():
            while True:
                snap = hub.state_snapshot()
                yield _sse(json.dumps(snap))
                time.sleep(0.25)
        return Response(gen(), mimetype="text/event-stream")

    @app.route("/api/prompts", methods=["GET", "POST"])
    def api_prompts():
        if request.method == "POST":
            p = request.get_json(force=True)
            for k in ("name", "greet", "converse", "goodbye"):
                p.setdefault(k, "")
            return jsonify(store.upsert(p))
        return jsonify(store.snapshot())

    @app.route("/api/prompts/<name>", methods=["DELETE"])
    def api_prompt_delete(name):
        return jsonify(store.delete(name))

    @app.route("/api/prompts/active", methods=["POST"])
    def api_prompt_active():
        name = (request.get_json(force=True) or {}).get("name", "")
        return jsonify(store.set_active(name))

    @app.route("/api/control", methods=["POST"])
    def api_control():
        action = (request.get_json(force=True) or {}).get("action", "")
        try:
            if action == "arm":
                controller.enable()
            elif action == "disarm":
                controller.disable()
            elif action == "toggle":
                controller.toggle()
            else:
                return jsonify({"ok": False, "error": "unknown action"}), 400
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500
        return jsonify({"ok": True, "enabled": getattr(controller, "enabled", None)})

    @app.route("/api/utterance", methods=["POST"])
    def api_utterance():
        wav = request.get_data() or b""
        if len(wav) < 100:
            return jsonify({"ok": False, "error": "no audio"}), 400
        if busy and busy.is_set():
            return jsonify({"ok": False, "error": "busy"}), 409
        try:
            heard, reply = controller.converse_once(wav)
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500
        return jsonify({"ok": True, "heard": heard, "reply": reply})

    # live-tunable presence/VAD thresholds (presence.py module globals, read at
    # use-time so changes take effect immediately).
    TUNABLES = {"prox": "PROXIMITY_FRAC", "arm": "ARM_SECS", "disarm": "DISARM_SECS",
                "silence_ms": "SILENCE_MS", "noise_mult": "NOISE_MULT",
                "continue_mult": "CONTINUE_MULT"}

    def _tune_snapshot():
        import presence
        out = {k: getattr(presence, a, None) for k, a in TUNABLES.items()}
        v = getattr(controller, "vad", None)
        out["noise_floor"] = getattr(v, "noise_floor", None)
        out["speech_thresh"] = getattr(v, "speech_thresh", None)
        out["vad_available"] = bool(getattr(v, "available", False))
        out["mic_source"] = getattr(v, "source", None)
        return out

    @app.route("/api/tune", methods=["GET", "POST"])
    def api_tune():
        import presence
        if request.method == "POST":
            body = request.get_json(force=True) or {}
            for k, val in body.items():
                if k in TUNABLES:
                    try:
                        setattr(presence, TUNABLES[k], float(val))
                    except Exception:
                        pass
            # recompute the live VAD thresholds from the current floor + mults
            v = getattr(controller, "vad", None)
            nf = getattr(v, "noise_floor", None)
            if v is not None and nf is not None:
                mf = getattr(v, "min_floor", presence.MIN_FLOOR)   # per mic source
                om = v.onset_mult() if hasattr(v, "onset_mult") else presence.NOISE_MULT
                v.speech_thresh = max(mf, nf * om)
                v.continue_thresh = max(mf * 0.8, nf * presence.CONTINUE_MULT)
        return jsonify(_tune_snapshot())

    # Gateway pipeline options: who transcribes, and whether Gemma answers from
    # the audio or from the transcript. presence.py globals, read per turn.
    def _gateway_snapshot():
        import presence
        out = {"transcribe": bool(presence.CONV_TRANSCRIBE),
               "asr_engine": presence.ASR_ENGINE, "llm_input": presence.LLM_INPUT,
               "engines": None, "gateway_default": None}
        try:  # which engines has the gateway loaded? (absent on older gateways)
            import urllib.request
            with urllib.request.urlopen(presence.GEMMA_URL + "/asr/engines", timeout=2.0) as r:
                info = json.loads(r.read().decode("utf-8"))
            out["engines"] = [e["name"] for e in info.get("engines", []) if e.get("available")]
            out["gateway_default"] = {"asr_engine": info.get("default"), "llm_input": info.get("llm_input")}
        except Exception:
            pass
        return out

    @app.route("/api/gateway", methods=["GET", "POST"])
    def api_gateway():
        import presence
        if request.method == "POST":
            body = request.get_json(force=True) or {}
            if "transcribe" in body:
                presence.CONV_TRANSCRIBE = bool(body["transcribe"])
            if body.get("asr_engine") in ("", "gemma", "parakeet"):
                presence.ASR_ENGINE = body["asr_engine"]
            if body.get("llm_input") in ("", "audio", "transcript"):
                presence.LLM_INPUT = body["llm_input"]
            print(f"[web] gateway options: transcribe={presence.CONV_TRANSCRIBE} "
                  f"asr_engine={presence.ASR_ENGINE or '(gateway default)'} "
                  f"llm_input={presence.LLM_INPUT or '(gateway default)'}")
        return jsonify(_gateway_snapshot())

    # Conversation link: HTTP (utterance per request, on-robot VAD) or the
    # streamed WebSocket mode with gateway-side VAD / Smart Turn / barge-in and
    # echo cancellation. presence.STREAM, read continuously by a live session.
    STREAM_BOOLS = ("barge_in", "aec", "aec_noise_suppression", "aec_gain_control")
    # gateway turn-detection options this UI can override (name -> is it a bool)
    STREAM_VAD = {"vad_threshold": False, "vad_min_speech_ms": False, "vad_min_silence_ms": False,
                  "vad_speech_pad_ms": False, "smart_turn": True, "smart_turn_threshold": False,
                  "reopen_ms": False, "smart_turn_max_wait_ms": False, "smart_turn_incomplete_delay_ms": False}

    def _stream_snapshot():
        import presence
        import aec
        sess = presence.STREAM_SESSION
        status = dict(sess.status) if sess is not None else dict(presence.STREAM_LAST_STATUS)
        out = {"options": {**presence.STREAM, "vad": dict(presence.STREAM["vad"])},
               "live": sess is not None, "status": status,
               "echo_delay_ms": status.get("echo_delay_ms", presence._ECHO_DELAY_MS)}
        # can echo cancellation run at all? (in this process, or the sidecar)
        if aec.EchoCanceller.available():
            out["aec_backend"] = "in-process"
        else:
            out["aec_backend"] = None
            try:
                host, _, port = aec.AEC_URL.rpartition(":")
                import socket as _s
                _s.create_connection((host or "127.0.0.1", int(port)), timeout=0.3).close()
                out["aec_backend"] = "sidecar"
            except Exception:
                pass
        return out

    @app.route("/api/stream", methods=["GET", "POST"])
    def api_stream():
        import presence
        if request.method == "POST":
            body = request.get_json(force=True) or {}
            o = presence.STREAM
            if body.get("transport") in ("http", "ws"):
                o["transport"] = body["transport"]
            for k in STREAM_BOOLS:
                if k in body:
                    o[k] = bool(body[k])
            for k in ("aec_delay_ms", "mic_gain"):
                if k in body:
                    v = body[k]
                    try:
                        o[k] = "auto" if v in (None, "", "auto") else max(0.0, float(v))
                    except (TypeError, ValueError):
                        pass
            if isinstance(body.get("vad"), dict):
                vad = dict(o["vad"])
                for k, v in body["vad"].items():
                    if k not in STREAM_VAD:
                        continue
                    if v is None:
                        vad.pop(k, None)              # back to the gateway default
                    else:
                        try:
                            vad[k] = bool(v) if STREAM_VAD[k] else float(v)
                        except (TypeError, ValueError):
                            pass
                o["vad"] = vad                        # replaced whole, so a live session sees one consistent dict
            if body.get("reset_vad"):
                o["vad"] = {}
            print(f"[web] conversation link: transport={o['transport']} barge_in={o['barge_in']} "
                  f"aec={o['aec']} vad_overrides={o['vad']}")
        return jsonify(_stream_snapshot())

    @app.route("/api/conversation")
    def api_conversation():
        """The current (or most recent) visitor's session, turn by turn, with the
        prompt Gemma answered from when the gateway returned it."""
        with hub._lock:
            turns = list(hub.turns)
            live = dict(hub.current) if hub.current is not None else None
        session = (live or (turns[-1] if turns else {})).get("session")
        out = [{"phase": t.get("phase"), "t": t.get("t"), "heard": t.get("heard", ""),
                "reply": t.get("reply", ""), "prompt": t.get("prompt"),
                "total_ms": (t.get("hops") or {}).get("total_ms")}
               for t in turns if session and t.get("session") == session]
        if live and live.get("session") == session:
            out.append({"phase": live.get("phase"), "t": live.get("t"), "heard": live.get("heard", ""),
                        "reply": live.get("reply", ""), "prompt": None, "in_progress": True})
        return jsonify({"session": session, "turns": out})

    @app.route("/api/speech_gain", methods=["GET", "POST"])
    def api_speech_gain():
        """Software gain on George's replies (applies from the next sentence)."""
        if not hasattr(controller, "gain"):
            return jsonify({"ok": False, "error": "no presence controller"})
        if request.method == "POST":
            try:
                g = float((request.get_json(force=True) or {}).get("gain"))
            except (TypeError, ValueError):
                return jsonify({"ok": False, "error": "gain must be a number"}), 400
            if not 0.25 <= g <= 4.0:
                return jsonify({"ok": False, "error": "gain must be 0.25-4"}), 400
            controller.gain = g
            print(f"[web] speech gain -> x{g:g}")
        return jsonify({"ok": True, "gain": controller.gain})

    @app.route("/api/volume", methods=["GET", "POST"])
    def api_volume():
        audio = getattr(controller, "audio", None)
        if audio is None or not hasattr(audio, "SetVolume"):
            return jsonify({"ok": False, "error": "no robot speaker on this host"})
        try:
            if request.method == "POST":
                v = int((request.get_json(force=True) or {}).get("volume", -1))
                if not 0 <= v <= 100:
                    return jsonify({"ok": False, "error": "volume must be 0-100"}), 400
                code = audio.SetVolume(v)
                code = code[0] if isinstance(code, tuple) else code
                if code not in (0, None):
                    return jsonify({"ok": False, "error": f"SetVolume failed ({code})"}), 502
                print(f"[web] speaker volume -> {v}")
            if not hasattr(audio, "GetVolume"):           # laptop stub
                return jsonify({"ok": True, "volume": v if request.method == "POST" else None})
            code, data = audio.GetVolume()
            if code != 0:
                return jsonify({"ok": False, "error": f"GetVolume failed ({code})"}), 502
            return jsonify({"ok": True, "volume": int((data or {}).get("volume", 0))})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/mic", methods=["POST"])
    def api_mic():
        v = getattr(controller, "vad", None)
        if v is None or not hasattr(v, "set_source"):
            return jsonify({"ok": False, "error": "no VAD mic on this host"}), 400
        ok, msg = v.set_source((request.get_json(force=True) or {}).get("source", ""))
        if not ok:
            return jsonify({"ok": False, "error": msg}), 400
        return jsonify({"ok": True, "mic_source": v.source})

    @app.route("/api/calibrate", methods=["POST"])
    def api_calibrate():
        v = getattr(controller, "vad", None)
        busy = getattr(controller, "busy", None)
        if busy is not None and busy.is_set() and getattr(v, "source", "usb") != "array":
            # a conversation owns the mic: opening a 2nd stream on it fails /
            # clobbers the live one. Let the presence loop do it when idle.
            controller._recalib = True
            return jsonify({"ok": False, "error": "conversation in progress; will calibrate when idle"})
        if v is not None and getattr(v, "available", False):
            res = v.calibrate()
            if res:
                return jsonify({"ok": True, "noise_floor": res[0], "speech_thresh": res[1]})
            return jsonify({"ok": False, "error": "calibration returned nothing"})
        # no local mic (e.g. browser-mic dev mode): queue for the next arm
        try:
            controller._recalib = True
        except Exception:
            pass
        return jsonify({"ok": False, "error": "VAD mic unavailable on this host; queued for next arm"})

    return app


def _sse(data):
    return "data: " + data.replace("\n", "\\n") + "\n\n"


# ========================== entry point ======================================
def start_control_center(detector, controller, busy):
    """Install the log tee, seed prompts, start the frame pump + Flask server.
    Returns the Hub (also set as presence.HUB so instrumentation can publish)."""
    import presence
    hub = Hub()
    presence.HUB = hub

    seed = {"greet": getattr(presence, "GREET_INSTRUCTION", ""),
            "converse": getattr(presence, "CONVERSE_INSTRUCTION", ""),
            "goodbye": getattr(presence, "GOODBYE_INSTRUCTION", "")}
    store = PromptStore(PROMPTS_PATH, seed)
    presence.PROMPTS = store

    sys.stdout = LogTee(sys.stdout, hub)

    # Single continuous camera reader; route the detector + turn snapshots through
    # it (presence.FRAME_SOURCE) so nothing pulls from the camera directly and the
    # driver buffer can't backlog (the cause of feed lag).
    grabber = Grabber()
    presence.FRAME_SOURCE = grabber.get
    grabber.start()

    FramePump(hub, detector, controller, busy, grabber).start()

    app = _build_app(hub, store, controller, busy)

    def _serve():
        try:
            app.run(host="0.0.0.0", port=PORT, threaded=True,
                    use_reloader=False, debug=False)
        except Exception as e:
            print(f"[web] server stopped: {e}")

    threading.Thread(target=_serve, daemon=True).start()
    print(f"[web] control centre on :{PORT} (user: any, password set via "
          f"PRESENCE_WEB_PASSWORD)")
    return hub


# ============================== UI ===========================================
PAGE = r"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>G1 Control Centre</title>
<style>
:root{
  --acc:#3ddc97; --acc2:#6ea8ff; --mag:#e368c8; --warn:#f0b94a; --bad:#ff6b81;
  --fg:#e9eef5; --mut:#8492a8; --line:rgba(255,255,255,.08);
  --glass:rgba(20,26,37,.66); --glass2:rgba(13,18,28,.75);
}
*{box-sizing:border-box}
html,body{margin:0;height:100%}
body{
  font:14px/1.45 ui-sans-serif,system-ui,Segoe UI,Roboto,sans-serif;color:var(--fg);
  background:
    radial-gradient(1200px 700px at 80% -10%,rgba(110,168,255,.10),transparent 60%),
    radial-gradient(900px 600px at -10% 110%,rgba(227,104,200,.08),transparent 55%),
    linear-gradient(160deg,#080b11,#0c111b 60%,#0a0e16);
  background-attachment:fixed;
}
a{color:var(--acc2)}
header{
  display:flex;align-items:center;gap:16px;padding:14px 22px;
  border-bottom:1px solid var(--line);backdrop-filter:blur(8px);
  position:sticky;top:0;z-index:5;background:rgba(8,11,17,.55)
}
.brand{font-weight:700;font-size:16px;letter-spacing:.3px;
  background:linear-gradient(90deg,#fff,#9fc6ff 60%,var(--acc));-webkit-background-clip:text;background-clip:text;color:transparent}
.chip{display:inline-flex;align-items:center;gap:7px;padding:5px 11px;border-radius:999px;
  border:1px solid var(--line);background:var(--glass);font-size:12px;color:var(--mut)}
.chip b{color:var(--fg);font-weight:600}
.dot{width:8px;height:8px;border-radius:50%;background:var(--mut);box-shadow:0 0 0 0 rgba(61,220,151,.5)}
.dot.live{background:var(--acc);animation:pulse 1.8s infinite}
.dot.armed{background:var(--acc)}
.dot.bad{background:var(--bad)}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(61,220,151,.45)}70%{box-shadow:0 0 0 7px rgba(61,220,151,0)}100%{box-shadow:0 0 0 0 rgba(61,220,151,0)}}
.gauge{width:90px;height:6px;border-radius:4px;background:rgba(255,255,255,.08);overflow:hidden}
.gauge i{display:block;height:100%;width:0;background:linear-gradient(90deg,var(--acc2),var(--acc));transition:width .25s}
.grid{display:grid;grid-template-columns:minmax(300px,.72fr) minmax(660px,1.9fr);
  gap:16px;padding:16px;align-items:start}
.col{display:flex;flex-direction:column;gap:16px;min-width:0}
.card{background:var(--glass);border:1px solid var(--line);border-radius:16px;
  padding:14px 16px;backdrop-filter:blur(10px);box-shadow:0 10px 30px rgba(0,0,0,.25)}
.card h2{margin:0 0 12px;font-size:11px;letter-spacing:.13em;text-transform:uppercase;color:var(--mut);font-weight:700}
img#feed{width:100%;max-height:300px;object-fit:cover;border-radius:12px;display:block;background:#000;border:1px solid var(--line)}
.controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:12px}
button{font:inherit;font-size:13px;color:var(--fg);background:rgba(255,255,255,.04);
  border:1px solid var(--line);border-radius:10px;padding:9px 14px;cursor:pointer;transition:.15s}
button:hover{border-color:var(--acc);background:rgba(61,220,151,.08)}
button.primary.on{background:linear-gradient(90deg,var(--acc),#2bb87c);color:#04221a;border-color:transparent;box-shadow:0 0 18px rgba(61,220,151,.35)}
#talk{margin-left:auto;border-color:rgba(227,104,200,.5);background:rgba(227,104,200,.10)}
#talk.rec{background:var(--mag);color:#1a0617;border-color:transparent;box-shadow:0 0 22px rgba(227,104,200,.5);animation:recpulse 1s infinite}
@keyframes recpulse{50%{box-shadow:0 0 30px rgba(227,104,200,.85)}}
.hint{font-size:12px;color:var(--mut)}
.wave{width:100%;height:46px;display:block;margin-top:6px;background:rgba(5,8,13,.55);border:1px solid var(--line);border-radius:8px}
.vmeter{margin-top:6px}
.vmtrack{position:relative;height:28px;background:rgba(5,8,13,.55);border:1px solid var(--line);border-radius:8px;overflow:hidden}
.vmfill{position:absolute;left:0;top:0;bottom:0;width:0;background:linear-gradient(90deg,#6ea8ff,#3ddc97);transition:width .12s linear}
.vmfill.voiced{background:linear-gradient(90deg,#3ddc97,#2bb87c)}
.vmmark{position:absolute;top:0;bottom:0;width:2px;opacity:.95}
.vmmark.floor{background:rgba(255,255,255,.55)}
.vmmark.onset{background:#3ddc97;box-shadow:0 0 4px #3ddc97}
.vmmark.offset{background:var(--warn)}
.vmlabel{font:600 11px ui-monospace,Consolas,monospace;color:var(--mut);margin-top:6px;letter-spacing:.02em}
/* diagram (SVG, built in JS via createElementNS) */
.diagram{width:100%;height:auto;display:block;overflow:visible}
.node{cursor:help}
.nbox{fill:url(#ng);stroke:rgba(255,255,255,.10);stroke-width:1.4;transition:.2s}
.nname{fill:var(--mut);font:600 12px ui-sans-serif;text-anchor:middle}
.nlat{fill:var(--fg);font:700 16px ui-monospace,Consolas,monospace;text-anchor:middle;transition:.2s}
.node.active .nbox{stroke:var(--acc);filter:url(#glow)}
.node.active .nlat{fill:var(--acc)}
.barbg{fill:rgba(255,255,255,.10)}
.barfg{fill:url(#bg2);transition:width .35s}
.gwbox{fill:rgba(110,168,255,.06);stroke:rgba(110,168,255,.35);stroke-dasharray:5 5}
.gwlabel{fill:var(--acc2);font:700 11px ui-sans-serif;letter-spacing:.12em}
.gwtotal{fill:var(--acc);font:700 14px ui-monospace,Consolas,monospace;cursor:help}
.flow{stroke:rgba(255,255,255,.18);stroke-width:2.5;fill:none;stroke-linecap:round}
.flowing .flow{stroke:var(--acc);stroke-dasharray:6 8;animation:dash .5s linear infinite}
@keyframes dash{to{stroke-dashoffset:-14}}
.node.talking .nbox{stroke:var(--acc);filter:url(#glow)}
.node.cut .nbox{stroke:var(--warn)}
#cv_turns{display:flex;flex-direction:column;gap:10px;margin-top:10px;max-height:560px;overflow:auto}
.cvt{border:1px solid var(--line);border-radius:12px;padding:9px 12px;background:var(--glass2)}
.cvt .ph{font:700 10px ui-sans-serif;letter-spacing:.08em;text-transform:uppercase;color:var(--acc2)}
.cvt .ln{font-size:13px;margin-top:4px;word-break:break-word}
.cvt .ln b{color:var(--mut);font-weight:600;margin-right:6px}
.cvt details{margin-top:6px}
.cvt summary{cursor:pointer;color:var(--mut);font-size:12px}
.cvmsg{font:12px/1.45 ui-monospace,Consolas,monospace;white-space:pre-wrap;word-break:break-word;margin-top:5px;padding:5px 8px;border-radius:8px;background:rgba(5,8,13,.55)}
.cvmsg .r{color:var(--acc2);font-weight:700;margin-right:6px}
.cvmsg.sys .r{color:var(--warn)} .cvmsg.asst .r{color:var(--acc)}
.flow.casc,.linklbl.casc{display:none}
.cascade .flow.casc{display:inline;stroke:#c792ea;stroke-dasharray:5 5}
.cascade .linklbl.casc{display:inline;fill:#c792ea}
.cascade .flow.a2l{display:none}
.flow.onlycasc{display:none}
.cascade .flow.onlycasc{display:inline}
.cascade .flow.notcasc{display:none}
.node.talking .nlat{fill:var(--acc)}
.eqbar{fill:var(--acc);opacity:0;transform-box:fill-box;transform-origin:center bottom}
.node.talking .eqbar{opacity:1;animation:eq .66s ease-in-out infinite}
.node.talking .eqbar:nth-of-type(2){animation-delay:.13s}
.node.talking .eqbar:nth-of-type(3){animation-delay:.26s}
.node.talking .eqbar:nth-of-type(4){animation-delay:.39s}
@keyframes eq{0%,100%{transform:scaleY(.28)}50%{transform:scaleY(1)}}
.flow.img{stroke:var(--acc2);stroke-dasharray:2 6;stroke-linecap:round}
.flowing .flow.img{stroke:var(--acc2)}
.link{stroke:rgba(233,238,245,.22);stroke-width:2;fill:none}
.link.arm{stroke:var(--warn);stroke-dasharray:4 4;opacity:.7}
.linklbl{fill:var(--mut);font:600 9px ui-sans-serif;letter-spacing:.08em}
.nsub{fill:var(--mut);font:600 9px ui-sans-serif;text-anchor:middle;text-transform:uppercase;letter-spacing:.07em}
/* in-diagram HEARD / REPLY boxes (foreignObject so the text wraps) */
.fonode{cursor:help}
.fobub{height:100%;box-sizing:border-box;border-radius:12px;border:1px solid var(--line);padding:7px 12px;overflow:hidden;display:flex;flex-direction:column;gap:3px;font-family:ui-sans-serif,system-ui,sans-serif}
.fobub.h{background:rgba(110,168,255,.10);border-color:rgba(110,168,255,.32)}
.fobub.r{background:rgba(61,220,151,.10);border-color:rgba(61,220,151,.32)}
.fobub .fowho{font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:var(--mut)}
.fotext{font-size:13px;color:var(--fg);line-height:1.32;overflow:hidden;word-break:break-word}
/* stat tiles */
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:2px 0 14px}
.tile{background:var(--glass2);border:1px solid var(--line);border-radius:10px;padding:9px 10px}
.tile .k{font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:var(--mut)}
.tile .v{font:700 17px ui-monospace,monospace;margin-top:2px}
/* dataflow */
.flowrow{display:flex;gap:14px;margin-top:14px;align-items:flex-start}
.flowrow img{width:132px;height:99px;object-fit:cover;border-radius:10px;border:1px solid var(--line);background:#000}
.bubbles{flex:1;min-width:0;display:flex;flex-direction:column;gap:8px}
.bub{border-radius:12px;padding:8px 11px;font-size:13px;border:1px solid var(--line);word-break:break-word}
.bub.h{background:rgba(110,168,255,.08)} .bub.r{background:rgba(61,220,151,.08)}
.bub .who{font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:var(--mut);margin-bottom:2px}
/* logs */
#log{height:230px;overflow:auto;border-radius:10px;background:rgba(5,8,13,.7);
  border:1px solid var(--line);padding:10px;font:12px/1.5 ui-monospace,Consolas,monospace;white-space:pre-wrap}
#log .g{color:var(--acc)} #log .w{color:var(--warn)} #log .b{color:var(--bad)} #log .m{color:var(--mut)}
/* prompts */
.prow{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:8px}
select,input,textarea{background:rgba(5,8,13,.7);color:var(--fg);border:1px solid var(--line);
  border-radius:9px;padding:8px 10px;font:13px ui-sans-serif}
textarea{width:100%;font:12px/1.45 ui-monospace,monospace;resize:vertical}
label{font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:var(--mut);display:block;margin:8px 0 3px}
.tag{font-size:11px;color:var(--acc)}
label.chk{display:inline-flex;align-items:center;gap:6px;text-transform:none;letter-spacing:0;font-size:12px;color:var(--fg);margin:0 10px 0 0;cursor:pointer}
label.chk input{padding:0;accent-color:var(--acc)}
.tune label{display:flex;justify-content:space-between;text-transform:none;letter-spacing:0;font-size:11px;color:var(--mut);margin:7px 0 1px}
.tune label span{color:var(--fg);font:700 12px ui-monospace,monospace}
.tune input[type=range]{width:100%;padding:0;accent-color:var(--acc);background:transparent;border:0;cursor:pointer}
@media(max-width:1080px){.grid{grid-template-columns:1fr}}
</style></head><body>
<header>
  <span class=brand>◢ G1 CONTROL CENTRE</span>
  <span class=chip><i class="dot" id=armdot></i><b id=armtxt>idle</b></span>
  <span class=chip>face <b id=facetxt>0.00</b><span class=gauge><i id=facebar></i></span></span>
  <span class=chip id=activechip>preset: <b id=activeName>—</b></span>
  <span class=chip style=margin-left:auto><i class="dot" id=conndot></i><span id=conn>connecting…</span></span>
</header>

<div class=grid>
  <div class=col>
    <div class=card>
      <h2>Presence feed</h2>
      <img id=feed src="/video.mjpg" alt="camera">
      <div class=controls>
        <button class=primary id=arm>Arm</button>
        <button id=disarm>Disarm</button>
        <button id=talk>● Hold to Talk</button>
      </div>
      <div class=prow style=margin-top:10px>
        <label for=vol title="George's head-speaker volume (the robot's own setting, kept across restarts).">speaker volume</label>
        <input type=range id=vol min=0 max=100 step=5 style=flex:1>
        <span class=tag id=volst>—</span>
      </div>
      <div class=prow style=margin-top:6px>
        <label for=sgain title="Software gain on George's replies, on top of the speaker volume. Lower = less of his own voice in the mic (barge-in tells him and the visitor apart better), but quieter in a noisy hall.">speech gain</label>
        <input type=range id=sgain min=0.5 max=3 step=0.25 style=flex:1>
        <span class=tag id=sgainst>—</span>
      </div>
      <div class=hint style=margin-top:10px>browser mic · Hold-to-Talk (needs https / localhost)</div>
      <canvas id=wave width=620 height=46 class=wave></canvas>
      <div class=hint id=micst style=margin-top:6px></div>
      <div class=hint style="margin-top:12px;color:var(--acc2)">robot VAD · live mic level &amp; calibration thresholds</div>
      <div class=vmeter>
        <div class=vmtrack>
          <div class=vmfill id=vm_fill></div>
          <div class="vmmark floor" id=vm_floor title="noise floor"></div>
          <div class="vmmark onset" id=vm_onset title="onset threshold (speech start)"></div>
          <div class="vmmark offset" id=vm_offset title="offset threshold (speech end)"></div>
        </div>
        <div class=vmlabel id=vm_label>VAD idle · run calibration to set thresholds</div>
      </div>
    </div>

    <div class=card>
      <h2>Tuning · detection &amp; VAD</h2>
      <div class=tune>
        <label>proximity gate <span id=v_prox>—</span></label><input type=range id=t_prox min=0.08 max=0.45 step=0.01>
        <label>arm seconds <span id=v_arm>—</span></label><input type=range id=t_arm min=0.3 max=3 step=0.1>
        <label>disarm seconds <span id=v_disarm>—</span></label><input type=range id=t_disarm min=0.5 max=5 step=0.1>
        <label>end-of-speech ms <span id=v_silence_ms>—</span></label><input type=range id=t_silence_ms min=300 max=1500 step=50>
        <label>VAD onset &times; <span id=v_noise_mult>—</span></label><input type=range id=t_noise_mult min=1.2 max=3 step=0.1>
      </div>
      <div class=prow style=margin-top:10px>
        <label for=micsrc>mic source</label>
        <select id=micsrc><option value=array>G1 mic array</option><option value=usb>USB mic</option></select>
        <span class=tag id=micsrcst></span>
      </div>
      <div class=prow style=margin-top:10px>
        <button id=calib>Run VAD calibration</button>
        <span class=tag id=calibst></span>
      </div>
    </div>

    <div class=card>
      <h2>Conversation link · experimental</h2>
      <div class=prow>
        <label for=st_transport title="How George talks to the gateway during a conversation.">link</label>
        <select id=st_transport>
          <option value=http>HTTP · one utterance per request (robot VAD)</option>
          <option value=ws>WebSocket · streamed (gateway VAD, barge-in)</option>
        </select>
      </div>
      <div id=st_ws>
        <div class=prow style=margin-top:8px>
          <label class=chk title="Let the visitor talk over George: his reply stops and the new speech is answered."><input type=checkbox id=st_barge_in> barge-in</label>
          <label class=chk title="Remove George's own voice from the mic (WebRTC echo canceller, as on Reachy Mini). Needed for barge-in."><input type=checkbox id=st_aec> echo cancellation</label>
          <label class=chk title="The canceller's built-in noise suppression."><input type=checkbox id=st_aec_noise_suppression> noise suppression</label>
          <label class=chk title="The canceller's built-in automatic gain control."><input type=checkbox id=st_aec_gain_control> auto gain</label>
        </div>
        <div class=prow>
          <label for=st_mic_gain title="Gain applied to the mic before streaming. Empty = automatic (4x for the quiet G1 array, 1x for USB).">mic gain</label>
          <input id=st_mic_gain type=number min=0 max=20 step=0.5 placeholder=auto style=width:84px>
          <label for=st_aec_delay_ms title="Where the echo reference is placed after a clip is sent to the speaker. Empty = automatic. Only the older canceller needs a figure.">echo delay ms</label>
          <input id=st_aec_delay_ms type=number min=0 max=1000 step=5 placeholder=auto style=width:84px>
        </div>
        <h2 style=margin-top:12px>Gateway turn detection</h2>
        <div class=tune>
          <label>speech threshold <span id=sv_vad_threshold>—</span></label><input type=range id=sx_vad_threshold min=0.2 max=0.95 step=0.05>
          <label>speech needed to start a turn, ms <span id=sv_vad_min_speech_ms>—</span></label><input type=range id=sx_vad_min_speech_ms min=96 max=1200 step=32>
          <label>silence that ends a segment, ms <span id=sv_vad_min_silence_ms>—</span></label><input type=range id=sx_vad_min_silence_ms min=32 max=800 step=32>
          <label>pre-roll kept before speech, ms <span id=sv_vad_speech_pad_ms>—</span></label><input type=range id=sx_vad_speech_pad_ms min=0 max=1000 step=50>
          <label>reply held after a finished turn, ms <span id=sv_reopen_ms>—</span></label><input type=range id=sx_reopen_ms min=0 max=3000 step=100>
        </div>
        <div class=prow style=margin-top:8px>
          <label class=chk title="Judge from the audio whether the visitor has finished, instead of treating every pause as the end of the turn."><input type=checkbox id=sx_smart_turn> Smart Turn</label>
        </div>
        <div class=tune>
          <label>Smart Turn threshold <span id=sv_smart_turn_threshold>—</span></label><input type=range id=sx_smart_turn_threshold min=0.1 max=0.9 step=0.05>
          <label>reply held after an unfinished turn, ms <span id=sv_smart_turn_max_wait_ms>—</span></label><input type=range id=sx_smart_turn_max_wait_ms min=500 max=6000 step=100>
          <label>wait before answering an unfinished turn, ms <span id=sv_smart_turn_incomplete_delay_ms>—</span></label><input type=range id=sx_smart_turn_incomplete_delay_ms min=0 max=3000 step=100>
        </div>
        <div class=prow style=margin-top:8px><button id=st_reset>Gateway defaults</button></div>
      </div>
      <div class=prow style=margin-top:6px><span class=tag id=st_status></span></div>
    </div>

    <div class=card>
      <h2>Gateway pipeline</h2>
      <div class=prow>
        <label for=gw_heard title="Who writes the HEARD transcript. Off is fastest when Gemma hears the audio itself.">transcript (HEARD)</label>
        <select id=gw_heard>
          <option value=off>off</option>
          <option value=gemma>Gemma (2nd LLM call)</option>
          <option value=parakeet>Parakeet</option>
        </select>
      </div>
      <div class=prow style=margin-top:10px>
        <label for=gw_input title="What Gemma answers from: the speech itself, or a Parakeet transcript of it (cascade).">Gemma answers from</label>
        <select id=gw_input>
          <option value=audio>the audio (audio-native)</option>
          <option value=transcript>the transcript (cascade)</option>
        </select>
      </div>
      <div class=prow style=margin-top:10px><span class=tag id=gwst></span></div>
    </div>

    <div class=card>
      <h2>Logs</h2>
      <div id=log></div>
    </div>
  </div>

  <div class=col>
    <div class=card>
      <h2>Pipeline · live latency</h2>
      <div class=tiles>
        <div class=tile title="Total round-trip — from audio sent to the reply finishing"><div class=k>total time</div><div class=v id=t_total>—</div></div>
        <div class=tile title="Time to the first generated word (time-to-first-token)"><div class=k>to first word</div><div class=v id=t_ttft>—</div></div>
        <div class=tile title="Generation speed — words (tokens) per second"><div class=k>words / sec</div><div class=v id=t_tps>—</div></div>
        <div class=tile title="Time until the first speech audio is ready to play"><div class=k>to first sound</div><div class=v id=t_fa>—</div></div>
      </div>
      <svg id=diagram class=diagram viewBox="0 0 1080 410" preserveAspectRatio="xMidYMid meet"></svg>
      <div class=flowrow>
        <div><div class=hint style=margin-bottom:4px>last image sent</div><img id=lastimg alt=""></div>
      </div>
    </div>

    <div class=card>
      <h2>Conversation</h2>
      <div class=hint id=cv_session>no visitor yet</div>
      <div id=cv_turns></div>
    </div>

    <div class=card>
      <h2>Prompt presets</h2>
      <div class=prow>
        <select id=preset></select>
        <button id=newp>+ New</button>
        <button id=activate>Activate</button>
        <button id=del>Delete</button>
        <span class=tag id=psaved></span>
      </div>
      <label>name</label><input id=pname style=width:100% placeholder="preset name">
      <label>greet · spoken on arrival</label><textarea id=greet rows=2></textarea>
      <label>converse · system prompt / persona</label><textarea id=converse rows=4></textarea>
      <label>goodbye · spoken on leaving</label><textarea id=goodbye rows=2></textarea>
      <div class=prow style=margin-top:10px><button class=primary id=save>Save preset</button></div>
    </div>
  </div>
</div>

<script>
const $=id=>document.getElementById(id);
const fmt=ms=>ms==null?'—':(ms>=1000?(ms/1000).toFixed(2)+'s':Math.round(ms)+'ms');
function ctl(action){fetch('/api/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action})});}
$('arm').onclick=()=>ctl('arm'); $('disarm').onclick=()=>ctl('disarm');

// Pipeline diagram — real SVG via createElementNS (renders reliably). Mirrors the
// actual flow: Camera feeds the Detector (presence GATE that arms the loop) and
// also sends a frame (IMAGE) to Gemma; Mic/VAD sends AUDIO to the gateway; the
// gateway runs decode -> ASR -> Gemma 4 E4B -> kokoro TTS; reply audio plays out
// the G1 head speaker.
const SVGNS='http://www.w3.org/2000/svg';
function E(t,a,k){const e=document.createElementNS(SVGNS,t);for(const n in(a||{}))e.setAttribute(n,a[n]);(k||[]).forEach(c=>c&&e.appendChild(c));return e;}
function T(s,a){const e=E('text',a);e.textContent=s;return e;}
const N={
  cam:{l:'Camera',x:16,y:26,w:118,h:76,sub:'live',
    desc:"Camera — the robot's webcam. A single frame is sent to Gemma each turn so it can SEE the person and react to what they look like / are wearing / are doing."},
  detector:{l:'Detector',x:16,y:150,w:118,h:76,sub:'gate',
    desc:"Detector — on-board face detector (OpenCV Haar cascade). It watches the camera and ARMS a greeting once a face is close enough (the proximity gate). It's only a trigger; it never talks to the gateway."},
  mic:{l:'Mic / VAD',x:16,y:274,w:118,h:76,sub:'audio in',
    desc:"Mic / VAD — microphone + voice-activity detection. Captures the person's speech, trims the silence, and sends the clip to the gateway. (In laptop dev mode the browser 'Hold to Talk' button stands in for this.)"},
  audio_decode:{l:'decode',x:300,y:198,w:104,h:84,bar:1,sub:'audio',
    desc:"Decode — gateway step 1: turns the uploaded audio file into raw samples the model can read. Usually a millisecond or two."},
  llm:{l:'Gemma 4B',x:452,y:104,w:150,h:82,bar:1,sub:'reason + see',
    desc:"Gemma 4 E4B — the audio-native vision-language model (the 'brain'). It listens to the audio AND looks at the image, then generates the spoken reply. TTFT is how long until the first word."},
  tts:{l:'kokoro',x:452,y:210,w:150,h:72,bar:1,sub:'speech',
    desc:"kokoro TTS — text-to-speech. Converts Gemma's reply text into speech audio, sentence by sentence, so the first words can start playing before the whole reply is written."},
  asr:{l:'ASR',x:452,y:300,w:150,h:82,bar:1,sub:'transcript',
    desc:"ASR — transcribes the speech to text: either a second Gemma call or Parakeet (see Gateway pipeline). Normally it only feeds HEARD and the reply doesn't depend on it, because Gemma hears the audio directly. In cascade mode Gemma answers from this transcript instead of the audio."},
  spk:{l:'G1 Speaker',x:690,y:206,w:120,h:80,sub:'playback',
    desc:"G1 Speaker — the robot's head speaker plays the reply audio (over DDS on the robot; your laptop speakers in dev mode). 'Playback' is how many seconds of speech were produced."},
  reply:{l:'REPLY',x:690,y:104,w:352,h:82,fo:1,cls:'r',tid:'reply',
    desc:"Reply — Gemma's spoken response text. The same text is sent to kokoro to be voiced out the speaker."},
  heard:{l:'HEARD',x:690,y:300,w:352,h:82,fo:1,cls:'h',tid:'heard',
    desc:"Heard — the ASR transcript of what the person said. Display only; only present when transcription was requested."},
};
const cyN=n=>N[n].y+N[n].h/2, rxN=n=>N[n].x+N[n].w, cxN=n=>N[n].x+N[n].w/2;
// decode FANS OUT to Gemma (reason, audio-native) + ASR (transcript). Gemma then
// fans to REPLY (text shown) and kokoro->Speaker (spoken); ASR -> HEARD.
const LINKS=[
  {a:'cam',b:'detector',v:1},
  {a:'detector',b:'mic',v:1,arm:1,lbl:'arm'},
  {a:'mic',b:'audio_decode',flow:1,lbl:'audio'},
  {a:'cam',b:'llm',top:1,flow:1,img:1,lbl:'image'},
  {a:'audio_decode',b:'llm',flow:1,a2l:1},
  {a:'audio_decode',b:'asr',flow:1,toasr:1},
  {a:'llm',b:'tts',v:1,flow:1},        // Gemma -> kokoro (down)
  {a:'tts',b:'spk',flow:1},            // kokoro -> Speaker
  {a:'llm',b:'reply',flow:1},          // Gemma -> REPLY text (right)
  {a:'asr',b:'heard',flow:1},          // ASR -> HEARD transcript (right)
  {a:'asr',b:'llm',casc:1,lbl:'transcript'},  // cascade: Gemma answers from the transcript
];
// Streamed (WebSocket) mode: the mic is open and streamed, the robot cancels its
// own voice (AEC), and the GATEWAY decides when a turn ends: Silero VAD (endpoint)
// -> Smart Turn (is the visitor done?) -> hold (reopen grace, the reply is drafted
// meanwhile) -> Gemma / ASR. Same canvas, decode swapped for those three stages.
const N_WS=Object.assign({},N,{
  mic:{...N.mic,l:'Mic · open',sub:'AEC',
    desc:"Mic, open — in streamed mode the mic is never cut into clips on the robot: every 30 ms frame goes to the gateway over a WebSocket. Echo cancellation (AEC, WebRTC) removes George's own voice first, so he can be interrupted without hearing himself. The number is the measured speaker delay the canceller works with."},
  vad:{l:'VAD',x:296,y:96,w:112,h:74,bar:1,sub:'silero · end',
    desc:"VAD — the gateway's Silero voice-activity detector finds where the visitor's speech starts and stops. The time is the endpoint: how long it waited in silence before closing the segment."},
  smart_turn:{l:'Smart Turn',x:296,y:190,w:112,h:74,bar:1,sub:'turn done?',
    desc:"Smart Turn — a small model listens to the end of the segment and judges whether the visitor has finished (a complete turn) or only paused. Shows its inference time and probability; an unfinished turn makes the gateway wait longer before replying."},
  reopen_grace:{l:'hold',x:296,y:284,w:112,h:74,bar:1,sub:'reopen grace',
    desc:"Hold — the reopen grace: the reply is held back briefly in case the visitor carries on talking. Gemma is already drafting the reply during the hold, so this overlaps the LLM time rather than adding to it."},
});
delete N_WS.audio_decode;
const LINKS_WS=[
  {a:'cam',b:'detector',v:1},
  {a:'detector',b:'mic',v:1,arm:1,lbl:'arm'},
  {a:'mic',b:'vad',flow:1,lbl:'stream'},
  {a:'cam',b:'llm',top:1,flow:1,img:1,lbl:'image'},
  {a:'vad',b:'smart_turn',v:1,flow:1},
  {a:'smart_turn',b:'reopen_grace',v:1,flow:1},
  {a:'reopen_grace',b:'llm',flow:1,a2l:1},
  {a:'reopen_grace',b:'asr',flow:1,toasr:1},
  {a:'llm',b:'tts',v:1,flow:1},
  {a:'tts',b:'spk',flow:1},
  {a:'llm',b:'reply',flow:1},
  {a:'asr',b:'heard',flow:1},
  {a:'asr',b:'llm',casc:1,lbl:'transcript'},
];
const CASC_LANE=430;   // x of the cascade transcript lane, in the gap left of the Gemma/ASR column
const DIAG={mode:null,nodes:N,gw:['audio_decode','asr','llm','tts']};
function build(mode){
  const ws=mode==='ws', NODES=ws?N_WS:N, LNK=ws?LINKS_WS:LINKS;
  DIAG.mode=mode; DIAG.nodes=NODES;
  DIAG.gw=ws?['vad','smart_turn','reopen_grace','asr','llm','tts']:['audio_decode','asr','llm','tts'];
  const cyN=n=>NODES[n].y+NODES[n].h/2, rxN=n=>NODES[n].x+NODES[n].w, cxN=n=>NODES[n].x+NODES[n].w/2;
  const svg=$('diagram'); while(svg.firstChild)svg.removeChild(svg.firstChild);
  svg.appendChild(E('defs',{},[
    E('linearGradient',{id:'ng',x1:0,y1:0,x2:0,y2:1},[E('stop',{offset:0,'stop-color':'#1d2738'}),E('stop',{offset:1,'stop-color':'#121925'})]),
    E('linearGradient',{id:'bg2',x1:0,y1:0,x2:1,y2:0},[E('stop',{offset:0,'stop-color':'#6ea8ff'}),E('stop',{offset:1,'stop-color':'#3ddc97'})]),
    E('filter',{id:'glow',x:'-50%',y:'-50%',width:'200%',height:'200%'},[E('feDropShadow',{dx:0,dy:0,stdDeviation:4,'flood-color':'#3ddc97','flood-opacity':0.85})]),
    E('marker',{id:'arr',viewBox:'0 0 12 12',refX:0,refY:6,markerWidth:12,markerHeight:12,markerUnits:'userSpaceOnUse',orient:'auto'},[E('path',{d:'M0 0 L12 6 L0 12 z',fill:'rgba(233,238,245,.6)'})])
  ]));
  svg.appendChild(E('rect',{class:'gwbox',x:284,y:86,width:336,height:300,rx:16}));
  svg.appendChild(T(ws?'GATEWAY · /ws/converse · Silero VAD + Smart Turn · Gemma 4 E4B':'GATEWAY · Gemma 4 E4B · vLLM / RTX 6000',{class:'gwlabel',x:294,y:404}));
  const gtot=T(ws?'stop talking → first audio —':'round-trip Σ —',{class:'gwtotal',id:'gw_total',x:1042,y:404,'text-anchor':'end'});
  const gtt=document.createElementNS(SVGNS,'title');
  gtt.textContent=ws?'From the visitor going quiet to the first reply audio: the VAD endpoint wait + the gateway time to first audio (Smart Turn, the hold and Gemma overlap, so the stages are not simply added).'
                    :'Aggregated gateway round-trip: sum of decode + ASR + LLM + TTS for this turn.';
  gtot.appendChild(gtt);
  svg.appendChild(gtot);
  const GAP=12;   // line stops this far before the node; the base-anchored 12-long
                  // arrowhead bridges it, so the line meets the arrow's BACK CENTRE.
  LNK.forEach(L=>{
    let ax,ay,bx,by,d;
    if(L.casc){   // ASR upper-left -> lane -> Gemma lower-left (clear of the ASR feed below)
      const cx=CASC_LANE,r=6; ax=NODES[L.a].x;ay=NODES[L.a].y+16;bx=NODES[L.b].x-GAP;by=NODES[L.b].y+NODES[L.b].h-18;
      d=`M${ax} ${ay} L${cx+r} ${ay} Q${cx} ${ay} ${cx} ${ay-r} L${cx} ${by+r} Q${cx} ${by} ${cx+r} ${by} L${bx} ${by}`;}
    else if(L.v){ax=cxN(L.a);ay=NODES[L.a].y+NODES[L.a].h;bx=cxN(L.b);by=NODES[L.b].y-GAP;d=`M${ax} ${ay} L${bx} ${by}`;}
    else if(L.top){ax=rxN(L.a);ay=cyN(L.a);bx=cxN(L.b);by=NODES[L.b].y-GAP;d=`M${ax} ${ay} C${(ax+bx)/2} ${ay}, ${bx} ${ay}, ${bx} ${by}`;}
    else{ax=rxN(L.a);ay=cyN(L.a);bx=NODES[L.b].x-GAP;by=cyN(L.b);d=`M${ax} ${ay} C${(ax+bx)/2} ${ay}, ${(ax+bx)/2} ${by}, ${bx} ${by}`;}
    const cls=L.casc?'flow casc':(L.flow?('flow'+(L.img?' img':'')+(L.a2l?' a2l':'')+(L.toasr?' notcasc':'')):('link'+(L.arm?' arm':'')));
    svg.appendChild(E('path',{class:cls,d:d,'marker-end':'url(#arr)'}));
    if(L.toasr){  // cascade: elbow on its own lane into ASR's lower-left, under the transcript lane
      const r=6, vx=rxN(L.a)+12, ey=NODES[L.b].y+NODES[L.b].h-22, sx=rxN(L.a), sy=cyN(L.a), ex=NODES[L.b].x-GAP;
      const de=`M${sx} ${sy} L${vx-r} ${sy} Q${vx} ${sy} ${vx} ${sy+r} L${vx} ${ey-r} Q${vx} ${ey} ${vx+r} ${ey} L${ex} ${ey}`;
      svg.appendChild(E('path',{class:'flow onlycasc',d:de,'marker-end':'url(#arr)'}));}
    if(L.lbl){let lx=(ax+bx)/2,ly=(ay+by)/2-5,anc='middle';
      if(L.img){lx=ax+(bx-ax)*0.26;ly=ay-7;}
      if(L.v){lx=ax+15;ly=(ay+by)/2+3;anc='start';}
      if(L.casc){lx=CASC_LANE+11;ly=(ay+by)/2;anc='middle';}   // vertical, right of the lane
      const la={class:'linklbl'+(L.casc?' casc':''),'text-anchor':anc,x:lx,y:ly};
      if(L.casc)la.transform=`rotate(-90 ${lx} ${ly})`;
      svg.appendChild(T(L.lbl,la));}
  });
  for(const k in NODES){const n=NODES[k];
    const g=E('g',{class:'node'+(n.fo?' fonode':''),id:'n_'+k,transform:`translate(${n.x},${n.y})`});
    if(n.desc){const ti=document.createElementNS(SVGNS,'title');ti.textContent=n.desc;g.appendChild(ti);}
    if(n.fo){
      const fo=E('foreignObject',{x:0,y:0,width:n.w,height:n.h});
      const div=document.createElement('div');div.className='fobub '+(n.cls||'');
      div.innerHTML='<div class="fowho">'+n.l+'</div><div class="fotext" id="'+n.tid+'">—</div>';
      fo.appendChild(div);g.appendChild(fo);svg.appendChild(g);continue;
    }
    g.appendChild(E('rect',{class:'nbox',width:n.w,height:n.h,rx:13}));
    g.appendChild(T(n.l,{class:'nname',x:n.w/2,y:n.bar?20:22}));
    g.appendChild(T('—',{class:'nlat',id:'lat_'+k,x:n.w/2,y:n.bar?42:44}));
    if(n.sub)g.appendChild(T(n.sub,{class:'nsub',id:'sub_'+k,x:n.w/2,y:n.bar?(n.h-22):(n.h-10)}));
    if(n.bar){g.appendChild(E('rect',{class:'barbg',x:12,y:n.h-13,width:n.w-24,height:5,rx:3}));
              g.appendChild(E('rect',{class:'barfg',id:'bar_'+k,x:12,y:n.h-13,width:0,height:5,rx:3}));}
    if(k==='spk'){for(let i=0;i<4;i++)g.appendChild(E('rect',{class:'eqbar',x:n.w/2-18+i*10,y:n.h-26,width:6,height:12,rx:2}));}
    svg.appendChild(g);
  }
}
build('http');
function node(key,label){const e=$('lat_'+key); if(e)e.textContent=label;}
function bar(key,frac){const b=$('bar_'+key),n=DIAG.nodes[key]; if(b&&n)b.setAttribute('width',Math.max(0,Math.min(1,frac))*(n.w-24));}
function subt(key,txt){const e=$('sub_'+key); if(e)e.textContent=txt;}
function active(key,on){const g=$('n_'+key); if(g)g.classList.toggle('active',on);}

const es=new EventSource('/state');
es.onopen=()=>{$('conn').textContent='live';$('conndot').className='dot live';};
es.onerror=()=>{$('conn').textContent='reconnecting…';$('conndot').className='dot bad';};
es.onmessage=e=>{let s;try{s=JSON.parse(e.data)}catch(_){return}
  const p=s.presence||{};
  $('armtxt').textContent=p.enabled?'ARMED':'idle';
  $('armdot').className='dot '+(p.enabled?'armed':'');
  $('facetxt').textContent=(p.face_frac||0).toFixed(2)+' / '+(p.thr||0).toFixed(2);
  const fr=p.thr?Math.min(1,(p.face_frac||0)/Math.max(p.thr,0.01)):0;
  $('facebar').style.width=(fr*100)+'%';
  $('facebar').style.background=(p.face_frac>=p.thr&&p.thr>0)?'linear-gradient(90deg,#3ddc97,#2bb87c)':'linear-gradient(90deg,#6ea8ff,#3ddc97)';
  $('arm').classList.toggle('on',!!p.enabled);
  $('diagram').classList.toggle('flowing',!!s.in_progress);
  DIAG.gw.forEach(k=>active(k,!!s.in_progress));
  const sp=$('n_spk'); if(sp)sp.classList.toggle('talking',!!s.speaking);

  // robot VAD meter (server-pushed; works over plain http unlike the browser mic)
  {const vd=s.vad||{},lvl=vd.level||0,fl=vd.floor||0,on=vd.onset||0,off=vd.offset||0;
   const mx=Math.max(0.06,on*2.2,lvl*1.15,fl*2.5);
   const pct=v=>Math.max(0,Math.min(100,(v/mx)*100));
   const fill=$('vm_fill'); if(fill){fill.style.width=pct(lvl)+'%';fill.classList.toggle('voiced',!!vd.voiced);}
   const setm=(id,v)=>{const e=$(id); if(e)e.style.left=pct(v)+'%';};
   setm('vm_floor',fl);setm('vm_onset',on);setm('vm_offset',off);
   const lab=$('vm_label');
   if(lab)lab.textContent='VAD '+(vd.active?(vd.phase||'…'):'idle')
     +'  ·  level '+lvl.toFixed(3)+'  ·  floor '+(fl?fl.toFixed(3):'—')+(vd.samples>1?' (avg '+vd.samples+')':'')
     +'  ·  onset '+on.toFixed(3)+'  ·  offset '+off.toFixed(3);}

  const t=s.turn||{},m=t.metrics||{},h=t.hops||{},comp={};
  (m.components||[]).forEach(c=>comp[c.name]=c.ms);
  if(comp.endpoint!=null)comp.vad=comp.endpoint;     // streamed: VAD node = endpoint wait
  // which ASR engine ran, and whether Gemma answered from the audio or the transcript
  const asrc=(m.components||[]).find(c=>c.name==='asr'), sa=$('sub_asr'), sl=$('sub_llm');
  const gwi=$('gw_input'), gwh=$('gw_heard');
  const casc=(m.llm_input||(gwi&&gwi.value)||'audio')==='transcript';
  const heardSel=gwh?gwh.value:'';
  const eng=(asrc&&asrc.engine)||(casc?'parakeet':(heardSel&&heardSel!=='off'?heardSel:''));
  $('diagram').classList.toggle('cascade',casc);
  if(sa)sa.textContent=eng?(eng+(casc?' · feeds Gemma':' · display only')):'off';
  if(sl)sl.textContent=casc?'reads transcript + sees':'reason + see';
  const secs=v=>(v==null)?'—':(+v).toFixed(1)+'s';
  node('cam','live'); node('detector','n='+(p.nfaces||0));
  const gw=DIAG.gw, ws=DIAG.mode==='ws';
  const mx=Math.max(1,...gw.map(k=>comp[k]||0));
  gw.forEach(k=>{node(k,fmt(comp[k])); bar(k,(comp[k]||0)/mx);});
  const gt=$('gw_total'), spn=$('n_spk');
  if(ws){
    // streamed: endpoint wait + gateway time to first audio (the stages overlap)
    const ep=comp.endpoint, vadc=(m.components||[]).find(c=>c.name==='endpoint');
    const stc=(m.components||[]).find(c=>c.name==='smart_turn');
    subt('smart_turn',stc?((stc.complete?'done ':'unfinished ')+(stc.probability!=null?'p='+(+stc.probability).toFixed(2):'')):'turn done?');
    if(!vadc)subt('vad','silero · end');
    const tfa=m.time_to_first_audio_ms;
    if(gt)gt.textContent='stop talking → first audio '+(tfa!=null?fmt((ep||0)+tfa):'—');
    const st=(ST&&ST.status)||{}, o=(ST&&ST.options)||{};
    const dly=st.echo_delay_ms!=null?st.echo_delay_ms:(ST&&ST.echo_delay_ms);
    node('mic',dly!=null?Math.round(dly)+'ms':secs(m.audio_seconds));
    subt('mic',!o.aec?'no AEC':(st.aec&&st.aec.indexOf('on')===0?(st.aec.indexOf('AEC3')>=0?'AEC3 · delay':'AEC · delay'):(ST&&ST.aec_backend?'AEC ready':'AEC missing')));
    const cut=/ —$/.test(t.reply||'');
    subt('spk',cut?'barge-in: cut':(st.barge_in?'barge-in on':'half-duplex'));
    if(spn)spn.classList.toggle('cut',cut);
  } else {
    node('mic',secs(m.audio_seconds));
    const gwsum=gw.reduce((a,k)=>a+(comp[k]||0),0);
    if(gt)gt.textContent='round-trip Σ '+(gwsum>0?fmt(gwsum):'—');
    subt('spk','playback'); if(spn)spn.classList.remove('cut');
  }
  node('spk',secs(m.tts_audio_seconds));
  $('t_total').textContent=fmt(h.total_ms);
  $('t_ttft').textContent=fmt(m.ttft_ms!=null?m.ttft_ms:h.first_token_ms);
  $('t_tps').textContent=m.tokens_per_sec?m.tokens_per_sec.toFixed(0):'—';
  $('t_fa').textContent=fmt(h.first_audio_ms!=null?h.first_audio_ms:m.time_to_first_audio_ms);
  $('heard').textContent=t.heard||'—'; $('reply').textContent=t.reply||'—';
  if(s.has_image)$('lastimg').src='/last_image.jpg?t='+Math.floor(Date.now()/800);
};

// logs
const lg=$('log'),les=new EventSource('/logs');
les.onmessage=e=>{const l=e.data.replace(/\\n/g,'\n');const d=document.createElement('div');
  d.className=/\[gateway\]|reply:|greeting|calibrated/.test(l)?'g':(/error|fail|WARN/i.test(l)?'w':(/\[presence\]|\[web\]|\[arm\]/.test(l)?'':'m'));
  d.textContent=l;lg.appendChild(d);if(lg.childElementCount>800)lg.removeChild(lg.firstChild);lg.scrollTop=lg.scrollHeight;};

// prompts
let P={presets:[],active:''};
function flash(t){$('psaved').textContent=t;setTimeout(()=>$('psaved').textContent='',1800);}
function loadP(sel){fetch('/api/prompts').then(r=>r.json()).then(d=>{P=d;const s=$('preset');s.innerHTML='';
  d.presets.forEach(p=>{const o=document.createElement('option');o.value=o.textContent=p.name;s.appendChild(o);});
  const pick=(sel&&d.presets.some(p=>p.name===sel))?sel:d.active;
  s.value=pick;$('activeName').textContent=d.active;show(pick);});}
function show(n){const p=(P.presets||[]).find(x=>x.name===n);if(!p)return;
  $('pname').value=p.name;$('greet').value=p.greet||'';$('converse').value=p.converse||'';$('goodbye').value=p.goodbye||'';}
$('preset').onchange=e=>show(e.target.value);
$('newp').onclick=()=>{$('pname').value='';$('greet').value='';$('converse').value='';$('goodbye').value='';$('pname').focus();flash('new preset — name it & Save');};
$('save').onclick=()=>{const nm=$('pname').value.trim();if(!nm){flash('name required');$('pname').focus();return;}
  fetch('/api/prompts',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({name:nm,greet:$('greet').value,converse:$('converse').value,goodbye:$('goodbye').value})})
    .then(r=>r.json()).then(d=>{P=d;loadP(nm);flash('saved ✓');});};
$('activate').onclick=()=>fetch('/api/prompts/active',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:$('preset').value})}).then(r=>r.json()).then(d=>{P=d;loadP($('preset').value);flash('activated ✓');});
$('del').onclick=()=>{if(!confirm('Delete '+$('preset').value+'?'))return;fetch('/api/prompts/'+encodeURIComponent($('preset').value),{method:'DELETE'}).then(r=>r.json()).then(d=>{P=d;loadP();});};
loadP();

// tuning: presence + VAD thresholds + calibration
const TKEYS=['prox','arm','disarm','silence_ms','noise_mult'];
let tuneT;
function loadTune(){fetch('/api/tune').then(r=>r.json()).then(d=>{TKEYS.forEach(k=>{
  const el=$('t_'+k); if(el&&d[k]!=null){el.value=d[k];$('v_'+k).textContent=(''+d[k]);}});
  if(d.mic_source){$('micsrc').value=d.mic_source;$('micsrcst').textContent='using '+d.mic_source;}});}
$('micsrc').onchange=()=>{const src=$('micsrc').value;$('micsrcst').textContent='switching…';
  fetch('/api/mic',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({source:src})})
    .then(r=>r.json()).then(d=>{$('micsrcst').textContent=d.ok?('using '+d.mic_source+' · recalibrates when idle'):(d.error||'failed');loadTune();});};
TKEYS.forEach(k=>{const el=$('t_'+k); if(el)el.oninput=()=>{$('v_'+k).textContent=el.value;
  clearTimeout(tuneT);tuneT=setTimeout(()=>fetch('/api/tune',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({[k]:parseFloat(el.value)})}),200);};});
// conversation link: HTTP vs streamed WebSocket, barge-in, echo cancellation, and
// this robot's overrides of the gateway's turn detection (applied live)
const ST_BOOLS=['barge_in','aec','aec_noise_suppression','aec_gain_control'];
const ST_VAD=['vad_threshold','vad_min_speech_ms','vad_min_silence_ms','vad_speech_pad_ms','reopen_ms',
              'smart_turn_threshold','smart_turn_max_wait_ms','smart_turn_incomplete_delay_ms'];
// the gateway's stock values, shown until a live session reports what is in effect
const ST_DEF={vad_threshold:0.6,vad_min_speech_ms:384,vad_min_silence_ms:64,vad_speech_pad_ms:500,reopen_ms:800,
              smart_turn:true,smart_turn_threshold:0.5,smart_turn_max_wait_ms:2000,smart_turn_incomplete_delay_ms:600};
let stBusy=false, ST=null;
function showStream(d){
  ST=d;                                            // read by the diagram (AEC / barge-in state)
  if(DIAG.mode!==d.options.transport)build(d.options.transport);   // diagram follows the link
  const o=d.options, gw=(d.status&&d.status.gateway)||{}, s=d.status||{};
  $('st_transport').value=o.transport;
  $('st_ws').style.display=o.transport==='ws'?'':'none';
  ST_BOOLS.forEach(k=>{$('st_'+k).checked=!!o[k];});
  ['mic_gain','aec_delay_ms'].forEach(k=>{ if(document.activeElement!==$('st_'+k))$('st_'+k).value=(o[k]==='auto'||o[k]==null)?'':o[k]; });
  const eff=k=>(k in o.vad)?o.vad[k]:((k in gw)?gw[k]:ST_DEF[k]);
  ST_VAD.forEach(k=>{const el=$('sx_'+k); if(document.activeElement!==el)el.value=eff(k);
    $('sv_'+k).textContent=eff(k)+((k in o.vad)?'':' · default');});
  $('sx_smart_turn').checked=!!eff('smart_turn');
  const bits=[];
  if(o.transport==='ws'){
    bits.push(d.live?(s.connected?'streaming':'connecting…'):'idle until the next visitor');
    if(o.aec)bits.push('echo cancellation: '+(d.live?(s.aec||'starting'):(d.aec_backend?('ready ('+d.aec_backend+')'):'NOT AVAILABLE — start the aec service')));
    if(d.live)bits.push('barge-in '+(s.barge_in?'on':'off'+(o.barge_in?' (waiting for the echo delay)':'')));
    if(o.barge_in&&(!o.aec||!d.aec_backend)&&!d.live)bits.push('⚠ barge-in without echo cancellation: George will interrupt himself');
    if(d.echo_delay_ms!=null)bits.push('speaker delay '+Math.round(d.echo_delay_ms)+' ms');
    if(gw.smart_turn_available===false)bits.push('Smart Turn is not loaded on the gateway');
    if(s.error)bits.push('error: '+s.error);
  } else bits.push('using the robot VAD and /converse');
  $('st_status').textContent=bits.join(' · ');
}
function loadStream(){ if(stBusy)return; fetch('/api/stream').then(r=>r.json()).then(showStream).catch(()=>{}); }
function saveStream(body){ stBusy=true;
  fetch('/api/stream',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})
    .then(r=>r.json()).then(d=>{stBusy=false;showStream(d);}).catch(()=>{stBusy=false;}); }
$('st_transport').onchange=()=>saveStream({transport:$('st_transport').value});
ST_BOOLS.forEach(k=>{$('st_'+k).onchange=()=>saveStream({[k]:$('st_'+k).checked});});
['mic_gain','aec_delay_ms'].forEach(k=>{$('st_'+k).onchange=()=>saveStream({[k]:$('st_'+k).value});});
let stT;
ST_VAD.forEach(k=>{const el=$('sx_'+k); el.oninput=()=>{$('sv_'+k).textContent=el.value;
  clearTimeout(stT);stT=setTimeout(()=>saveStream({vad:{[k]:parseFloat(el.value)}}),200);};});
$('sx_smart_turn').onchange=()=>saveStream({vad:{smart_turn:$('sx_smart_turn').checked}});
$('st_reset').onclick=()=>saveStream({reset_vad:true});
loadStream(); setInterval(loadStream,2000);

// gateway pipeline: transcript engine + what Gemma answers from (applies to the next turn)
function showGw(d){
  $('gw_heard').value=d.transcribe?(d.asr_engine||(d.gateway_default&&d.gateway_default.asr_engine)||'gemma'):'off';
  $('gw_input').value=d.llm_input||(d.gateway_default&&d.gateway_default.llm_input)||'audio';
  const eng=d.engines, hasP=!!eng&&eng.includes('parakeet');
  $('gw_heard').querySelector('option[value=parakeet]').disabled=!hasP;
  $('gw_input').querySelector('option[value=transcript]').disabled=!hasP;
  $('gwst').textContent=!eng?'gateway unreachable, or too old for these options'
    :(hasP?'gateway engines: '+eng.join(', '):'Parakeet is not loaded on the gateway')
     +(d.llm_input==='transcript'?' · cascade: HEARD is always shown':'');
}
function loadGw(){fetch('/api/gateway').then(r=>r.json()).then(showGw);}
function saveGw(){
  const h=$('gw_heard').value, inp=$('gw_input').value;
  // The cascade runs on Parakeet; "off" there only means no extra engine choice.
  const body={transcribe:h!=='off', asr_engine:h==='off'?(inp==='transcript'?'parakeet':''):h, llm_input:inp};
  $('gwst').textContent='saving…';
  fetch('/api/gateway',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})
    .then(r=>r.json()).then(showGw);
}
$('gw_heard').onchange=saveGw;
$('gw_input').onchange=()=>{ if($('gw_input').value==='transcript'&&$('gw_heard').value==='gemma')$('gw_heard').value='parakeet'; saveGw(); };
loadGw();

$('calib').onclick=()=>{$('calibst').textContent='calibrating…';
  fetch('/api/calibrate',{method:'POST'}).then(r=>r.json()).then(d=>{
    $('calibst').textContent=d.ok?('floor '+(+d.noise_floor).toFixed(4)+' → thr '+(+d.speech_thresh).toFixed(4)):(d.error||'failed');});};
loadTune();
// conversation: the current visitor's turns, each with the prompt Gemma saw
let cvKey='';
function esc(s){return String(s==null?'':s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function renderConv(d){
  const turns=d.turns||[], key=(d.session||'')+'|'+turns.length+'|'+turns.map(t=>(t.reply||'').length+(t.prompt?'p':'')).join(',');
  if(key===cvKey)return; cvKey=key;
  $('cv_session').textContent=d.session?('session '+d.session+' · '+turns.length+' turn(s)'):'no visitor yet';
  const open=new Set([...document.querySelectorAll('#cv_turns details[open]')].map(x=>x.dataset.i));
  $('cv_turns').innerHTML=turns.map((t,i)=>{
    const last=t.prompt&&t.prompt.length?t.prompt[t.prompt.length-1].content:'';
    const visitor=t.heard||(t.phase==='greet'?'(walked up)':t.phase==='goodbye'?'(walking away)':'');
    const msgs=(t.prompt||[]).map(m=>'<div class="cvmsg '+(m.role==='system'?'sys':m.role==='assistant'?'asst':'')+'"><span class=r>'+esc(m.role)+'</span>'+esc(m.content)+'</div>').join('');
    return '<div class=cvt><div class=ph>'+esc(t.phase||'turn')+(t.in_progress?' · in progress':'')
      +(t.total_ms?' · '+fmt(t.total_ms):'')+'</div>'
      +'<div class=ln><b>visitor</b>'+esc(visitor||'—')+'</div>'
      +'<div class=ln><b>George</b>'+esc(t.reply||'—')+'</div>'
      +(t.prompt?'<details data-i="'+i+'"'+(open.has(String(i))?' open':'')+'><summary>what Gemma saw · '+t.prompt.length+' messages</summary>'+msgs+'</details>'
                :(t.in_progress?'':'<div class=hint style=margin-top:4px>prompt not returned (gateway without debug_prompt)</div>'))
      +'</div>';}).join('');
  const box=$('cv_turns'); box.scrollTop=box.scrollHeight;
}
function loadConv(){fetch('/api/conversation').then(r=>r.json()).then(renderConv).catch(()=>{});}
loadConv(); setInterval(loadConv,2000);
// speaker volume (robot AudioClient Get/SetVolume)
let volT;
function showVol(d){ if(d&&d.ok&&d.volume!=null){ if(document.activeElement!==$('vol'))$('vol').value=d.volume; $('volst').textContent=d.volume+'%'; }
  else $('volst').textContent=(d&&d.error)||'unavailable'; }
fetch('/api/volume').then(r=>r.json()).then(showVol).catch(()=>showVol(null));
let sgT;
function showSg(d){ if(d&&d.ok){ if(document.activeElement!==$('sgain'))$('sgain').value=d.gain; $('sgainst').textContent='x'+(+d.gain).toFixed(2); }
  else $('sgainst').textContent=(d&&d.error)||'unavailable'; }
fetch('/api/speech_gain').then(r=>r.json()).then(showSg).catch(()=>showSg(null));
$('sgain').oninput=()=>{$('sgainst').textContent='x'+(+$('sgain').value).toFixed(2);clearTimeout(sgT);
  sgT=setTimeout(()=>fetch('/api/speech_gain',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({gain:parseFloat($('sgain').value)})}).then(r=>r.json()).then(showSg),250);};
$('vol').oninput=()=>{$('volst').textContent=$('vol').value+'%';clearTimeout(volT);
  volT=setTimeout(()=>fetch('/api/volume',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({volume:parseInt($('vol').value,10)})}).then(r=>r.json()).then(showVol),250);};

// browser mic: ONE persistent stream + analyser. Holding Talk just flips a flag
// (instant) and uploads on release; the analyser drives a live waveform.
let actx,stream,source,analyser,proc,chunks=[],capturing=false,micReady=false;
async function ensureMic(){
  if(micReady){ if(actx.state==='suspended'){try{await actx.resume();}catch(_){}} return true; }
  try{stream=await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,echoCancellation:true,noiseSuppression:true}});}
  catch(e){$('micst').textContent='mic blocked: '+e.message+'  (open via http://localhost)';return false;}
  actx=new (window.AudioContext||window.webkitAudioContext)();
  source=actx.createMediaStreamSource(stream);
  analyser=actx.createAnalyser();analyser.fftSize=1024;analyser.smoothingTimeConstant=0.6;
  proc=actx.createScriptProcessor(2048,1,1);
  proc.onaudioprocess=e=>{ if(capturing) chunks.push(new Float32Array(e.inputBuffer.getChannelData(0))); };
  source.connect(analyser);analyser.connect(proc);proc.connect(actx.destination);  // proc writes silence -> no feedback
  micReady=true;$('micst').textContent='mic ready';drawWave();
  return true;
}
function drawWave(){
  const c=$('wave');if(!c)return;const x=c.getContext('2d'),W=c.width,H=c.height,buf=new Uint8Array(analyser.fftSize);
  (function loop(){requestAnimationFrame(loop);analyser.getByteTimeDomainData(buf);
    x.clearRect(0,0,W,H);
    x.strokeStyle='rgba(255,255,255,.07)';x.lineWidth=1;x.beginPath();x.moveTo(0,H/2);x.lineTo(W,H/2);x.stroke();
    x.lineWidth=2;x.strokeStyle=capturing?'#e368c8':'#3ddc97';x.beginPath();
    for(let i=0;i<buf.length;i++){const v=buf[i]/128-1,px=i/(buf.length-1)*W,py=H/2+v*(H/2-3);i?x.lineTo(px,py):x.moveTo(px,py);}
    x.stroke();
  })();
}
async function startRec(){const ok=await ensureMic();if(!ok)return;chunks=[];capturing=true;$('talk').classList.add('rec');$('micst').textContent='listening…';}
async function stopRec(){
  if(!capturing)return;capturing=false;$('talk').classList.remove('rec');
  let n=chunks.reduce((a,c)=>a+c.length,0);if(!n){$('micst').textContent='(nothing captured)';return;}
  const buf=new Float32Array(n);let o=0;chunks.forEach(c=>{buf.set(c,o);o+=c.length;});
  const wav=encodeWav(downsample(buf,actx.sampleRate,16000),16000);$('micst').textContent='thinking…';
  try{const r=await fetch('/api/utterance',{method:'POST',headers:{'Content-Type':'audio/wav'},body:wav});
    const d=await r.json();$('micst').textContent=d.ok?('heard: '+(d.heard||'(?)')):('error: '+d.error);}
  catch(e){$('micst').textContent='upload failed: '+e.message;}
}
function downsample(b,inR,outR){if(outR>=inR)return b;const r=inR/outR,n=Math.floor(b.length/r),o=new Float32Array(n);for(let i=0;i<n;i++)o[i]=b[Math.floor(i*r)];return o;}
function encodeWav(s,rate){const b=new ArrayBuffer(44+s.length*2),v=new DataView(b);const w=(o,t)=>{for(let i=0;i<t.length;i++)v.setUint8(o+i,t.charCodeAt(i));};
  w(0,'RIFF');v.setUint32(4,36+s.length*2,true);w(8,'WAVE');w(12,'fmt ');v.setUint32(16,16,true);v.setUint16(20,1,true);v.setUint16(22,1,true);
  v.setUint32(24,rate,true);v.setUint32(28,rate*2,true);v.setUint16(32,2,true);v.setUint16(34,16,true);w(36,'data');v.setUint32(40,s.length*2,true);
  let o=44;for(let i=0;i<s.length;i++,o+=2){let x=Math.max(-1,Math.min(1,s[i]));v.setInt16(o,x<0?x*0x8000:x*0x7fff,true);}return b;}
const tk=$('talk');
tk.addEventListener('mousedown',e=>{e.preventDefault();startRec();});tk.addEventListener('mouseup',stopRec);tk.addEventListener('mouseleave',()=>capturing&&stopRec());
tk.addEventListener('touchstart',e=>{e.preventDefault();startRec();});tk.addEventListener('touchend',e=>{e.preventDefault();stopRec();});
$('arm').addEventListener('click',()=>{ensureMic();},{once:true});   // pre-warm so first Talk is instant
</script></body></html>"""
