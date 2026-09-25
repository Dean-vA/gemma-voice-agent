#!/usr/bin/env python3
"""
Hands-free presence-greeter for the G1 (audio-native, off-board inference).

This is the autonomous counterpart to the F1 push-to-talk path in
g1_gemma_client.py. When presence-mode is armed (F2), a background controller:

    1. Watches the USB camera for a frontal face that is CLOSE (proximity gate).
    2. Debounces it: a face present ~1s -> "armed"; absent ~2s -> "left" (and the
       greeted latch resets so the next arrival is greeted fresh).
    3. On a NEW arrival (and only if no conversation is already busy) fires the
       proven greeting call -- silent wav + camera frame + instruction -> the
       gateway returns a contextual, image-grounded greeting as kokoro audio,
       which we speak out the G1 head speaker.
    4. Then runs a hands-free conversation loop: webrtcvad segments the person's
       speech (energy-VAD numpy fallback if webrtcvad is unavailable), each
       utterance goes to /converse with a fresh frame, the reply is spoken, and
       it loops as long as the person is still present. Greets ONCE per arrival.

It NEVER publishes motion commands. It reuses the on-robot-verified I/O:
  - camera: the persistent OpenCV handle in usb_cam (same _get_cap()/_lock)
  - mic:    the USB input device discovered by usb_mic._find_usb_input()
  - speaker: AudioClient.PlayStream over DDS (via the wav_to_pcm16k helper)

Coordination with the F1 push-to-talk path is via a shared CONVERSATION_BUSY
event passed in from the client: presence won't greet while PTT is mid-turn, and
PTT is ignored while presence is mid-conversation. The busy event being held
during playback is also what stops the robot self-triggering off its own
speaker (we only open the VAD mic between our own utterances, after playback has
fully drained).

All thresholds are env-overridable so they can be tuned live without a rebuild.
"""

import io
import os
import time
import wave
import uuid
import base64
import json
import queue
import socket
import struct
import threading

import numpy as np
import cv2

import usb_cam   # persistent camera handle (_get_cap / _lock / grab_frame_jpeg)
import usb_mic   # USB mic device discovery (_find_usb_input)

# httpx ships in the greeter image (anthropic SDK dep).
import httpx

# webrtcvad is preferred; fall back to a numpy energy VAD if it's not installed.
try:
    import webrtcvad
    _HAVE_WEBRTCVAD = True
except Exception:
    webrtcvad = None
    _HAVE_WEBRTCVAD = False

# sounddevice for the hands-free mic stream (same lib usb_mic uses).
try:
    import sounddevice as sd
    _HAVE_SD = True
except Exception:
    sd = None
    _HAVE_SD = False

# Web control centre hooks (set by control_center.start_control_center). Both
# stay None when PRESENCE_WEB is off, so every _pub is a cheap no-op and the
# greeter behaves exactly as before.
HUB = None          # control_center.Hub — receives published state/metrics
PROMPTS = None      # control_center.PromptStore — live prompt presets
FRAME_SOURCE = None # callable -> latest BGR frame (set by control_center.Grabber);
                    # when set, the detector/turn snapshots read it instead of
                    # grabbing the camera directly (avoids capture backlog/lag).

def _pub(kind, **data):
    h = HUB
    if h is not None:
        try:
            h.publish(kind, data)
        except Exception:
            pass


# =============================== CONFIG ======================================
GEMMA_URL   = os.environ.get("GEMMA_URL", "http://localhost:2730").rstrip("/")
SESSION_ID  = os.environ.get("SESSION_ID", "")
TTS_ENGINE  = os.environ.get("TTS_ENGINE", "kokoro")     # only reachable engine
HTTP_TIMEOUT = float(os.environ.get("GEMMA_TIMEOUT", "60"))

# --- Vision / proximity ------------------------------------------------------
PROXIMITY_FRAC = float(os.environ.get("PRESENCE_PROX_FRAC", "0.18"))  # face_w / frame_w (~1.5-2m)
DETECT_FPS     = float(os.environ.get("PRESENCE_DETECT_FPS", "5"))    # Haar tick rate
HAAR_SCALE     = float(os.environ.get("PRESENCE_HAAR_SCALE", "1.2"))
HAAR_NEIGHBORS = int(os.environ.get("PRESENCE_HAAR_NEIGHBORS", "5"))
HAAR_MIN_PX    = int(os.environ.get("PRESENCE_HAAR_MIN_PX", "80"))    # min face px

# --- Debounce timing (seconds) ----------------------------------------------
ARM_SECS    = float(os.environ.get("PRESENCE_ARM_SECS", "1.0"))   # present -> armed
DISARM_SECS = float(os.environ.get("PRESENCE_DISARM_SECS", "2.0"))  # absent -> left
# Mid-conversation the visitor glances away, leans back, looks down: keep them
# on a looser leash (smaller face + longer grace) than the arrival gate uses.
STAY_FRAC        = float(os.environ.get("PRESENCE_STAY_FRAC", "0.12"))
CONV_DISARM_SECS = float(os.environ.get("PRESENCE_CONV_DISARM_SECS", "5.0"))
DEBUG       = os.environ.get("PRESENCE_DEBUG", "").strip().lower() in ("1", "true", "yes", "on")

# --- Hands-free VAD capture --------------------------------------------------
VAD_AGGR        = int(os.environ.get("PRESENCE_VAD_AGGR", "2"))     # 0..3 webrtcvad
VAD_FRAME_MS    = 30                                                # 10/20/30 only
ONSET_FRAMES    = int(os.environ.get("PRESENCE_ONSET_FRAMES", "3"))   # ~90ms voiced
SILENCE_MS      = float(os.environ.get("PRESENCE_SILENCE_MS", "800"))  # end-of-utt pause tolerance
PREROLL_FRAMES  = int(os.environ.get("PRESENCE_PREROLL", "5"))      # keep ~150ms lead
UTTER_MAX_SECS  = float(os.environ.get("PRESENCE_UTTER_MAX", "15"))
ONSET_TIMEOUT   = float(os.environ.get("PRESENCE_ONSET_TIMEOUT", "1.0"))  # poll cam
ENERGY_THRESH   = float(os.environ.get("PRESENCE_ENERGY_THRESH", "0.012"))  # fallback floor
# --- ambient noise auto-calibration -----------------------------------------
# The speech threshold is set to (measured room noise floor * NOISE_MULT), so it
# adapts to whatever room/mic the robot is in. Re-measured when presence arms and
# whenever the room empties. CALIB_PCTL picks how "quiet" a level counts as the
# floor (75th pct of frame RMS over the sample = robust to brief sounds).
NOISE_MULT  = float(os.environ.get("PRESENCE_NOISE_MULT", "1.7"))   # onset = floor*this
CONTINUE_MULT = float(os.environ.get("PRESENCE_CONTINUE_MULT", "1.3"))  # offset = floor*this
# End-of-utterance is adaptive: a frame counts as silence when its RMS falls
# within SIL_MARGIN of the quietest recent level seen DURING the utterance
# (the gaps between words reveal the true in-room floor). This self-calibrates
# per utterance, so end detection survives calibration drift / a noisier room.
SIL_MARGIN  = float(os.environ.get("PRESENCE_SIL_MARGIN", "1.8"))
MIN_FLOOR   = float(os.environ.get("PRESENCE_MIN_FLOOR", "0.010"))  # never below this
CALIB_SECS  = float(os.environ.get("PRESENCE_CALIB_SECS", "1.5"))
CALIB_PCTL  = float(os.environ.get("PRESENCE_CALIB_PCTL", "50"))    # median = true quiet level
# Periodic ambient recalibration: when the room is EMPTY (no one near) and idle,
# resample the floor every RECALIB_SECS and keep a rolling AVERAGE of the last
# RECALIB_AVG_N samples -> a stable floor that smooths transients and tracks slow
# drift. A short sample (RECALIB_SAMPLE_SECS) keeps the detection gap tiny. 0=off.
RECALIB_SECS        = float(os.environ.get("PRESENCE_RECALIB_SECS", "30"))
RECALIB_AVG_N       = int(os.environ.get("PRESENCE_RECALIB_AVG_N", "5"))
RECALIB_SAMPLE_SECS = float(os.environ.get("PRESENCE_RECALIB_SAMPLE_SECS", "0.8"))
# Use a floor slightly BELOW the measured average (a little headroom so the
# thresholds aren't set right at ambient -> catches quieter speech onset).
FLOOR_TRIM          = float(os.environ.get("PRESENCE_FLOOR_TRIM", "0.9"))
# webrtcvad has been unreliable on this mic (aliasing/noise); the calibrated
# energy gate is the dependable signal. Energy-only by default; opt back in with
# PRESENCE_USE_WEBRTCVAD=1.
USE_WEBRTCVAD = os.environ.get("PRESENCE_USE_WEBRTCVAD", "").strip().lower() in ("1", "true", "yes", "on")
QUIET_TURNS_END = int(os.environ.get("PRESENCE_QUIET_TURNS", "0"))  # 0 = stay til left
SPEAK_DRAIN_PAD = float(os.environ.get("PRESENCE_DRAIN_PAD", "0.25"))  # post-playback
# Transcript echo costs a separate ASR pass on the gateway (the model is
# audio-native and doesn't need it). Off by default for latency; PRESENCE_TRANSCRIBE=1
# turns it back on so the `heard:` line shows what the VAD captured.
CONV_TRANSCRIBE = os.environ.get("PRESENCE_TRANSCRIBE", "").strip().lower() in ("1", "true", "yes", "on")

# --- Instructions (env-overridable) -----------------------------------------
GREET_INSTRUCTION = os.environ.get(
    "PRESENCE_GREET_INSTRUCTION",
    "A person has just walked up to you. Give a short, warm, natural spoken "
    "greeting in one sentence, grounded in what you can see in the image. "
    "Don't ask them to do anything -- just welcome them.")
CONVERSE_INSTRUCTION = os.environ.get(
    "PRESENCE_CONVERSE_INSTRUCTION",
    "You are George, a friendly Unitree G1 robot talking to a person standing "
    "in front of you. Reply briefly and conversationally to what they just "
    "said, grounded in what you can see.")
GOODBYE_INSTRUCTION = os.environ.get(
    "PRESENCE_GOODBYE_INSTRUCTION",
    "The person you were talking with is walking away. Give a short, warm "
    "spoken goodbye in one sentence.")

def _prompt(kind):
    """Active instruction for `kind` in {greet, converse, goodbye}. Reads the
    live PromptStore preset when the web UI is enabled (hot-reload, no restart),
    else the env-seeded constants above."""
    store = PROMPTS
    if store is not None:
        try:
            p = store.active() or {}
            v = p.get(kind)
            if v:
                return v
        except Exception:
            pass
    return {"greet": GREET_INSTRUCTION, "converse": CONVERSE_INSTRUCTION,
            "goodbye": GOODBYE_INSTRUCTION}[kind]

# --- Goodbye + wave when the person leaves ----------------------------------
# The wave uses the dedicated G1ArmActionClient (ExecuteAction) -- a predefined
# arm motion that does NOT seize the locomotion lease (robot keeps standing,
# stays under the hand controller). Gated to a standing arm-action mode and
# always released afterward. Action IDs verified on-robot in g1_greeter_shadow.
WAVE_ON_LEAVE  = os.environ.get("PRESENCE_WAVE", "1").strip().lower() in ("1", "true", "yes", "on")
WAVE_ACTION_ID = int(os.environ.get("PRESENCE_WAVE_ACTION", "25"))   # wave_under_head
ARM_RELEASE_ID = 99                                                  # release_arm
SAFE_MODES     = {5}                                                 # standing, gesture-ready
WAVE_HOLD_SECS = float(os.environ.get("PRESENCE_WAVE_HOLD", "2.5"))

_RATE_16K = 16000
_RATE_48K = 48000
_FRAME_16K = int(_RATE_16K * VAD_FRAME_MS / 1000)   # 480 samples = 30ms @16k
_FRAME_48K = _FRAME_16K * 3                          # 1440 samples = 30ms @48k
MIC_READ_TIMEOUT = float(os.environ.get("PRESENCE_MIC_READ_TIMEOUT", "1.0"))  # 30ms frames
# Mic source: "usb" (USB dongle via PortAudio) or "array" (the G1's built-in
# 4-mic array: 16k mono s16le on UDP multicast, only while voice-service "mic
# mode" is on -- enabled via voice API 1008 {"mode": 1}). The array has ~4x lower
# noise floor in a noisy hall; falls back to USB if its stream stalls.
MIC_SOURCE       = os.environ.get("PRESENCE_MIC", "usb").strip().lower()
ARRAY_GROUP      = "239.168.123.161"
ARRAY_PORT       = 5555
ARRAY_LOCAL_IP   = os.environ.get("PRESENCE_ARRAY_LOCAL_IP", "192.168.123.164")  # eth0
ARRAY_MIN_FLOOR  = float(os.environ.get("PRESENCE_ARRAY_MIN_FLOOR", "0.004"))
ARRAY_STALL_FALLBACK = 2          # consecutive array stalls before falling back to USB
# The array hears the whole room (distant chatter), so its onset sits higher
# above the floor than the USB mic's: onset = floor * NOISE_MULT * this.
ARRAY_ONSET_BOOST = float(os.environ.get("PRESENCE_ARRAY_ONSET_BOOST", "1.5"))
_VOICE_API_MIC_MODE = 1008        # undocumented voice-service API (mode 1 on, 2 off)


def enable_array_mic(audio_client):
    """Turn on the G1 voice service's mic mode so the array streams. Idempotent;
    doesn't survive a robot power cycle, so call on every arm."""
    try:
        audio_client._RegistApi(_VOICE_API_MIC_MODE, 0)
        code, _ = audio_client._Call(_VOICE_API_MIC_MODE, json.dumps({"mode": 1}))
        print(f"[presence] array mic mode on -> {code}")
        return code == 0
    except Exception as e:
        print(f"[presence] array mic mode failed: {e}")
        return False


class MicStall(RuntimeError):
    """The mic stream stopped delivering audio (wedged USB device)."""


# =========================== small audio helpers =============================
def _silent_wav_bytes(secs=0.3, rate=_RATE_16K):
    """0.3s of silence as 16-bit PCM WAV -- satisfies /converse's required audio."""
    n = int(rate * secs)
    pcm = np.zeros(n, dtype="<i2")
    bio = io.BytesIO()
    w = wave.open(bio, "wb")
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
    w.writeframes(pcm.tobytes()); w.close()
    return bio.getvalue()


def _float_to_wav_bytes(audio, rate=_RATE_16K):
    """float32 mono [-1,1] -> in-memory 16-bit PCM WAV bytes for upload."""
    pcm16 = np.clip(audio * 32768.0, -32768, 32767).astype("<i2")
    bio = io.BytesIO()
    w = wave.open(bio, "wb")
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
    w.writeframes(pcm16.tobytes()); w.close()
    return bio.getvalue()


def _extract_audio_b64(payload):
    """The /converse 'audio' event carries the wav under one of these keys."""
    for k in ("wav_base64", "audio", "data", "wav", "pcm"):
        v = payload.get(k)
        if isinstance(v, str) and v:
            return v
    return None


# =============================== detector ====================================
class FaceDetector:
    """Frontal-face Haar detector reading the shared persistent camera handle.

    Reads RAW frames via usb_cam's _get_cap()/_lock so it never fights the JPEG
    grab used for the greeting frame -- same handle, same lock."""

    def __init__(self):
        path = os.path.join(cv2.data.haarcascades,
                            "haarcascade_frontalface_default.xml")
        self.cascade = cv2.CascadeClassifier(path)
        if self.cascade.empty():
            raise RuntimeError(f"failed to load Haar cascade at {path}")
        # local (tiled) contrast, not global equalizeHist: a big bright wall
        # dominates the global histogram and crushes the face to mush.
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # last-detection stats (for the debug line / frame dump)
        self.last_nfaces = 0
        self.last_wh = (0, 0)
        self.last_frame = None
        self.last_faces = ()

    def _grab_bgr(self):
        # Prefer the shared grabber's freshest frame when the web stack is active
        # (no direct camera read -> no buffer backlog). Falls back to a direct
        # grab on the robot/headless path where FRAME_SOURCE is unset.
        if FRAME_SOURCE is not None:
            try:
                return FRAME_SOURCE()
            except Exception:
                return None
        with usb_cam._lock:
            cap = usb_cam._get_cap()
            if cap is None:
                return None
            cap.grab()                  # flush the single-frame buffer
            ok, frame = cap.read()
        if not ok or frame is None:
            return None
        return frame

    def nearest_face_frac(self):
        """Return (face_width / frame_width) of the largest frontal face in
        view, or 0.0 if none. >0 means a face is present; the magnitude is the
        proximity signal used by the gate."""
        frame = self._grab_bgr()
        if frame is None:
            self.last_nfaces, self.last_frame, self.last_faces = 0, None, ()
            return 0.0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = self.clahe.apply(gray)          # normalize contrast for Haar (backlit-safe)
        faces = self.cascade.detectMultiScale(
            gray, scaleFactor=HAAR_SCALE, minNeighbors=HAAR_NEIGHBORS,
            minSize=(HAAR_MIN_PX, HAAR_MIN_PX))
        h, w = gray.shape[:2]
        self.last_nfaces = len(faces)
        self.last_wh = (w, h)
        self.last_frame = frame
        self.last_faces = faces
        if len(faces) == 0:
            return 0.0
        widest = max(int(fw) for (_x, _y, fw, _h) in faces)
        return widest / float(w)

    def dump_annotated(self, path, frac):
        """Write the last grabbed frame with face boxes drawn, for inspection."""
        if self.last_frame is None:
            return
        img = _draw_overlay(self.last_frame, self.last_faces, frac,
                            {"thr": PROXIMITY_FRAC, "nfaces": self.last_nfaces})
        try:
            cv2.imwrite(path, img)
        except Exception:
            pass


def _draw_overlay(frame, faces, frac, state):
    """Draw face boxes + a status line on a copy of `frame`. Shared by the debug
    dump and the web FramePump so the live feed and the saved frames match."""
    img = frame.copy()
    thr = (state or {}).get("thr", PROXIMITY_FRAC)
    frame_w = img.shape[1]
    # `faces` is a numpy ndarray from detectMultiScale -- never use `faces or ()`
    # (bool() on a non-empty array raises "ambiguous truth value").
    flist = [] if faces is None else list(faces)
    for f in flist:
        x, y, fw, fh = int(f[0]), int(f[1]), int(f[2]), int(f[3])
        close = (fw / float(frame_w)) >= thr
        col = (0, 255, 0) if close else (0, 180, 255)
        cv2.rectangle(img, (x, y), (x + fw, y + fh), col, 2)
    s = state or {}
    n = s.get("nfaces", len(flist))
    status = "ARMED" if s.get("enabled") else "idle"
    extra = []
    if s.get("greeted"): extra.append("greeted")
    if s.get("busy"):    extra.append("talking")
    # human-readable: "1 face  ·  24% close (need 18%)  ·  ARMED  ·  greeted"
    label = (f"{n} face{'' if n == 1 else 's'}  -  {frac*100:.0f}% close "
             f"(need {thr*100:.0f}%)  -  {status}")
    if extra:
        label += "  -  " + ", ".join(extra)
    cv2.putText(img, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 120), 1, cv2.LINE_AA)
    return img


def _grab_jpeg():
    """JPEG of the current camera frame for a turn image. Uses the shared
    grabber's freshest frame when the web stack is active, else a direct grab."""
    src = FRAME_SOURCE
    if src is not None:
        try:
            f = src()
            if f is None:
                return None
            ok, jpg = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 60])
            return jpg.tobytes() if ok else None
        except Exception:
            return None
    return usb_cam.grab_frame_jpeg(None)


# =============================== VAD capture =================================
class VadCapture:
    """Holds a 48k mono mic stream open and segments utterances with webrtcvad
    (energy-VAD fallback). Decimates 48k->16k by [::3] to match usb_mic, and
    normalizes each utterance to 0.5 peak (the proven, ASR-friendly level)."""

    def __init__(self):
        self.vad = webrtcvad.Vad(VAD_AGGR) if _HAVE_WEBRTCVAD else None
        self.stream = None
        self.device = usb_mic._find_usb_input() if _HAVE_SD else None
        self.noise_floor = None                 # measured ambient RMS (averaged)
        self.speech_thresh = ENERGY_THRESH       # onset floor (set by calibrate)
        self.continue_thresh = ENERGY_THRESH     # offset floor (lower, hysteresis)
        self.floor_hist = []                     # recent per-calibration floors (rolling avg)
        self._degraded = False                   # mic wedged -> disabled (greet-only)
        self.source = "array" if MIC_SOURCE == "array" else "usb"
        self.min_floor = ARRAY_MIN_FLOOR if self.source == "array" else MIN_FLOOR
        self.need_recalib = False                # set on source fallback
        self._array_stalls = 0
        self._sock = None
        self._stop_rx = None
        print(f"[presence] mic source: {self.source}"
              + (f" (usb fallback device={self.device})" if self.source == "array" else
                 f" (device={self.device})"))

    @property
    def available(self):
        if self._degraded:
            return False
        if self.source == "array":
            return True
        return _HAVE_SD and self.device is not None

    def set_source(self, source):
        """Switch mic source at runtime (control centre). Takes effect on the next
        listen; the controller recalibrates since floors differ ~4x between mics."""
        source = (source or "").strip().lower()
        if source not in ("array", "usb"):
            return False, "source must be 'array' or 'usb'"
        if source == "usb" and not (_HAVE_SD and self.device is not None):
            return False, "no USB mic found"
        if source != self.source:
            self.source = source
            self.min_floor = ARRAY_MIN_FLOOR if source == "array" else MIN_FLOOR
            self.floor_hist = []
            self._array_stalls = 0
            self.need_recalib = True
            print(f"[presence] mic source -> {source}")
        return True, source

    def _note_stall(self):
        """Count array stalls; after a few in a row, fall back to the USB mic
        (and ask the controller to recalibrate -- the array's thresholds are far
        too low for the noisier USB mic)."""
        if self.source != "array":
            return
        self._array_stalls += 1
        if (self._array_stalls >= ARRAY_STALL_FALLBACK and _HAVE_SD
                and self.device is not None):
            print("[presence] array mic stalled repeatedly -> falling back to USB mic")
            self.source = "usb"
            self.min_floor = MIN_FLOOR
            self.floor_hist = []
            self.need_recalib = True

    def mark_degraded(self):
        """Disable the mic after a hung calibration/read so hands-free listening
        stands down (greet-only) and NOTHING waits on the mic again. Face
        detection is independent and keeps running. Recovers on restart."""
        self._degraded = True

    def open(self):
        # Availability check only. The real mic stream is opened per listen in
        # next_utterance and closed before the network/playback step, so it never
        # sits idle (an idle PortAudio input stream overflows and the next read
        # wedges -- which froze the conversation on the 2nd turn).
        return self.available

    def _open_stream(self):
        # Callback mode feeding a queue, so every read has a timeout. A blocking
        # stream.read() on a wedged USB mic never returns and froze the whole
        # presence loop (detection included) mid-recording.
        q = queue.Queue()
        self._q = q
        self._open_src = self.source        # a runtime switch can't change an open stream
        if self._open_src == "array":
            self._open_array(q)
            return
        def _cb(indata, frames, t, status):
            q.put(indata.copy())
        self.stream = sd.InputStream(
            samplerate=_RATE_48K, channels=1, dtype="int16",
            device=self.device, blocksize=_FRAME_48K, callback=_cb)
        self.stream.start()

    def _open_array(self, q):
        """Join the array multicast group; a reader thread re-frames the packets
        (5120 B each) into 30ms / 480-sample int16 frames on `q`."""
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("", ARRAY_PORT))
        s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP,
                     struct.pack("4s4s", socket.inet_aton(ARRAY_GROUP),
                                 socket.inet_aton(ARRAY_LOCAL_IP)))
        s.settimeout(0.2)
        stop = threading.Event()
        self._sock, self._stop_rx = s, stop
        nbytes = _FRAME_16K * 2
        def _rx():
            buf = bytearray()
            while not stop.is_set():
                try:
                    buf += s.recv(8192)
                except socket.timeout:
                    continue
                except OSError:
                    return
                while len(buf) >= nbytes:
                    q.put(np.frombuffer(bytes(buf[:nbytes]), dtype=np.int16).copy())
                    del buf[:nbytes]
        threading.Thread(target=_rx, daemon=True).start()

    def _array_sample_rms(self, secs):
        """Per-30ms-frame RMS of `secs` of array audio, on a private socket."""
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("", ARRAY_PORT))
            s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP,
                         struct.pack("4s4s", socket.inet_aton(ARRAY_GROUP),
                                     socket.inet_aton(ARRAY_LOCAL_IP)))
            s.settimeout(MIC_READ_TIMEOUT)
            buf, t0, i = bytearray(), time.time(), 0
            while time.time() - t0 < secs:
                try:
                    buf += s.recv(8192)
                except socket.timeout:
                    return []
            a = np.frombuffer(bytes(buf[:len(buf) // 2 * 2]), dtype=np.int16).astype(np.float32) / 32768.0
            fr = a[: len(a) // _FRAME_16K * _FRAME_16K].reshape(-1, _FRAME_16K)
            rms = np.sqrt((fr * fr).mean(axis=1))
            for j in range(0, len(rms), 3):
                self._pub_vad(level=float(rms[j]), phase="calibrating")
            return [float(x) for x in rms]
        except OSError as e:
            print(f"[presence] array calib socket failed: {e}")
            return []
        finally:
            s.close()

    def _close_stream(self):
        if self._sock is not None:
            self._stop_rx.set()
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        if self.stream is not None:
            s, self.stream = self.stream, None
            def _shut():
                try:
                    s.abort(); s.close()
                except Exception:
                    pass
            # a wedged device can hang close() too: never let it block us
            w = threading.Thread(target=_shut, daemon=True)
            w.start()
            w.join(2.0)

    def close(self):
        self._close_stream()

    def _pub_vad(self, level=None, phase="idle", voiced=None, end=None, active=True):
        """Push live mic level + calibration thresholds to the web Hub so the
        control centre can draw a VAD meter. Works over plain HTTP (server-pushed
        via /state) -- unlike the browser waveform, which needs a secure context.
        Cheap; called throttled from the calibrate/listen loops."""
        _pub("vad",
             floor=(None if self.noise_floor is None else round(self.noise_floor, 4)),
             onset=round(self.speech_thresh, 4),
             offset=round(self.continue_thresh, 4),
             level=(None if level is None else round(level, 4)),
             end=(None if end is None else round(end, 4)),
             voiced=voiced, active=active, phase=phase,
             samples=len(self.floor_hist))

    def _apply_floor(self, floor):
        """Fold a freshly-sampled floor into the rolling average and recompute the
        onset/offset thresholds from the average (not the single sample)."""
        self.floor_hist.append(floor)
        if len(self.floor_hist) > RECALIB_AVG_N:
            self.floor_hist = self.floor_hist[-RECALIB_AVG_N:]
        self.noise_floor = float(np.mean(self.floor_hist))       # true measured floor
        # FLOOR_TRIM lowers ONLY the onset (catch a quieter speech start). The
        # offset / end threshold stays on the true floor -- trimming it too made
        # end-of-speech never fire in a quiet room ("mic won't turn off").
        self.speech_thresh = max(self.min_floor, self.noise_floor * self.onset_mult() * FLOOR_TRIM)
        self.continue_thresh = max(self.min_floor * 0.8, self.noise_floor * CONTINUE_MULT)
        return self.noise_floor

    def calibrate(self, secs=CALIB_SECS, reset=True):
        """Sample ~`secs` of ambient room noise, fold it into the rolling average,
        and set the thresholds from the average. Returns (noise_floor, speech_thresh).

        `reset=True` (manual button / on-arm) starts a fresh average from this one
        sample; `reset=False` (periodic recalibration) appends to the running
        average so it smooths transients and tracks drift. Opens its own
        short-lived stream; call from the controller thread when the room is quiet."""
        if not self.available:
            return None
        if self.source == "array":
            # Own socket, not self._sock/_q: the array multicast takes many
            # listeners, so this can run live, even alongside an active listen.
            rmss = self._array_sample_rms(secs)
            if not rmss:
                print("[presence] calib: no array audio -> keeping previous thresholds")
                self._note_stall()
                return None
            if reset:
                self.floor_hist = []
            self._apply_floor(float(np.percentile(rmss, CALIB_PCTL)))
            self._pub_vad(level=self.noise_floor, phase="idle", active=False)
            return self.noise_floor, self.speech_thresh
        try:
            self._open_stream()
        except Exception as e:
            print(f"[presence] calib mic open failed: {e}")
            return None
        try:
            rmss = []
            t0 = time.time()
            i = 0
            while (time.time() - t0) < secs:
                f = self._read_frame16().astype(np.float32) / 32768.0
                rms = float(np.sqrt(np.mean(f * f)))
                rmss.append(rms)
                if i % 3 == 0:
                    self._pub_vad(level=rms, phase="calibrating")
                i += 1
        except MicStall as e:
            print(f"[presence] calib: {e} -> keeping previous thresholds")
            self._note_stall()
            return None
        finally:
            self._close_stream()
        if not rmss:
            return None
        if reset:
            self.floor_hist = []
        floor = float(np.percentile(rmss, CALIB_PCTL))
        self._apply_floor(floor)
        self._pub_vad(level=self.noise_floor, phase="idle", active=False)
        return self.noise_floor, self.speech_thresh

    def onset_mult(self):
        """Onset multiplier for the current source (read live: slider-tunable)."""
        return NOISE_MULT * (ARRAY_ONSET_BOOST if self.source == "array" else 1.0)

    def _fold_utterance_floor(self, frames16):
        """Learn the floor from a finished recording: its quietest 10% of frames
        are the gaps between words (or the crowd bed). In a crowd every frame can
        read as 'voiced', which starves _update_floor (it only sees non-voiced
        frames) and left the array's floor stuck at a quiet-moment calibration."""
        if len(frames16) < 10:
            return
        r = [float(np.sqrt(np.mean((f.astype(np.float32) / 32768.0) ** 2))) for f in frames16]
        floor = float(np.percentile(r, 10))
        self.noise_floor = floor if self.noise_floor is None else 0.7 * self.noise_floor + 0.3 * floor
        self.speech_thresh = max(self.min_floor, self.noise_floor * self.onset_mult())
        self.continue_thresh = max(self.min_floor * 0.8, self.noise_floor * CONTINUE_MULT)

    def _update_floor(self, ambient_rmss):
        """Nudge the threshold from observed silence (onset windows with no
        speech), so it tracks slow drift between calibrations. EMA, gentle."""
        if not ambient_rmss:
            return
        floor = float(np.percentile(ambient_rmss, CALIB_PCTL))
        if self.noise_floor is None:
            self.noise_floor = floor
        else:
            self.noise_floor = 0.8 * self.noise_floor + 0.2 * floor
        self.speech_thresh = max(self.min_floor, self.noise_floor * self.onset_mult())
        self.continue_thresh = max(self.min_floor * 0.8, self.noise_floor * CONTINUE_MULT)
        self._pub_vad(active=False, phase="idle")

    def _read_frame16(self):
        """One 30ms frame: read 48k int16 and downsample 3x -> 480 samples @16k.

        Uses a 3-tap box filter (mean of each triplet), NOT a naive stride: the
        stride aliases 48k mic noise into the speech band and makes webrtcvad
        read silence as speech (so utterances never end). The box filter is a
        cheap anti-alias low-pass that fixes that."""
        try:
            block = self._q.get(timeout=MIC_READ_TIMEOUT)
        except queue.Empty:
            raise MicStall(f"no {self._open_src} mic audio for {MIC_READ_TIMEOUT:.1f}s")
        if self._open_src == "array":
            self._array_stalls = 0
            return block                               # already 480 samples @16k
        a48 = block.reshape(-1).astype(np.int32)
        n = (len(a48) // 3) * 3
        a16 = a48[:n].reshape(-1, 3).mean(axis=1)
        return a16.astype(np.int16)                 # 480 samples = 30ms @16k

    def _is_voiced(self, frame16, thresh):
        """Voiced = RMS above `thresh` (AND webrtcvad if enabled). The energy gate
        is what makes end-of-utterance reliable: when the talker stops (or leaves)
        the RMS collapses below the floor and the frame reads as silence
        regardless of how the VAD classifies the residual noise.

        Onset uses the higher `speech_thresh` (won't false-start on noise); the
        record loop uses the lower `continue_thresh` (won't cut off a quiet
        syllable). That hysteresis is what de-twitches the energy VAD."""
        f = frame16.astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(f * f)))
        if rms < thresh:
            return False
        if USE_WEBRTCVAD and self.vad is not None:
            try:
                return self.vad.is_speech(frame16.tobytes(), _RATE_16K)
            except Exception:
                pass
        return True

    def next_utterance(self, still_active):
        """Wait (up to ONSET_TIMEOUT, re-checking still_active()) for speech
        onset, then record until SILENCE_MS of trailing silence.

        Returns ('speech', float32@16k), ('timeout', None) if no onset within the
        window (so the caller can re-poll the camera), or ('left', None) if
        still_active() went False while waiting.

        Opens a fresh mic stream for this listen and closes it before returning,
        so the stream is never left idle during the gateway/playback step."""
        if not self.available:
            return ("left", None)
        try:
            self._open_stream()
        except Exception as e:
            print(f"[presence] mic stream open failed: {e}")
            return ("left", None)
        try:
            silence_frames = max(1, int(SILENCE_MS / VAD_FRAME_MS))
            preroll = []                                # rolling lead-in buffer
            ambient = []                                # non-speech RMS this window
            voiced_run = 0
            t0 = time.time()

            # --- onset: wait for ONSET_FRAMES consecutive voiced frames -----
            onset_i = 0
            while True:
                if not still_active():
                    return ("left", None)
                f = self._read_frame16()
                preroll.append(f)
                if len(preroll) > PREROLL_FRAMES:
                    preroll.pop(0)
                ff = f.astype(np.float32) / 32768.0
                rms = float(np.sqrt(np.mean(ff * ff)))
                voiced = self._is_voiced(f, self.speech_thresh)   # onset: high thresh
                if voiced:
                    voiced_run += 1
                else:
                    voiced_run = 0
                    ambient.append(rms)
                if onset_i % 3 == 0:
                    self._pub_vad(level=rms, phase="listening", voiced=voiced)
                onset_i += 1
                if voiced_run >= ONSET_FRAMES:
                    # Refresh the floor from the silence right before this
                    # utterance, so the offset threshold tracks the CURRENT
                    # conversational noise (fixes "records too long" when the
                    # room is noisier than it was at calibration time).
                    if len(ambient) >= 5:
                        self._update_floor(ambient)
                    break
                if (time.time() - t0) > ONSET_TIMEOUT:
                    self._update_floor(ambient)         # learn the room's quiet level
                    return ("timeout", None)

            if DEBUG:
                print("[presence][dbg] vad: onset detected -> recording")

            # --- record: until trailing silence or hard cap -----------------
            collected = list(preroll)                   # keep the lead-in
            silence_run = 0
            rec_t0 = time.time()
            recent = []                                 # recent frame RMS (~1s)
            rec_win = max(20, int(1000 / VAD_FRAME_MS))
            end_thresh = self.continue_thresh
            rec_i = 0
            while True:
                f = self._read_frame16()
                collected.append(f)
                ff = f.astype(np.float32) / 32768.0
                rms = float(np.sqrt(np.mean(ff * ff)))
                recent.append(rms)
                if len(recent) > rec_win:
                    recent.pop(0)
                # adaptive floor: quietest recent level (10th pct of the window),
                # but never below the calibrated continue threshold.
                noise_est = float(np.percentile(recent, 10))
                end_thresh = max(self.continue_thresh, noise_est * SIL_MARGIN)
                over = rms > end_thresh
                if over:
                    silence_run = 0
                else:
                    silence_run += 1
                if rec_i % 3 == 0:
                    self._pub_vad(level=rms, phase="recording", voiced=over, end=end_thresh)
                rec_i += 1
                if silence_run >= silence_frames:
                    break
                if (time.time() - rec_t0) > UTTER_MAX_SECS:
                    break

            self._fold_utterance_floor(collected)
            audio = np.concatenate(collected).astype(np.float32) / 32768.0
            peak = float(np.abs(audio).max()) or 1.0
            if DEBUG:
                fl = (len(audio) // _FRAME_16K) * _FRAME_16K
                fr = audio[:fl].reshape(-1, _FRAME_16K)
                rmss = np.sqrt((fr * fr).mean(axis=1)) if len(fr) else np.array([0.0])
                print(f"[presence][dbg] utt: raw_peak={peak:.3f} onset={self.speech_thresh:.4f} "
                      f"end={end_thresh:.4f} cont={self.continue_thresh:.4f} "
                      f"floor={self.noise_floor if self.noise_floor is None else round(self.noise_floor,4)} "
                      f"dur={len(audio)/_RATE_16K:.2f}s | frame_rms "
                      f"min={rmss.min():.4f} p10={np.percentile(rmss,10):.4f} "
                      f"median={np.median(rmss):.4f} max={rmss.max():.4f}")
            audio = audio * (0.5 / peak)                # proven normalize, no clip
            return ("speech", np.clip(audio, -1.0, 1.0))
        except MicStall as e:
            # drop this listen; the caller re-polls presence and the next listen
            # opens a fresh stream, which usually recovers the device.
            print(f"[presence] mic stalled ({e}) -> reopening next listen")
            self._note_stall()
            return ("timeout", None)
        finally:
            self._close_stream()
            self._pub_vad(active=False, phase="idle")


# ========================= /converse turn (with instruction) =================
def _converse_turn(audio_client, wav_bytes, image_bytes, instruction,
                   set_led, wav_to_pcm16k, gain, seq_ref, transcribe=True,
                   session_id=None, phase="converse"):
    """POST audio (+frame) to /converse with a per-call instruction, play each
    kokoro 'audio' SSE event as it arrives, and BLOCK until playback fully
    drains (so we never capture our own speaker). Returns (heard, reply).

    transcribe=False for the silent-wav greeting (nothing to transcribe; avoids
    the gateway echoing the instruction back as a bogus 'heard' and the extra
    latency)."""
    url   = f"{GEMMA_URL}/converse"
    files = {"audio": ("speech.wav", wav_bytes, "audio/wav")}
    data  = {"engine": TTS_ENGINE}
    if transcribe:
        data["transcribe"] = "true"
    if instruction:
        data["instruction"] = instruction
    sid = session_id or SESSION_ID
    if sid:
        data["session_id"] = sid     # gateway threads conversation history per session
    if image_bytes:
        files["image"] = ("frame.jpg", image_bytes, "image/jpeg")

    t0 = time.perf_counter()
    _pub("turn", ev="start", phase=phase, session=sid,
         image_jpeg=image_bytes, audio_bytes=len(wav_bytes),
         audio_secs=round(len(wav_bytes) / 2 / _RATE_16K, 2))
    marks = {"first_token": False, "first_audio": False}

    say_q = queue.Queue()
    DONE = object()

    def worker():
        while True:
            item = say_q.get()
            if item is DONE:
                return
            try:
                # write the temp wav only now, right before playing: queueing a
                # shared path let later sentences overwrite earlier queued ones
                # (skipped one sentence, repeated the next).
                pcm, dur = wav_to_pcm16k(_wav_path(item), gain)   # handles 24k(kokoro)->16k
                seq_ref[0] += 1
                audio_client.PlayStream("gemma", str(seq_ref[0]), pcm)
                time.sleep(dur + 0.2)
            except Exception as e:
                print(f"[presence] play failed: {e}")

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    reply_parts, heard = [], ""
    done_metrics = {}
    spoke = {"led": False}
    payload = {}
    try:
        with httpx.stream("POST", url, files=files, data=data,
                          timeout=HTTP_TIMEOUT) as r:
            r.raise_for_status()
            event = None
            for line in r.iter_lines():
                if not line:
                    event = None
                    continue
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    try:
                        payload = json.loads(line[5:].strip())
                    except Exception:
                        continue
                    if event == "transcript":
                        heard = payload.get("text", "")
                        if not marks["first_token"]:
                            marks["first_token"] = True
                            _pub("turn", ev="first_token",
                                 ms=(time.perf_counter() - t0) * 1000)
                        _pub("turn", ev="heard", heard=heard)   # show live
                        print(f"  heard: {heard!r}")
                    elif event == "token":
                        tok = payload.get("text", "")
                        if tok:
                            if not marks["first_token"]:
                                marks["first_token"] = True
                                _pub("turn", ev="first_token",
                                     ms=(time.perf_counter() - t0) * 1000)
                            reply_parts.append(tok)
                            _pub("turn", ev="reply", reply="".join(reply_parts))  # stream
                    elif event == "audio":
                        b64 = _extract_audio_b64(payload)
                        if b64:
                            if not marks["first_audio"]:
                                marks["first_audio"] = True
                                _pub("turn", ev="first_audio",
                                     ms=(time.perf_counter() - t0) * 1000)
                                _pub("speaking", on=True)   # start playback animation
                            if not spoke["led"]:
                                set_led(audio_client, 0, 80, 160)  # blue: speaking
                                spoke["led"] = True
                            try:
                                say_q.put(base64.b64decode(b64))
                            except Exception as e:
                                print(f"[presence] audio decode failed: {e}")
                    elif event == "done":
                        m = payload.get("metrics") or {}
                        done_metrics = m
                        # publish the per-component latencies NOW (gateway is
                        # done generating) so the boxes fill in while the robot
                        # is still speaking, not after playback drains.
                        _pub("turn", ev="metrics", metrics=m)
                        if m.get("ttft_ms") is not None:
                            print(f"  [gateway] TTFT={m['ttft_ms']}ms "
                                  f"backend={m.get('backend')}")
    finally:
        say_q.put(DONE)
        t.join()                       # block until ALL queued audio has played
        _pub("speaking", on=False)     # stop playback animation
        time.sleep(SPEAK_DRAIN_PAD)    # let the speaker tail decay before we listen

    reply = "".join(reply_parts)
    _pub("turn", ev="done", metrics=done_metrics, heard=heard, reply=reply,
         total_ms=(time.perf_counter() - t0) * 1000)
    return heard, reply


def _wav_path(wav_bytes):
    """Persist a wav payload to a temp file (wav_to_pcm16k takes a path)."""
    import tempfile
    path = os.path.join(tempfile.gettempdir(), "g1_presence_say.wav")
    with open(path, "wb") as f:
        f.write(wav_bytes)
    return path


# ============================ controller =====================================
class PresenceController(threading.Thread):
    """Background presence-mode loop. start()/stop() the thread; set enabled via
    enable()/disable()/toggle() (F2). Holds `busy` during a greeting+conversation
    so the F1 PTT path stands down and we never self-trigger."""

    def __init__(self, audio_client, busy_event, set_led, wav_to_pcm16k, gain,
                 seq_ref, arm=None, mode_fn=None):
        super().__init__(daemon=True)
        self.audio = audio_client
        self.busy = busy_event
        self.set_led = set_led
        self.wav_to_pcm16k = wav_to_pcm16k
        self.gain = gain
        self.seq_ref = seq_ref
        self.arm = arm              # G1ArmActionClient for the wave (or None)
        self.mode_fn = mode_fn      # () -> mode_machine, for the gesture gate

        self.enabled = False
        self._stop = threading.Event()

        self.detector = FaceDetector()
        self.vad = VadCapture()

        # debounce state
        self.first_seen = None      # when the current continuous presence began
        self.last_seen = 0.0        # last time a qualifying face was seen
        self.greeted = False        # one greeting per arrival
        self._last_dbg = 0.0        # throttle for the debug line
        self._recalib = True        # measure the room's noise floor when idle
        self._present = False       # last tick: was a close face in view?
        self._last_recalib = 0.0    # last ambient (re)calibration time

    # -- external controls ----------------------------------------------------
    def enable(self):
        self.enabled = True
        self._recalib = True                       # recalibrate to this room on arm
        print("[presence] ENABLED (watching for arrivals)")
        self.set_led(self.audio, 0, 60, 30)        # dim green: armed/watching
        _pub("presence", enabled=True)

    def disable(self):
        self.enabled = False
        self._reset_arrival()
        print("[presence] DISABLED")
        self.set_led(self.audio, 20, 20, 20)       # dim: off
        _pub("presence", enabled=False, greeted=False)

    def toggle(self):
        self.disable() if self.enabled else self.enable()

    def stop(self):
        self._stop.set()

    # -- internals ------------------------------------------------------------
    def _reset_arrival(self):
        self.first_seen = None
        self.last_seen = 0.0
        self.greeted = False

    def _present_now(self, frac=PROXIMITY_FRAC):
        """True if a close-enough frontal face is in view right now."""
        return self.detector.nearest_face_frac() >= frac

    def _debug(self, now, frac):
        if not DEBUG or (now - self._last_dbg) < 1.0:
            return
        self._last_dbg = now
        if self.first_seen is None:
            armed_in = "-"
        else:
            armed_in = f"{max(0.0, ARM_SECS - (now - self.first_seen)):.1f}s"
        w, h = self.detector.last_wh
        print(f"[presence][dbg] face_frac={frac:.3f} thr={PROXIMITY_FRAC:.2f} "
              f"nfaces={self.detector.last_nfaces} frame={w}x{h} "
              f"present={frac >= PROXIMITY_FRAC} greeted={self.greeted} "
              f"armed_in={armed_in} busy={self.busy.is_set()}")
        self.detector.dump_annotated("/tmp/presence_dbg.jpg", frac)

    def run(self):
        period = 1.0 / max(1.0, DETECT_FPS)
        if not self.vad.available:
            print("[presence] WARNING: no USB mic for VAD; greet-only "
                  "(hands-free conversation disabled)")
        while not self._stop.is_set():
            if not self.enabled or self.busy.is_set():
                time.sleep(period)
                continue
            # Calibration is BOUNDED and off the detection critical path. A wedged
            # USB mic can make calibrate() block forever; we must never let that
            # stall face detection. _run_calib caps the wait and disables the mic
            # (greet-only) if it hangs, so _tick() below ALWAYS runs.
            if self.vad.available:
                if self.vad.need_recalib:              # mic source fell back
                    self.vad.need_recalib = False
                    self._recalib = True
                if self._recalib:
                    self._recalib = False
                    self._last_recalib = time.time()
                    if self.vad.source == "array":
                        enable_array_mic(self.audio)   # (re)arm the array stream
                    self._run_calib(reset=True, secs=CALIB_SECS)
                elif (RECALIB_SECS > 0 and not self._present
                      and (time.time() - self._last_recalib) >= RECALIB_SECS):
                    self._last_recalib = time.time()
                    self._run_calib(reset=False, secs=RECALIB_SAMPLE_SECS)
            try:
                self._tick()
            except Exception as e:
                print(f"[presence] tick error (idling): {e}")
            time.sleep(period)

    def _run_calib(self, reset, secs):
        """Run VAD calibration with a hard time bound. A wedged USB mic makes
        calibrate() block forever, which would starve face detection; capping the
        wait keeps detection alive. If it hangs past the bound, disable the mic
        (greet-only) -- detection keeps running; recovers on restart."""
        result = {}
        def _work():
            try:
                result["res"] = self.vad.calibrate(secs=secs, reset=reset)
            except Exception as e:
                print(f"[presence] calib error: {e}")
        w = threading.Thread(target=_work, daemon=True)
        w.start()
        w.join(secs + 3.0)                       # generous bound; healthy calib ~1.5s
        if w.is_alive():
            print("[presence] WARNING: mic calibration hung -> VAD disabled "
                  "(greet-only). FACE DETECTION CONTINUES; restart to recover the mic.")
            self.vad.mark_degraded()
            return
        res = result.get("res")
        if res and (reset or DEBUG):
            print(f"[presence] calibrated: noise_floor={res[0]:.4f} "
                  f"speech_thresh={res[1]:.4f} (mult={NOISE_MULT})")

    def _tick(self):
        now = time.time()
        frac = self.detector.nearest_face_frac()
        self._debug(now, frac)
        _pub("presence", face_frac=round(frac, 3), thr=PROXIMITY_FRAC,
             nfaces=self.detector.last_nfaces, enabled=self.enabled,
             greeted=self.greeted, busy=self.busy.is_set())
        present = frac >= PROXIMITY_FRAC
        self._present = present
        if present:
            self.last_seen = now
            if self.first_seen is None:
                self.first_seen = now
        else:
            self.first_seen = None
            if self.last_seen and (now - self.last_seen) >= DISARM_SECS:
                if self.greeted:
                    print("[presence] person left; re-arming")
                    self._recalib = True               # refresh floor on empty room
                self.greeted = False

        armed = (self.first_seen is not None
                 and (now - self.first_seen) >= ARM_SECS)
        if armed and not self.greeted and not self.busy.is_set():
            self._greet_and_converse()

    def _greet_and_converse(self):
        """Fire the greeting, then run the hands-free conversation loop while the
        person stays present. Holds `busy` for the whole exchange."""
        self.busy.set()
        self.greeted = True
        # Fresh session per arrival: greeting + all turns + goodbye share history,
        # but a new visitor starts a clean conversation.
        self._session = f"g1p-{uuid.uuid4().hex[:8]}"
        try:
            self.set_led(self.audio, 200, 120, 0)          # amber: thinking
            frame = _grab_jpeg()
            print(f"[presence] new arrival -> greeting (session={self._session})")
            heard, reply = _converse_turn(
                self.audio, _silent_wav_bytes(), frame, _prompt("greet"),
                self.set_led, self.wav_to_pcm16k, self.gain, self.seq_ref,
                transcribe=False, session_id=self._session, phase="greet")
            print(f"[presence] greeting: {reply!r}")

            if not self.vad.available:
                self.last_seen = time.time()
                return

            # We just greeted them, so they're definitely here: refresh the
            # presence stamp before the loop so the greeting's playback time
            # (during which the detector was paused) can't trip an instant leave.
            self.last_seen = time.time()

            # --- hands-free conversation loop -------------------------------
            if DEBUG:
                print(f"[presence][dbg] opening mic (device={self.vad.device}) "
                      "for hands-free...")
            opened = self.vad.open()
            if DEBUG:
                print(f"[presence][dbg] mic open -> {opened}")
            quiet = 0
            departed = False
            while not self._stop.is_set():
                # presence check between/while waiting for utterances
                if not self._present_now(STAY_FRAC):
                    if (time.time() - self.last_seen) >= CONV_DISARM_SECS:
                        print("[presence] person left mid-conversation")
                        departed = True
                        break
                else:
                    self.last_seen = time.time()

                self.set_led(self.audio, 0, 200, 0)        # green: listening
                if DEBUG:
                    print("[presence][dbg] listening for speech...")
                status, clip = self.vad.next_utterance(
                    still_active=lambda: self.enabled and not self._stop.is_set())
                if DEBUG:
                    dur = (len(clip) / _RATE_16K) if clip is not None else 0.0
                    print(f"[presence][dbg] vad status={status} dur={dur:.2f}s "
                          f"webrtcvad={_HAVE_WEBRTCVAD}")

                if status == "left":
                    break
                if status == "timeout":
                    # silence window elapsed with no speech -> re-poll presence
                    quiet += 1
                    if QUIET_TURNS_END and quiet >= QUIET_TURNS_END:
                        print("[presence] conversation idle; standing down")
                        break
                    continue
                quiet = 0

                self.set_led(self.audio, 200, 120, 0)      # amber: thinking
                frame = _grab_jpeg()
                wav = _float_to_wav_bytes(clip)
                heard, reply = _converse_turn(
                    self.audio, wav, frame, _prompt("converse"),
                    self.set_led, self.wav_to_pcm16k, self.gain, self.seq_ref,
                    transcribe=CONV_TRANSCRIBE, session_id=self._session,
                    phase="converse")
                print(f"[presence] reply: {reply!r}")
                self.last_seen = time.time()

            # Say goodbye + wave only on a real departure (not on F2 disable).
            if departed:
                self.vad.close()                # free the mic before the farewell
                self._farewell()
        except Exception as e:
            print(f"[presence] greet/converse error: {e}")
        finally:
            self.vad.close()
            self.busy.clear()
            if self.enabled:
                self.set_led(self.audio, 0, 60, 30)        # back to dim green

    # -- goodbye + wave -------------------------------------------------------
    def _wave(self):
        """Friendly mode-gated wave via G1ArmActionClient, then release. Safe:
        a predefined arm motion that never seizes the locomotion lease."""
        if self.arm is None:
            return
        mode = self.mode_fn() if self.mode_fn else None
        if mode not in SAFE_MODES:
            print(f"[presence] skip wave -- not gesture-ready (mode={mode}, "
                  f"need {SAFE_MODES})")
            return
        try:
            code = self.arm.ExecuteAction(WAVE_ACTION_ID)
            print(f"[presence] wave (action {WAVE_ACTION_ID}) -> {code}")
            time.sleep(WAVE_HOLD_SECS)
        except Exception as e:
            print(f"[presence] wave errored: {e}")
        finally:
            try:
                self.arm.ExecuteAction(ARM_RELEASE_ID)   # never leave arm parked
            except Exception:
                pass

    def _farewell(self):
        """Say a short goodbye (kokoro) while waving. Runs while `busy` is held."""
        print("[presence] farewell: goodbye + wave")
        wt = None
        if WAVE_ON_LEAVE and self.arm is not None:
            wt = threading.Thread(target=self._wave, daemon=True)
            wt.start()                                  # wave concurrently with speech
        try:
            _converse_turn(self.audio, _silent_wav_bytes(), None, _prompt("goodbye"),
                           self.set_led, self.wav_to_pcm16k, self.gain, self.seq_ref,
                           transcribe=False, session_id=getattr(self, "_session", None),
                           phase="goodbye")
        except Exception as e:
            print(f"[presence] goodbye failed: {e}")
        if wt is not None:
            wt.join(timeout=WAVE_HOLD_SECS + 2.0)

    # -- browser "Talk" / remote PTT -----------------------------------------
    def converse_once(self, wav_bytes):
        """Run a single converse turn from externally-supplied audio (the web
        UI's Hold-to-Talk). Reuses a persistent web session so multi-turn works,
        grabs the current camera frame, and holds `busy` so it never overlaps the
        presence loop or a push-to-talk turn. Returns (heard, reply)."""
        if self.busy.is_set():
            raise RuntimeError("busy")
        self.busy.set()
        try:
            if not getattr(self, "_web_session", None):
                self._web_session = f"g1w-{uuid.uuid4().hex[:8]}"
            self.set_led(self.audio, 200, 120, 0)          # amber: thinking
            frame = _grab_jpeg()
            heard, reply = _converse_turn(
                self.audio, wav_bytes, frame, _prompt("converse"),
                self.set_led, self.wav_to_pcm16k, self.gain, self.seq_ref,
                transcribe=True, session_id=self._web_session, phase="converse")
            print(f"[presence] (web) heard={heard!r} reply={reply!r}")
            return heard, reply
        finally:
            self.busy.clear()
            if self.enabled:
                self.set_led(self.audio, 0, 60, 30)
            else:
                self.set_led(self.audio, 20, 20, 20)
