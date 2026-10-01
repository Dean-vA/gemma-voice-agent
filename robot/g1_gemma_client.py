#!/usr/bin/env python3
"""
G1 thin client for the Gemma voice agent (audio-native, off-board inference).

This is the robot-side counterpart to the gemma-voice-agent gateway. The heavy
Gemma 4 E4B model runs OFF-BOARD on a server (e.g. the RTX 5090 box); the robot
only captures audio, streams it to the gateway, and speaks the reply. It does
NOT run any model locally, so it works on the G1's JetPack 5 / CUDA 11.4 Orin
with no torch/vLLM build at all.

Pipeline (push-to-talk):
    F1 held  -> record G1 mic array (UDP multicast, 16 kHz mono s16le)
    release  -> POST audio to  {GEMMA_URL}/chat/stream  (multipart, audio-native)
    SSE      -> stream Gemma's text reply, sentence by sentence
    each     -> Piper (warm HTTP) -> 16 kHz PCM -> AudioClient.PlayStream (DDS)

It reuses the exact, on-robot-verified audio I/O from g1_greeter_shadow.py:
  - The 4-mic array is published on UDP multicast 239.168.123.161:5555 (NOT ALSA).
  - The head speaker is driven over DDS via AudioClient.PlayStream (NOT ALSA).

Safety: like the shadow greeter, this NEVER publishes motion commands. The mic
socket is read-only (joins a multicast group); the speaker uses the high-level
AudioClient. The locomotion controller is never touched, so the robot stays
fully drivable by the hand controller and is unaffected if this process dies.

Run (in the container on the Orin; robot in normal/ready state):
    python3 g1_gemma_client.py <network_interface>      # e.g. eth0

Config via env:
    GEMMA_URL         gateway base URL          (default http://localhost:2730)
    G1_IFACE          DDS interface             (default eth0; also argv[1])
    PIPER_HTTP_URL    warm Piper server         (default http://127.0.0.1:5000)
    SESSION_ID        gateway session id        (default a fresh random id)
    SEND_IMAGE        1/true -> attach a camera frame to each turn (default off)
    SEND_TRANSCRIPT   1/true -> ask gateway for an ASR transcript echo (adds latency)
"""

import io
import os
import sys
import json
import time
import wave
import uuid
import socket
import struct
import queue as _q
import tempfile
import threading
import subprocess

import numpy as np

# =============================== CONFIG ======================================
GEMMA_URL       = os.environ.get("GEMMA_URL", "http://localhost:2730").rstrip("/")
SESSION_ID      = os.environ.get("SESSION_ID") or f"g1-{uuid.uuid4().hex[:8]}"
SEND_IMAGE      = os.environ.get("SEND_IMAGE", "").strip().lower() in ("1", "true", "yes", "on")
SEND_TRANSCRIPT = os.environ.get("SEND_TRANSCRIPT", "").strip().lower() in ("1", "true", "yes", "on")
HTTP_TIMEOUT    = float(os.environ.get("GEMMA_TIMEOUT", "60"))

# --- Controller (Unitree wireless remote) key bitmask --------------------
# Verified on this G1: F2=0x0080, F1=0x0040 (both unmapped in sport mode).
KEY_TOGGLE = 0x0080       # F2  -> master on/off (latching, rising edge)
KEY_TALK   = 0x0040       # F1  -> push-to-talk (hold to record)

# --- Mic (G1 4-mic array via UDP multicast; works in NORMAL mode) --------
MIC_GROUP    = "239.168.123.161"   # G1 mic multicast group
MIC_PORT     = 5555
MIC_RATE     = 16000               # the array publishes 16 kHz mono s16le
MIC_GAIN     = 6.0                 # array runs quiet; lift before sending (clipped)
MIC_LOCAL_IP = None                # None = auto-detect this host's 192.168.123.x IP
REC_MAX_SECS = 15.0                # hard cap on a single utterance

# --- TTS (Piper warm HTTP -> CLI -> espeak), then PlayStream over DDS -----
TTS_RATE       = 16000     # PlayStream requires 16k mono s16le
PIPER_GAIN     = 3.0       # Piper output is near full-scale; needs little lift
TTS_ESPEAK_AMP = "200"
TTS_GAIN       = 4.0
PIPER_VOICE        = os.environ.get("PIPER_VOICE")
PIPER_HTTP_URL     = os.environ.get("PIPER_HTTP_URL", "http://127.0.0.1:5000")
PIPER_HTTP_TIMEOUT = 15.0

# --- Camera (optional; native G1 front camera via VideoClient) -----------
CAM_ENABLE = SEND_IMAGE

# ============================ UNITREE SDK ====================================
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_

try:
    from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
except Exception:
    AudioClient = None
try:
    from unitree_sdk2py.go2.video.video_client import VideoClient
except Exception:
    VideoClient = None
# Arm-action client for the goodbye wave (predefined arm motion; does NOT seize
# the locomotion lease). LowState gives mode_machine for the gesture gate.
try:
    from unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient
except Exception:
    G1ArmActionClient = None
try:
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as HGLowState_
except Exception:
    HGLowState_ = None

# httpx ships with the anthropic SDK already in the greeter image.
import httpx


# ===================== CONTROLLER (passive read) =============================
_keys_lock = threading.Lock()
_latest_keys = 0

def _wc_handler(msg):
    global _latest_keys
    with _keys_lock:
        _latest_keys = int(getattr(msg, "keys", 0))

def get_keys():
    with _keys_lock:
        return _latest_keys


# ===================== MODE (passive read for gesture gate) ==================
_mode_lock = threading.Lock()
_latest_mode = None

def _lowstate_handler(msg):
    global _latest_mode
    with _mode_lock:
        _latest_mode = int(getattr(msg, "mode_machine", -1))

def get_mode():
    with _mode_lock:
        return _latest_mode


# ============================ MIC ============================================
def _find_robot_subnet_ip():
    """Find this host's 192.168.123.x address (the robot subnet), or None."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("192.168.123.164", 9))   # no packet sent; just selects iface
        ip = s.getsockname()[0]
        s.close()
        if ip.startswith("192.168.123."):
            return ip
    except Exception:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if ip.startswith("192.168.123."):
                return ip
    except Exception:
        pass
    return None

def open_mic_socket(local_ip):
    """Join the G1 mic multicast group on the robot-subnet interface."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("", MIC_PORT))
    mreq = struct.pack("4s4s", socket.inet_aton(MIC_GROUP),
                       socket.inet_aton(local_ip))
    s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
    s.setblocking(False)
    return s

def _drain(sock):
    try:
        while True:
            sock.recv(8192)
    except (BlockingIOError, OSError):
        pass

def record_while_held(sock, is_held):
    """Collect multicast PCM while is_held() is True -> float32 mono @16k."""
    if sock is None:
        print("[mic] multicast socket not available")
        return None
    _drain(sock)
    buf = bytearray()
    t0 = time.time()
    max_bytes = int(MIC_RATE * 2 * REC_MAX_SECS)
    while is_held() and (time.time() - t0) < REC_MAX_SECS:
        try:
            buf += sock.recv(8192)
        except BlockingIOError:
            time.sleep(0.005)
        except OSError:
            break
        if len(buf) >= max_bytes:
            break
    if len(buf) < MIC_RATE:          # < ~0.03s of audio -> nothing useful
        return None
    if len(buf) % 2:
        buf = buf[:-1]
    a = np.frombuffer(bytes(buf), dtype=np.int16).astype(np.float32) / 32768.0
    a = np.clip(a * MIC_GAIN, -1.0, 1.0)
    return a

def float_to_wav_bytes(audio, rate=MIC_RATE):
    """float32 mono [-1,1] -> in-memory 16-bit PCM WAV bytes for upload."""
    pcm16 = np.clip(audio * 32768.0, -32768, 32767).astype("<i2")
    bio = io.BytesIO()
    w = wave.open(bio, "wb")
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(rate)
    w.writeframes(pcm16.tobytes())
    w.close()
    return bio.getvalue()


# ============================ CAMERA (optional) ==============================
def grab_frame_jpeg(video):
    """Grab one JPEG frame from the G1 front camera as bytes, or None."""
    if video is None:
        return None
    try:
        code, data = video.GetImageSample()
        if code != 0 or not data:
            return None
        return bytes(data)
    except Exception as e:
        print(f"[cam] GetImageSample failed: {e}")
        return None


# ============================ TTS ============================================
# The G1 speaker is driven over DDS via AudioClient.PlayStream (16 kHz mono
# s16le). We synthesize with Piper (warm HTTP), falling back to Piper CLI then
# espeak-ng so TTS never goes silent.
_tts_seq = 0
USE_CONVERSE = os.environ.get("USE_CONVERSE", "1").strip().lower() in ("1", "true", "yes", "on")
_tts_seq_ref = [0]  # PlayStream sequence counter for converse mode

def _wav_to_pcm16k(wav_path, gain):
    w = wave.open(wav_path, "rb")
    sr, ch = w.getframerate(), w.getnchannels()
    a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    w.close()
    if ch == 2:
        a = a.reshape(-1, 2).mean(axis=1).astype(np.int16)
    if sr != TTS_RATE:
        n = int(len(a) * TTS_RATE / sr)
        a = np.interp(np.linspace(0, len(a), n, endpoint=False),
                      np.arange(len(a)), a).astype(np.int16)
    a = np.clip(a.astype(np.float32) * gain, -32768, 32767).astype(np.int16)
    return a.tobytes(), len(a) / TTS_RATE

def _synth_piper_http(text):
    if not PIPER_HTTP_URL:
        return None
    try:
        import urllib.request
        req = urllib.request.Request(
            PIPER_HTTP_URL,
            data=json.dumps({"text": text}).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=PIPER_HTTP_TIMEOUT) as resp:
            wav_bytes = resp.read()
        wav_path = os.path.join(tempfile.gettempdir(), "g1_say_http.wav")
        with open(wav_path, "wb") as f:
            f.write(wav_bytes)
        return _wav_to_pcm16k(wav_path, PIPER_GAIN)
    except Exception as e:
        print(f"[tts] piper http failed ({e}); trying CLI/espeak")
        return None

def _synth_piper_cli(text):
    if not PIPER_VOICE:
        return None
    wav_path = os.path.join(tempfile.gettempdir(), "g1_say_piper.wav")
    try:
        subprocess.run(
            ["python3", "-m", "piper", "-m", PIPER_VOICE, "--length-scale", "1.08", "-f", wav_path],
            input=text.encode(), check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return _wav_to_pcm16k(wav_path, PIPER_GAIN)
    except Exception as e:
        print(f"[tts] piper cli failed ({e}); falling back to espeak")
        return None

def _synth_espeak(text):
    wav_path = os.path.join(tempfile.gettempdir(), "g1_say.wav")
    subprocess.run(
        ["espeak-ng", "-v", "en", "-a", TTS_ESPEAK_AMP, "-s", "150",
         "-w", wav_path, text],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return _wav_to_pcm16k(wav_path, TTS_GAIN)

def _synth_pcm(text):
    for fn in (_synth_piper_http, _synth_piper_cli):
        out = fn(text)
        if out is not None:
            return out
    return _synth_espeak(text)

def speak(audio, text):
    """Speak English out the real head speaker via AudioClient.PlayStream."""
    global _tts_seq
    if not text:
        return
    if audio is None:
        print(f"[tts] (no audio client) would say: {text!r}")
        return
    try:
        pcm, dur = _synth_pcm(text)
        _tts_seq += 1
        audio.PlayStream("gemma", str(_tts_seq), pcm)
        time.sleep(dur + 0.3)          # let playback finish before next action
    except Exception as e:
        print(f"[tts] PlayStream failed ({e}); would say: {text!r}")

def set_led(audio, r, g, b):
    if audio is None:
        return
    try:
        audio.LedControl(r, g, b)
    except Exception:
        pass


# ===================== SENTENCE SPLIT (stream -> TTS) ========================
_SENT_END = (".", "!", "?")

class SentenceEmitter:
    """Feed streamed text; emit complete sentences once each so TTS can start
    before the full reply finishes. A sentence ends on terminal punctuation
    FOLLOWED by a space/newline already in the buffer (so a trailing '.' that's
    still streaming, or a decimal like '3.5', won't split early)."""
    def __init__(self, on_sentence):
        self.buf = ""
        self.spoken = 0
        self.on_sentence = on_sentence

    def feed(self, text):
        self.buf += text
        n = len(self.buf)
        i = self.spoken
        while i < n - 1:
            if self.buf[i] in _SENT_END and self.buf[i + 1] in (" ", "\n"):
                seg = self.buf[self.spoken:i + 1].strip()
                if len(seg) >= 2:
                    self.on_sentence(seg)
                self.spoken = i + 1
            i += 1

    def flush(self):
        tail = self.buf[self.spoken:].strip()
        if tail:
            self.on_sentence(tail)
        self.spoken = len(self.buf)


# ============================ GEMMA GATEWAY ==================================
def stream_gemma(wav_bytes, image_bytes, on_token):
    """POST audio (+ optional image) to {GEMMA_URL}/chat/stream and stream the
    reply. Calls on_token(text) for each token chunk. Returns the full reply."""
    files = {"audio": ("speech.wav", wav_bytes, "audio/wav")}
    if image_bytes:
        files["image"] = ("frame.jpg", image_bytes, "image/jpeg")
    data = {"session_id": SESSION_ID}
    if SEND_TRANSCRIPT:
        data["transcribe"] = "true"

    reply_parts = []
    url = f"{GEMMA_URL}/chat/stream"
    with httpx.stream("POST", url, files=files, data=data, timeout=HTTP_TIMEOUT) as r:
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
                    print(f"  heard: {payload.get('text')!r}")
                elif event == "token":
                    tok = payload.get("text", "")
                    if tok:
                        reply_parts.append(tok)
                        on_token(tok)
                elif event == "done":
                    m = payload.get("metrics") or {}
                    ttft = m.get("ttft_ms")
                    if ttft is not None:
                        print(f"  [gateway] TTFT={ttft}ms backend={m.get('backend')}")
                    return payload.get("reply", "".join(reply_parts))
    return "".join(reply_parts)


# ============================ INTERACTION ====================================
def run_interaction(audio, video, mic_sock):
    print("  listening (hold talk key)...")
    set_led(audio, 0, 200, 0)                        # green: listening
    clip = record_while_held(mic_sock, lambda: bool(get_keys() & KEY_TALK))
    if clip is None:
        print("  (no audio captured)")
        set_led(audio, 0, 80, 160)
        return
    image_bytes = grab_frame_jpeg(video) if CAM_ENABLE else None
    set_led(audio, 200, 120, 0)                      # amber: thinking
    wav_bytes = float_to_wav_bytes(clip)

    if USE_CONVERSE:
        from converse_mode import run_converse
        try:
            heard, reply = run_converse(audio, wav_bytes, image_bytes,
                                        set_led, _wav_to_pcm16k, PIPER_GAIN, _tts_seq_ref)
            print(f"  Gemma: {reply!r}")
        except Exception as e:
            print(f"[converse] failed ({e}); falling back to Piper path")
        else:
            set_led(audio, 20, 20, 20)
            return

    # Serialize TTS playback (PlayStream is blocking) on a worker thread so we
    # can keep parsing the SSE stream and queue sentences as they complete.
    say_q = _q.Queue()
    done = object()

    def speaker_worker():
        while True:
            item = say_q.get()
            if item is done:
                return
            speak(audio, item)

    sp_thread = threading.Thread(target=speaker_worker, daemon=True)
    sp_thread.start()

    first = {"led": False}
    def on_sentence(s):
        if not first["led"]:
            set_led(audio, 0, 80, 160)               # blue: speaking
            first["led"] = True
        print(f"  say: {s!r}")
        say_q.put(s)

    emitter = SentenceEmitter(on_sentence)
    try:
        reply = stream_gemma(wav_bytes, image_bytes, emitter.feed)
        emitter.flush()
        print(f"  Gemma: {reply!r}")
    except Exception as e:
        print(f"[gemma] request failed: {e}")
        say_q.put("Sorry, I could not reach my brain just now.")
    finally:
        say_q.put(done)
        sp_thread.join()


# ============================ MAIN ===========================================
def main():
    iface = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("G1_IFACE", "eth0")

    print(f"Gemma G1 client -> {GEMMA_URL}  (session={SESSION_ID}, iface={iface})")

    # Probe the gateway up front so misconfig is obvious before the first turn.
    try:
        h = httpx.get(f"{GEMMA_URL}/health", timeout=10.0)
        info = h.json()
        print(f"[gateway] ok: backend={info.get('backend')} quant={info.get('quant_mode')}")
    except Exception as e:
        print(f"[gateway] WARNING: /health unreachable at {GEMMA_URL} ({e}). "
              "Check the server IP/port and the robot's WiFi.")

    ChannelFactoryInitialize(0, iface)

    # Passive controller subscription (read-only) -- never publishes commands.
    wc = ChannelSubscriber("rt/wirelesscontroller", WirelessController_)
    wc.Init(_wc_handler, 10)

    # Passive mode subscription (read-only): mode_machine for the wave gate.
    if HGLowState_ is not None:
        try:
            ls = ChannelSubscriber("rt/lowstate", HGLowState_)
            ls.Init(_lowstate_handler, 10)
        except Exception as e:
            print(f"[mode] lowstate subscribe failed (wave will stay disabled): {e}")

    audio = None
    if AudioClient is not None:
        try:
            audio = AudioClient()
            audio.SetTimeout(10.0)
            audio.Init()
            try:
                audio.SetVolume(100)
            except Exception:
                pass
        except Exception as e:
            print(f"[audio] init failed (no voice/LED): {e}")
            audio = None

    video = None
    if CAM_ENABLE and VideoClient is not None:
        try:
            video = VideoClient()
            video.SetTimeout(3.0)
            video.Init()
            code, data = video.GetImageSample()
            if code != 0 or not data:
                print(f"[cam] GetImageSample code={code}; disabling vision")
                video = None
            else:
                print(f"[cam] front camera ready ({len(data)} bytes/frame)")
        except Exception as e:
            print(f"[cam] VideoClient init failed; running without vision: {e}")
            video = None

    mic_sock = None
    try:
        local_ip = MIC_LOCAL_IP or _find_robot_subnet_ip() or "192.168.123.164"
        mic_sock = open_mic_socket(local_ip)
        print(f"[mic] joined {MIC_GROUP}:{MIC_PORT} via {local_ip}")
    except Exception as e:
        print(f"[mic] could not open multicast socket: {e}")
        mic_sock = None

    # Presence-mode (F2) runs an autonomous greet+converse loop in a background
    # thread; F1 push-to-talk stays available as a manual fallback. A shared
    # CONVERSATION_BUSY event keeps the two paths from overlapping (and stops the
    # robot self-triggering off its own speaker during presence playback).
    CONVERSATION_BUSY = threading.Event()

    # Arm-action client for the goodbye wave (None if SDK/init unavailable).
    # Init alone does NOT take control from the hand controller.
    arm = None
    if G1ArmActionClient is not None:
        try:
            arm = G1ArmActionClient()
            arm.SetTimeout(10.0)
            arm.Init()
            print("[arm] gesture client ready (goodbye wave enabled)")
        except Exception as e:
            print(f"[arm] init failed (no wave): {e}")
            arm = None

    presence = None
    try:
        import presence as _presence_mod
        presence = _presence_mod.PresenceController(
            audio, CONVERSATION_BUSY, set_led,
            _wav_to_pcm16k, PIPER_GAIN, _tts_seq_ref,
            arm=arm, mode_fn=get_mode)
        presence.start()
        print("[presence] controller ready (press F2 to arm)")
    except Exception as e:
        print(f"[presence] unavailable ({e}); running F1 push-to-talk only")

    # Web control centre (PRESENCE_WEB=1). Guarded: a UI failure must never take
    # down the greeter.
    if presence is not None and os.environ.get("PRESENCE_WEB", "").strip().lower() \
            in ("1", "true", "yes", "on"):
        try:
            from control_center import start_control_center
            start_control_center(presence.detector, presence, CONVERSATION_BUSY)
        except Exception as e:
            print(f"[web] control centre unavailable ({e}); continuing headless")

    prev = 0
    set_led(audio, 20, 20, 20)        # dim: off
    print("Ready. F2 = presence-mode on/off, hold F1 = push-to-talk. Ctrl+C to quit.")

    try:
        while True:
            keys = get_keys()

            if (keys & KEY_TOGGLE) and not (prev & KEY_TOGGLE):   # F2 rising edge
                if presence is not None:
                    presence.toggle()
                else:
                    print("[presence] not available on this run")

            if (keys & KEY_TALK) and not (prev & KEY_TALK):       # F1 rising edge
                if CONVERSATION_BUSY.is_set():
                    print("[talk] busy (presence active); ignoring F1")
                else:
                    CONVERSATION_BUSY.set()
                    try:
                        run_interaction(audio, video, mic_sock)
                    except Exception as e:
                        print(f"[loop] interaction error (idling): {e}")
                    finally:
                        CONVERSATION_BUSY.clear()
                    prev = get_keys()
                    continue

            prev = keys
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        if presence is not None:
            presence.stop()
        set_led(audio, 0, 0, 0)


from usb_mic import record_while_held  # USB-mic override
from usb_cam import grab_frame_jpeg  # USB-cam override
if __name__ == "__main__":
    main()
