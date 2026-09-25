"""USB-mic capture drop-in: replaces the multicast record_while_held.
float32 mono @16k, same return type the greeter's transcribe() expects.

Import-safe: if sounddevice (PortAudio) can't load -- e.g. on the win-arm64 dev
laptop -- the module still imports and mic capture is simply unavailable
(_HAVE_SD=False), so the rest of the app keeps working and the browser "Talk"
path provides audio instead."""
import os
import time
import numpy as np

try:
    import sounddevice as sd
    _HAVE_SD = True
except Exception as _e:                 # PortAudio missing/broken -> degrade
    sd = None
    _HAVE_SD = False
    print(f"[mic] sounddevice unavailable ({_e}); local mic capture disabled")

USB_RATE = 48000        # USB webcam mic native rate (verified hw:2,0)
USB_NAME = "USB"        # substring match in sounddevice device name
MIC_GAIN = 1.5          # USB mic is hot (peaked ~6210); was 6.0 for the quiet array
REC_MAX_SECS = 15.0
MIC_INDEX = os.environ.get("MIC_INDEX")  # explicit input device index (optional)

def _find_usb_input():
    """Pick an input device: explicit MIC_INDEX, else the robot's "USB" mic,
    else the system default input (laptop built-in), else the first input."""
    if not _HAVE_SD:
        return None
    if MIC_INDEX not in (None, ""):
        try:
            return int(MIC_INDEX)
        except ValueError:
            pass
    try:
        devs = sd.query_devices()
    except Exception:
        return None
    for i, d in enumerate(devs):                          # robot CyberTrack mic
        if d["max_input_channels"] > 0 and USB_NAME in d["name"]:
            return i
    try:                                                   # laptop default input
        default_in = sd.default.device[0]
        if isinstance(default_in, int) and default_in >= 0:
            return default_in
    except Exception:
        pass
    for i, d in enumerate(devs):                           # any input device
        if d["max_input_channels"] > 0:
            return i
    return None

def record_while_held(sock, is_held):
    """Capture USB-mic audio while is_held() is True -> float32 mono @16k.
    `sock` is ignored (kept for call-site compatibility)."""
    if not _HAVE_SD:
        print("[mic] sounddevice unavailable; cannot record")
        return None
    dev = _find_usb_input()
    if dev is None:
        print("[mic] no input device found")
        return None
    frames = []
    t0 = time.time()
    with sd.InputStream(samplerate=USB_RATE, channels=1, dtype="int16",
                        device=dev, blocksize=2048) as stream:
        while is_held() and (time.time() - t0) < REC_MAX_SECS:
            block, _ = stream.read(2048)
            frames.append(block.copy())
    if not frames:
        return None
    raw = np.concatenate(frames, axis=0).reshape(-1)        # int16 @48k
    if len(raw) < USB_RATE * 0.3:                           # < ~0.3s -> nothing
        return None
    a48 = raw.astype(np.float32) / 32768.0
    a16 = a48[::3]                                          # 48000/3 = 16000
    peak = float(np.abs(a16).max()) or 1.0
    a16 = a16 * (0.5 / peak)          # normalize to 0.5 peak, no clipping
    return np.clip(a16, -1.0, 1.0)

if __name__ == "__main__":
    # standalone smoke test: records 4s unconditionally, prints peak + duration
    import wave
    t0 = time.time()
    a = record_while_held(None, lambda: time.time() - t0 < 4.0)
    if a is None:
        print("no audio captured"); raise SystemExit(1)
    print(f"captured {len(a)} samples = {len(a)/16000:.2f}s @16k, "
          f"peak {float(np.abs(a).max()):.3f}")
    pcm = (np.clip(a, -1, 1) * 32767).astype(np.int16)
    with wave.open("/tmp/usbmic16k.wav", "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(pcm.tobytes())
    print("wrote /tmp/usbmic16k.wav")
