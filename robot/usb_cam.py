"""USB camera override: persistent capture handle, one JPEG frame per call.
Mirrors grab_frame_jpeg(video) -> bytes|None. `video` ignored.

Cross-platform: on Linux/robot it opens /dev/video{CAM_DEV} via V4L2 (unchanged);
on the dev laptop (Windows/macOS) it opens an integer device index via the
platform-default backend (MSMF/DSHOW/AVFoundation). Controlled by CAM_BACKEND
(`v4l2` on Linux, `auto` elsewhere) and CAM_INDEX (laptop) / CAM_DEV (robot).

CAM_SOURCE=videohub instead reads the G1's head camera (the RealSense colour
stream) from Unitree's video hub over DDS. The hub keeps that device open
itself, so V4L2 can't have it; frames come back as 1920x1080 JPEGs."""
import os, sys, cv2, threading, time
import numpy as np

CAM_SOURCE = os.environ.get("CAM_SOURCE", "usb")  # usb | videohub
CAM_DEV   = os.environ.get("CAM_DEV", "6")     # robot V4L2 node number
CAM_INDEX = os.environ.get("CAM_INDEX", "0")   # laptop capture index
CAM_BACKEND = os.environ.get(
    "CAM_BACKEND", "v4l2" if sys.platform.startswith("linux") else "auto")
CAM_W   = int(os.environ.get("CAM_W", "640"))
CAM_H   = int(os.environ.get("CAM_H", "480"))
CAM_FPS = float(os.environ.get("CAM_FPS", "15"))  # videohub poll rate
JPEG_Q  = int(os.environ.get("CAM_JPEG_Q", "60"))

_cap = None
_lock = threading.Lock()

class _VideoHubCap:
    """The few cv2.VideoCapture methods callers use, backed by the video hub.
    read() paces itself to CAM_FPS so the grabber loop doesn't spin; frames are
    scaled to CAM_W wide, keeping the hub's 16:9 aspect."""
    def __init__(self):
        from unitree_sdk2py.go2.video.video_client import VideoClient
        self._client = VideoClient()      # needs ChannelFactoryInitialize first
        self._client.SetTimeout(3.0)
        self._client.Init()
        self._next = 0.0
        self._open = True

    def isOpened(self):
        return self._open

    def set(self, prop, value):
        return False

    def grab(self):
        return True                       # no buffer to flush

    def read(self):
        wait = self._next - time.time()
        if wait > 0:
            time.sleep(wait)
        self._next = max(self._next, time.time()) + 1.0 / CAM_FPS
        try:
            code, data = self._client.GetImageSample()
        except Exception as e:
            print(f"[cam] videohub: {e}")
            return False, None
        if code != 0 or not data:
            return False, None
        img = cv2.imdecode(np.frombuffer(bytes(data), np.uint8),
                           cv2.IMREAD_REDUCED_COLOR_2)
        if img is None:
            return False, None
        h, w = img.shape[:2]
        if w != CAM_W:
            img = cv2.resize(img, (CAM_W, round(h * CAM_W / w)),
                             interpolation=cv2.INTER_AREA)
        return True, img

    def release(self):
        self._open = False

def _open_cap():
    """Open the capture handle per platform/backend."""
    if CAM_SOURCE == "videohub":
        try:
            return _VideoHubCap(), "videohub"
        except Exception as e:
            print(f"[cam] videohub unavailable: {e}")
            return cv2.VideoCapture(), "videohub"   # not opened
    if CAM_BACKEND == "v4l2":
        target = f"/dev/video{CAM_DEV}"
        return cv2.VideoCapture(target, cv2.CAP_V4L2), target
    # laptop / non-Linux: integer index, default backend
    try:
        idx = int(CAM_INDEX)
    except ValueError:
        idx = 0
    return cv2.VideoCapture(idx), f"index {idx}"

def _get_cap():
    global _cap
    if _cap is None or not _cap.isOpened():
        c, target = _open_cap()
        if not c.isOpened():
            print(f"[cam] could not open camera ({target})")
            return None
        c.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
        c.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        c.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # don't accumulate stale frames
        _cap = c
    return _cap

def reset():
    """Force a fresh capture handle on the next _get_cap(). A wedged USB camera
    often keeps reporting isOpened()==True while read() returns nothing, so
    _get_cap() won't reopen it on its own; the Grabber watchdog calls this to
    release the dead handle and let the next grab reopen a clean one."""
    global _cap
    with _lock:
        if _cap is not None:
            try:
                _cap.release()
            except Exception:
                pass
        _cap = None

def grab_frame_jpeg(video):
    with _lock:
        cap = _get_cap()
        if cap is None:
            return None
        # one grab to flush the single-frame buffer, one to get current
        cap.grab()
        ok, frame = cap.read()
        if not ok or frame is None:
            print("[cam] no frame")
            return None
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
        return jpg.tobytes() if ok else None

if __name__ == "__main__":
    import time
    for i in range(3):
        t = time.time()
        b = grab_frame_jpeg(None)
        print(f"grab {i}: {len(b) if b else None} bytes in {(time.time()-t)*1000:.0f}ms")
