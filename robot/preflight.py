#!/usr/bin/env python3
"""
G1 shadow greeter -- HARDWARE PRE-FLIGHT.

Runs an ordered battery of checks that confirm each assumption the demo relies
on, on the real robot, in the same container. Read-only w.r.t. the robot:
it subscribes to DDS topics and reads the FSM id, but sends NO motion command.

Order is dependency-first: env/imports -> internet -> DDS -> controller ->
camera -> mic -> speaker -> STT -> Claude -> LocoClient read. Each check is
isolated and reports [PASS]/[FAIL]/[WARN]/[SKIP]; the suite never aborts early,
so you get the full picture in one run, then fix what's flagged and re-run.

Run it in the container (so it tests the real deployment env), e.g.:
    docker compose run --rm \
        -v "$PWD/preflight.py:/app/preflight.py" \
        g1-greeter python3 /app/preflight.py eth0

Some checks are interactive (press buttons / speak / "did you hear it?").
"""

import sys
import time
import threading

# ----- config (match the main script) --------------------------------------
CAM_INDEX = 0
MIC_RATE  = 16000
WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE = "base", "cpu", "int8"
CLAUDE_MODEL = "claude-haiku-4-5-20251001"

IFACE = sys.argv[1] if len(sys.argv) > 1 else "eth0"

# ----- result tracking ------------------------------------------------------
RESULTS = []
def record(name, status, msg=""):
    RESULTS.append((name, status, msg))
    print(f"  [{status}] {name}" + (f" -- {msg}" if msg else ""))

# shared artifacts passed between checks
STATE = {"frame_jpg": None, "audio": None}

_dds_ready = False
def ensure_dds():
    """Initialise the DDS channel factory exactly once."""
    global _dds_ready
    if _dds_ready:
        return True
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    ChannelFactoryInitialize(0, IFACE)
    _dds_ready = True
    return True


# ============================ CHECKS ========================================
def check_imports():
    print("\n[1] Python & imports")
    print(f"  python: {sys.version.split()[0]}  (need >= 3.9 for ctranslate2)")
    mods = [
        ("numpy", "numpy"),
        ("cv2 (opencv)", "cv2"),
        ("sounddevice", "sounddevice"),
        ("faster_whisper", "faster_whisper"),
        ("anthropic", "anthropic"),
        ("cyclonedds", "cyclonedds"),
        ("unitree core", "unitree_sdk2py.core.channel"),
    ]
    for label, mod in mods:
        try:
            __import__(mod)
            record(f"import {label}", "PASS")
        except Exception as e:
            record(f"import {label}", "FAIL", repr(e))
    # IDL + client imports (these are the ones most likely to differ by version)
    for label, stmt in [
        ("WirelessController_ idl",
         "from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_"),
        ("hg LowState_ idl",
         "from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_"),
        ("LocoClient",
         "from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient"),
    ]:
        try:
            exec(stmt, {})
            record(label, "PASS")
        except Exception as e:
            record(label, "FAIL", repr(e))


def check_internet():
    print("\n[2] Internet egress (for the Claude API)")
    import urllib.request
    try:
        req = urllib.request.Request("https://api.anthropic.com", method="HEAD")
        urllib.request.urlopen(req, timeout=8)
        record("reach api.anthropic.com", "PASS")
    except urllib.error.HTTPError as e:
        # Any HTTP response (e.g. 401/404) proves connectivity.
        record("reach api.anthropic.com", "PASS", f"HTTP {e.code} (reachable)")
    except Exception as e:
        record("reach api.anthropic.com", "FAIL", f"{e!r} -- is the Orin's WiFi up?")


def check_dds_robot():
    print("\n[3] DDS link to the robot (subscribe rt/lowstate)")
    try:
        ensure_dds()
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
        got = threading.Event()
        sub = ChannelSubscriber("rt/lowstate", LowState_)
        sub.Init(lambda msg: got.set(), 10)
        if got.wait(timeout=5.0):
            record("rt/lowstate messages arriving", "PASS",
                   "DDS + interface + unitree_hg IDL all good")
        else:
            record("rt/lowstate messages arriving", "FAIL",
                   f"no data in 5s -- wrong interface ('{IFACE}'?), CycloneDDS, or robot off")
    except Exception as e:
        record("DDS subscribe", "FAIL", repr(e))


def check_controller():
    print("\n[4] Controller -- press your intended on/off + talk buttons now (10s)")
    seen_keys = set()
    lowstate_changed = {"v": False}
    try:
        ensure_dds()
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

        def on_wc(msg):
            k = int(getattr(msg, "keys", 0))
            if k:
                seen_keys.add(k)

        baseline = {"b": None}
        def on_ls(msg):
            wr = tuple(getattr(msg, "wireless_remote", []) or [])
            if baseline["b"] is None:
                baseline["b"] = wr
            elif wr != baseline["b"]:
                lowstate_changed["v"] = True

        s1 = ChannelSubscriber("rt/wirelesscontroller", WirelessController_)
        s1.Init(on_wc, 10)
        s2 = ChannelSubscriber("rt/lowstate", LowState_)
        s2.Init(on_ls, 10)

        print("  ...listening (press a few buttons)")
        time.sleep(10.0)

        if seen_keys:
            record("rt/wirelesscontroller carries buttons", "PASS",
                   "observed keys: " + ", ".join(hex(k) for k in sorted(seen_keys)))
            print("  -> set KEY_TOGGLE / KEY_TALK in the main script to values seen above")
        elif lowstate_changed["v"]:
            record("rt/wirelesscontroller carries buttons", "WARN",
                   "no keys here, but LowState.wireless_remote DID change -> use the LowState fallback reader")
        else:
            record("controller input seen", "FAIL",
                   "no button activity on either topic -- is the controller on / paired?")
    except Exception as e:
        record("controller subscribe", "FAIL", repr(e))


def check_camera():
    print("\n[5] USB webcam")
    try:
        import cv2
        cap = cv2.VideoCapture(CAM_INDEX)
        if not cap.isOpened():
            record(f"open /dev/video{CAM_INDEX}", "FAIL",
                   "not opened -- check device passthrough / try another index")
            return
        for _ in range(3):
            cap.read()
        ok, frame = cap.read()
        if ok and frame is not None:
            h, w = frame.shape[:2]
            ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok2:
                STATE["frame_jpg"] = buf.tobytes()
                with open("/tmp/preflight_cam.jpg", "wb") as f:
                    f.write(STATE["frame_jpg"])
            record("capture frame", "PASS", f"{w}x{h}, saved /tmp/preflight_cam.jpg")
        else:
            record("capture frame", "FAIL", "opened but no frame")
        cap.release()
    except Exception as e:
        record("camera", "FAIL", repr(e))


def check_mic():
    print("\n[6] Microphone -- SPEAK NOW for ~3 seconds")
    try:
        import numpy as np
        import sounddevice as sd
        rec = sd.rec(int(3 * MIC_RATE), samplerate=MIC_RATE, channels=1, dtype="float32")
        sd.wait()
        audio = rec.flatten()
        STATE["audio"] = audio
        rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 0.01:
            record("mic capture", "PASS", f"level rms={rms:.4f} peak={peak:.3f} (audio detected)")
        else:
            record("mic capture", "WARN",
                   f"near-silence (rms={rms:.4f}) -- mic not default source, muted, or wrong Pulse socket")
    except Exception as e:
        record("mic", "FAIL", f"{e!r} -- PulseAudio socket/UID? run: ls /run/user/*/pulse/")


def check_speaker():
    print("\n[7] Speaker -- playing a 1s tone through the default sink")
    try:
        import numpy as np
        import sounddevice as sd
        t = np.linspace(0, 1.0, MIC_RATE, endpoint=False)
        tone = (0.2 * np.sin(2 * np.pi * 440 * t)).astype("float32")
        sd.play(tone, MIC_RATE)
        sd.wait()
        ans = input("  Did you hear the tone from the robot's speaker? [y/N] ").strip().lower()
        record("speaker output", "PASS" if ans == "y" else "WARN",
               "" if ans == "y" else "no tone heard -- check the default Pulse sink / volume")
    except Exception as e:
        record("speaker", "FAIL", repr(e))


def check_stt():
    print("\n[8] Speech-to-text (faster-whisper, CPU)")
    if STATE["audio"] is None:
        record("whisper", "SKIP", "no mic audio captured in step 6")
        return
    try:
        from faster_whisper import WhisperModel
        t0 = time.time()
        model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
        load = time.time() - t0
        t1 = time.time()
        segments, _ = model.transcribe(STATE["audio"], language="en")
        text = " ".join(s.text for s in segments).strip()
        record("whisper transcribe", "PASS",
               f"load {load:.1f}s, infer {time.time()-t1:.1f}s -> {text!r}")
    except Exception as e:
        record("whisper", "FAIL", repr(e))


def check_claude():
    print("\n[9] Claude API (forced tool call" +
          (" + vision" if STATE["frame_jpg"] else ", text-only") + ")")
    try:
        import base64, anthropic
        client = anthropic.Anthropic()
        content = []
        if STATE["frame_jpg"]:
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/jpeg",
                "data": base64.b64encode(STATE["frame_jpg"]).decode()}})
        content.append({"type": "text", "text": "Say a one-word hello and pick gesture 'none'."})
        tool = {"name": "respond", "input_schema": {"type": "object", "properties": {
            "speech": {"type": "string"},
            "gesture": {"type": "string", "enum": ["none", "wave", "shake_hand"]}},
            "required": ["speech", "gesture"]}}
        msg = client.messages.create(
            model=CLAUDE_MODEL, max_tokens=100,
            tools=[tool], tool_choice={"type": "tool", "name": "respond"},
            messages=[{"role": "user", "content": content}])
        out = next((b.input for b in msg.content if b.type == "tool_use"), None)
        if out:
            record("claude round-trip", "PASS", f"speech={out.get('speech')!r} gesture={out.get('gesture')}")
        else:
            record("claude round-trip", "FAIL", "no tool_use block returned")
    except Exception as e:
        record("claude", "FAIL", f"{e!r} -- API key set? network up?")


def check_loco_fsm():
    print("\n[10] LocoClient FSM read (read-only -- NO motion)")
    try:
        ensure_dds()
        from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
        loco = LocoClient()
        loco.SetTimeout(5.0)
        loco.Init()
        res = loco.GetFsmId()
        record("GetFsmId()", "PASS", f"returned {res!r}  <-- note this value for SAFE_FSM when going live")
    except Exception as e:
        record("loco FSM read", "WARN",
               f"{e!r} -- harmless for shadow (gesture is logged), but fix before going live")


# ============================ RUN ===========================================
def main():
    print(f"=== G1 shadow greeter pre-flight (interface: {IFACE}) ===")
    for chk in (check_imports, check_internet, check_dds_robot, check_controller,
                check_camera, check_mic, check_speaker, check_stt,
                check_claude, check_loco_fsm):
        try:
            chk()
        except KeyboardInterrupt:
            print("\n(interrupted -- skipping to summary)")
            break
        except Exception as e:
            record(chk.__name__, "FAIL", f"unexpected: {e!r}")

    print("\n=== SUMMARY ===")
    counts = {}
    for name, status, msg in RESULTS:
        counts[status] = counts.get(status, 0) + 1
    for status in ("PASS", "WARN", "FAIL", "SKIP"):
        if counts.get(status):
            print(f"  {status}: {counts[status]}")
    fails = [n for n, s, _ in RESULTS if s == "FAIL"]
    if fails:
        print("\nBlocking issues to fix before the shadow run:")
        for n in fails:
            print(f"  - {n}")
    else:
        print("\nNo blocking failures. You're clear to run the shadow demo "
              "(DRY_RUN=True). Review any WARN items above.")


if __name__ == "__main__":
    main()
