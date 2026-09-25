# Presence Detection & VAD — How it works

Deep-dive on the two on-robot subsystems in `presence.py`: **presence detection**
(camera → Haar → proximity gate → debounce state machine) and the **hands-free
VAD** (mic → anti-alias downsample → auto-calibrated adaptive energy gate). Both
were tuned hard against the live robot; this captures the *why*, not just the *what*.

Last updated: 2026-09-25 (AI in the City, Breda). Companion to `CONTROL_CENTRE.md`.

---

## A. PRESENCE DETECTION

### A.1 Flow / state machine (`PresenceController`)
Runs in a background thread at `DETECT_FPS` (5 Hz). Each tick (`_tick`):
1. `frac = detector.nearest_face_frac()` — grab a frame, detect the largest frontal face, return `face_width / frame_width` (0 if none).
2. Publish `{face_frac, thr, nfaces, enabled, greeted, busy}` to the web Hub.
3. Debounce:
   - **present** = `frac >= PROXIMITY_FRAC`. If present: stamp `last_seen`; start `first_seen` if unset.
   - **armed** = present continuously for `ARM_SECS` (~1 s).
   - **left** = absent for `DISARM_SECS` (~2 s) → reset the `greeted` latch (and queue a recalibration of the room).
   - **Mid-conversation** the leash is looser: the visitor counts as present down to
     `STAY_FRAC` (0.12) and only "leaves" after `CONV_DISARM_SECS` (5 s). With the arrival
     gate (18%, 2 s) a glance away or leaning back said goodbye to someone still talking.
4. If `armed and not greeted and not busy` → `_greet_and_converse()`.

`_greet_and_converse()`: hold `CONVERSATION_BUSY`, mint a per-arrival `session_id`,
**greet once** (silent WAV + camera frame + greet instruction → kokoro reply),
then run the hands-free **conversation loop** while the person stays present.
On a real departure → `_farewell()` (spoken goodbye + arm wave). Greets exactly
once per arrival; re-arms after they leave.

F2 toggles presence-mode; F1 push-to-talk is an always-available manual fallback
(both guarded by `CONVERSATION_BUSY` so they never overlap / self-trigger).

### A.2 Face detection (`FaceDetector`)
- OpenCV **Haar** frontal cascade (`haarcascade_frontalface_default.xml`).
- **CLAHE before `detectMultiScale`** (`clipLimit=2.0`, 8×8 tiles). Contrast
  normalization is essential — backlit by windows, raw Haar detects nothing. The
  original fix was global `cv2.equalizeHist`, but in a hall with a big bright wall
  the global histogram is dominated by the wall and the face is crushed to mush:
  a centred frontal face gave `nfaces=0` with every cascade. CLAHE equalizes per
  tile, so it handles both the backlit window and the bright wall (verified on the
  old backlit debug frames + live event frames, no false positive on an empty frame).
  Slightly more prone to false positives when the lens is blocked (e.g. a hand).
- `detectMultiScale(scaleFactor=HAAR_SCALE, minNeighbors=HAAR_NEIGHBORS, minSize=(HAAR_MIN_PX,HAAR_MIN_PX))`.
- **Proximity gate**: instead of "is there a face", we use "is a face *close enough*"
  — `widest_face_width / frame_width >= PROXIMITY_FRAC`. 0.18 ≈ 1.5–2 m; 0.25 ≈ arm's reach.

### A.3 The on-feed overlay (`_draw_overlay`)
Shown **only when armed**. Boxes are green if a face clears the gate, blue-ish if
detected-but-too-far. Label is human-readable: `"1 face · 24% close (need 18%) · ARMED · greeted"`.
> Pitfall: never write `faces or ()` — `faces` is a numpy ndarray and `bool(ndarray)`
> raises "ambiguous truth value" the moment a face is found. Iterate `list(faces)`.

### A.4 Presence params
| env | default | meaning |
|---|---|---|
| `PRESENCE_PROX_FRAC` | 0.18 | gate: face width ÷ frame width to count as "present" |
| `PRESENCE_ARM_SECS` | 1.0 | continuous presence before arming a greeting |
| `PRESENCE_DISARM_SECS` | 2.0 | continuous absence before "left" (resets greeted) |
| `PRESENCE_STAY_FRAC` | 0.12 | face gate while a conversation is running |
| `PRESENCE_CONV_DISARM_SECS` | 5.0 | absence before a mid-conversation goodbye |
| `PRESENCE_DETECT_FPS` | 5 | detection tick rate |
| `PRESENCE_HAAR_SCALE` | 1.2 | Haar scaleFactor |
| `PRESENCE_HAAR_NEIGHBORS` | 4 | Haar minNeighbors (lower = more sensitive, more false +) |
| `PRESENCE_HAAR_MIN_PX` | 60 | min face size in px |

### A.5 Presence gotchas
- **USB camera renumbers** (e.g. video6→video7 when the RealSense grabs lower nodes). Mapped by the stable **by-id** symlink in compose → always `/dev/video6` in-container.
- **Single-opener V4L2**: only one process can hold the camera. In the control-centre build a single `Grabber` thread owns it; the detector reads its snapshot via `FRAME_SOURCE` (no contention, no capture backlog/lag).

---

## B. VAD (hands-free voice capture)

The on-robot conversation loop needs to know **when the person starts and stops
talking**. This was the hardest part — several dead ends before the current design.

### B.1 The signal path (`VadCapture`)
Two mic sources, chosen by `PRESENCE_MIC` (compose default `array`) or live from the
control-centre **mic source** dropdown (`POST /api/mic`, takes effect on the next listen):

**G1 mic array** (`array`) — the robot's built-in 4-mic array, published by the
voice service as **16 kHz mono s16le on UDP multicast `239.168.123.161:5555`**, but
**only while "mic mode" is on**: undocumented voice API **1008** `{"mode": 1}`
(`{"mode": 2}` = off; the app / remote L1+L2 "wake-up mode" is the same switch).
`enable_array_mic()` calls it on every arm — it does not survive a power cycle.
A reader thread joins the group (via `ARRAY_LOCAL_IP`, the eth0 address) and
re-frames the 5120-byte packets into 30 ms / 480-sample frames on the same queue the
USB path uses. Already 16 kHz, so no downsampling. Measured at the event, same
speaker, same moment: **array 27 dB SNR (floor 0.0047) vs USB 8.6 dB (floor 0.0154)**.
The flip side: it hears the whole room (see B.3). Two consecutive stalls → falls back
to the USB mic and recalibrates.

**USB mic** (`usb`):
- USB mic captured at **48 kHz** mono int16 (the CyberTrack's native rate).
- Downsampled to **16 kHz** by a **3-tap box filter** (mean of each triplet), **not**
  a naive `[::3]` stride. *Why:* naive stride aliases 48 kHz mic noise into the
  speech band, which made webrtcvad classify silence as speech **forever** (utterances
  ran to the 15 s cap). The box filter is a cheap anti-alias low-pass.
- Each captured utterance is **normalized to 0.5 peak** (the proven, ASR-friendly
  level — clipping was breaking ASR upstream).

### B.2 Energy gate, not webrtcvad
`webrtcvad` proved unreliable on this mic (aliasing + noise), so it's **off by
default** (`PRESENCE_USE_WEBRTCVAD=0`). The dependable signal is a **calibrated
energy gate**: a frame is "voiced" if its RMS exceeds a threshold (optionally
AND webrtcvad if re-enabled). `_is_voiced(frame, thresh)`.

### B.3 Per-room auto-calibration (`calibrate`)
Room noise differs everywhere, so we **measure it**:
- On **arm** and whenever the **room empties**, sample ~`CALIB_SECS` (1.5 s) of ambient.
- `noise_floor = percentile(frame_rms, CALIB_PCTL)` — `CALIB_PCTL=50` (median = the
  true quiet level; the 75th pct over-measured and missed speech onset).
- `speech_thresh (onset) = max(MIN_FLOOR, noise_floor × NOISE_MULT)` (1.7×)
- `continue_thresh (offset) = max(.., noise_floor × CONTINUE_MULT)` (1.3×)
- Shown live in the UI; the **Run VAD calibration** button triggers it on demand.
  USB: queued until the conversation is idle (the USB mic can't be opened twice —
  it failed with "Device unavailable"). Array: runs immediately on its own socket,
  even mid-conversation (multicast takes many readers).
- **Per source:** the array uses `ARRAY_MIN_FLOOR` (0.004) instead of `MIN_FLOOR`
  (0.010), and its onset is `floor × NOISE_MULT × ARRAY_ONSET_BOOST` (1.5×) — the array
  hears distant chatter, so the person in front must clear it by more.
- **Floor learned from every recording** (`_fold_utterance_floor`): the quietest 10% of
  a finished recording's frames (gaps between words, or the crowd bed) are folded into
  the floor (EMA 0.7/0.3). *Why:* in a crowd every frame can read as voiced, which
  starves the onset-window update (it only sees non-voiced frames). On the array the
  floor stayed at a quiet-moment calibration (0.006) while chatter sat at 0.013 — above
  onset — so distant conversations (Swedish, German, Portuguese) were transcribed as the
  visitor and every turn ran to the 15 s cap.

### B.4 Onset / offset with hysteresis + **adaptive end-detection** (`next_utterance`)
A fresh mic stream is opened per listen (see B.5), then:

**Onset** — wait for `ONSET_FRAMES` (3) consecutive frames above `speech_thresh`
(keeping a short `PREROLL` of audio so the first phoneme isn't clipped). If no
onset within `ONSET_TIMEOUT` (1 s) → return `timeout` and **refresh the floor**
from the silence just observed (so it tracks the *current* room).

**Record** — until trailing silence. The end threshold is **adaptive**, not the
static calibrated value:
```
recent = rolling ~1 s of frame RMS
noise_est = percentile(recent, 10)          # the true in-room floor right now
end_thresh = max(continue_thresh, noise_est × SIL_MARGIN)   # SIL_MARGIN = 1.8
# a frame is "silence" if rms < end_thresh; end after SILENCE_MS of it (or UTTER_MAX_SECS cap)
```
*Why adaptive:* a single calibrated threshold is fragile — if the room gets a bit
louder than at calibration time, trailing noise stays above it and the utterance
**never ends** ("records too long"). Keying the end off the *quietest recent level
within the utterance* (the gaps between words reveal the real floor) makes
end-detection survive calibration drift. This was the final fix. The first attempts
(static hysteresis, `SILENCE_MS` tweaks) over- or under-shot; this is the one that stuck.

Also: when onset fires, the floor is refreshed from the ~150 ms of pre-speech
silence in the onset window, so the offset threshold always reflects current noise.

### B.5 Mic stream lifecycle
**Every read has a timeout.** The USB stream runs in callback mode feeding a queue;
`_read_frame16` waits at most `MIC_READ_TIMEOUT` (1 s) and otherwise raises `MicStall`.
A stall drops that listen (`timeout`), the loop re-checks presence, and the next listen
opens a fresh stream. `close()` runs in a bounded thread (a wedged device can hang it).
*Why:* a blocking `stream.read()` on a wedged USB mic never returned — it froze the whole
presence loop mid-recording, detection included, and arm/disarm couldn't recover it.

The stream is **opened per-listen and closed before each network/playback step** —
**never left idle**. *Why:* holding a `sounddevice.InputStream` open across the
seconds of TTS playback overflows PortAudio's input buffer, and the next `read()`
**wedges** — that froze the 2nd conversation turn. PTT works the same way (open,
read, close), which is why it never hit this.

### B.6 Laptop / browser mic path
On the win-arm64 dev laptop PortAudio won't load, so native `VadCapture` is disabled
(`_HAVE_SD=False`, greet-only). The **browser Hold-to-Talk** is the mic there:
WebAudio captures PCM → downsample to 16 kHz → WAV → `POST /api/utterance` →
`PresenceController.converse_once()` runs one turn. No VAD (manual press/release).
It's also a remote PTT for the robot (secure-context caveat — see `CONTROL_CENTRE.md` §5).

### B.7 VAD params & tuning
| env | default | meaning |
|---|---|---|
| `PRESENCE_SILENCE_MS` | 800 | trailing silence that ends an utterance |
| `PRESENCE_NOISE_MULT` | 1.7 | onset threshold = floor × this |
| `PRESENCE_CONTINUE_MULT` | 1.3 | offset (continue) threshold = floor × this |
| `PRESENCE_SIL_MARGIN` | 1.8 | adaptive end = quietest-recent × this |
| `PRESENCE_CALIB_PCTL` | 50 | percentile of ambient = noise floor |
| `PRESENCE_MIN_FLOOR` | 0.010 | absolute floor (never below), USB mic |
| `PRESENCE_MIC` | `usb` (compose: `array`) | mic source: `array` or `usb` |
| `PRESENCE_ARRAY_MIN_FLOOR` | 0.004 | absolute floor, mic array |
| `PRESENCE_ARRAY_ONSET_BOOST` | 1.5 | array onset = floor × NOISE_MULT × this |
| `PRESENCE_MIC_READ_TIMEOUT` | 1.0 | seconds without audio before a mic stall |
| `PRESENCE_USE_WEBRTCVAD` | 0 | re-enable webrtcvad on top of the energy gate |
| `PRESENCE_TRANSCRIBE` | compose 1 | ask the gateway for the `heard` transcript (extra Gemma call) |
| `PRESENCE_DEBUG` | 0 | log per-frame RMS stats + onset/timeout/dur per utterance |

**Symptom → knob:**
- *Records too long / won't stop* → raise `NOISE_MULT`/`CONTINUE_MULT`, lower `SILENCE_MS`; confirm calibration ran (check the live floor in the UI).
- *Cuts me off mid-sentence* → raise `SILENCE_MS`, lower `CONTINUE_MULT`/`SIL_MARGIN`.
- *Misses my speech (no onset)* → lower `NOISE_MULT`; re-calibrate in a quiet moment.
- *Triggers on noise* → raise `NOISE_MULT`; raise `MIN_FLOOR`.
- *Array picks up other people's conversations* → raise `ARRAY_ONSET_BOOST` (2.0), or switch the dropdown to the USB mic.
- *Array ignores a normal voice up close* → lower `ARRAY_ONSET_BOOST` (1.2).
- Use the **Tuning** sliders in the control centre to dial these live, or `Run VAD calibration` after the room changes.

### B.8 VAD gotchas (recap)
- Anti-alias the 48→16 k downsample (box filter), or webrtcvad sees noise as speech.
- Energy gate > webrtcvad on this hardware (webrtcvad off by default).
- Calibrate per room; use the **median** for the floor.
- End-detection must be **adaptive** (in-utterance), not a static threshold.
- Open/close the mic stream per utterance (idle PortAudio stream wedges).
- Never block forever on a mic read — time out and reopen.
- The array only streams in voice-service mic mode (API 1008); re-enable on every arm.
- In a crowd, learn the floor from recordings too — non-voiced-frame updates starve.

---

## C. Code pointers (`presence.py`)
- `FaceDetector` — `nearest_face_frac()`, `_grab_bgr()` (uses `FRAME_SOURCE` when web is up), `dump_annotated()`, and module fn `_draw_overlay()`.
- `VadCapture` — `calibrate()`, `_update_floor()`, `_fold_utterance_floor()`, `_read_frame16()` (box-filter / array frames, `MicStall` timeout), `_is_voiced()`, `next_utterance()` (onset + adaptive record), `set_source()`, `_open_array()` / `_array_sample_rms()`, `onset_mult()`.
- `enable_array_mic()` — voice API 1008 mic mode.
- `PresenceController` — `run()`/`_tick()` (state machine), `_greet_and_converse()` (greet + loop + farewell), `_farewell()`/`_wave()`, `converse_once()` (browser PTT).
- Calibration/threshold config + `_prompt()` (hot-reload prompts) near the top.

