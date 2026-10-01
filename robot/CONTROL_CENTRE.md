# G1 Demo — Control Centre & Presence-Greeter

Session notes / handoff doc. Covers the web control centre, the presence-greeter
changes it instruments, how to run it (laptop + robot), all tuning knobs, the
verified gateway architecture, hard-won gotchas, and the plan for the next
session (Gemma-driven gestures).

Last updated: 2026-09-25 (AI in the City, Breda). **Companion doc:** `PRESENCE_VAD.md` —
deep-dive on the presence detection + VAD algorithms, parameters, and tuning.
**What changed at the event:** see §10.

---

## 1. What this is

`~/g1demo` is the robot-side client for a Unitree G1 "presence greeter". It runs
in a Docker container on the Orin (`unitree@192.168.123.164`), talks to an
**off-board Gemma 4 E4B gateway** (`http://<gateway-host>:2730`, repo at
the root of this repo), and never publishes motion
commands except a safe predefined arm wave.

This session added:
1. **Goodbye + wave** when a greeted person leaves.
2. **Robust VAD** (auto-calibrated energy gate + anti-alias downsample + per-utterance adaptive end-detection) and live **tuning controls**.
3. A full in-process **web Control Centre**: live video w/ CV overlay, an interactive pipeline diagram with live per-stage latencies, streamed logs, a live prompt-preset library, browser Hold-to-Talk, and detection/VAD tuning + calibration.
4. A **laptop dev harness** (`dev_run.py`) so the whole thing runs without the robot.

---

## 2. File map

**New:**
- `control_center.py` — the whole web app: `Hub` (shared state), `LogTee` (stdout capture), `PromptStore` (prompts.json), `Grabber` (sole camera reader), `FramePump` (annotated MJPEG), Flask routes, and the embedded single-page UI (HTML/CSS/JS as a string, SVG diagram built via `createElementNS`).
- `dev_run.py` — laptop entrypoint. Stubs the robot actuators (`LocalAudio` → winsound/sounddevice playback, no Unitree SDK, web-driven Arm/Talk). Reuses `presence.py` + `control_center.py`.
- `prompts.json` — named prompt presets `{active, presets:[{name,greet,converse,goodbye}]}`. **Must exist** (the compose mounts it `:rw`).
- `requirements-dev.txt` — laptop deps for `uv` (flask, numpy, opencv-python, sounddevice, httpx).

**Edited:**
- `presence.py` — added: goodbye+wave (`_farewell`, `_wave`), VAD calibration + adaptive end-detection, `_draw_overlay` (shared), hot-reload prompts (`_prompt()`), `converse_once()` (browser PTT), and guarded `_pub` instrumentation that streams state/metrics/heard/reply/speaking to the Hub. Globals: `HUB`, `PROMPTS`, `FRAME_SOURCE`.
- `g1_gemma_client.py` — arm client + `rt/lowstate` mode subscription; F2 toggles presence-mode, F1 = PTT; `PRESENCE_WEB` starts the control centre.
- `usb_cam.py` — cross-platform capture (`CAM_BACKEND` v4l2 on robot / default on laptop; `CAM_INDEX`/`CAM_DEV`).
- `usb_mic.py` — import-safe (degrades if PortAudio missing); device fallback (`MIC_INDEX` / default input).
- `docker-compose.jetson.yml` — web env + bind-mounts (`control_center.py:ro`, `prompts.json:rw`), camera mapped by stable **by-id** path, `PRESENCE_*` passthroughs.

---

## 3. Architecture (verified against the gateway code)

```
Camera ─┬─▶ Detector (Haar presence GATE — arms the loop; trigger only)
        │
        └─image─▶ ┌─ GATEWAY · Gemma 4 E4B · vLLM/RTX 6000 ──────────────┐
Mic/VAD ─audio──▶ │  decode ─┬─▶ Gemma 4B (reason+see, audio-native) ─▶ kokoro (TTS) ─┼─▶ G1 Speaker
                  │          └─▶ ASR (transcript — 2nd Gemma call, optional)         │
                  └──────────────────────────────── round-trip Σ ───────────────────┘
```

Key facts (confirmed in `gemma-voice-agent/app/main.py` + `metrics.py`):
- **Gemma is audio-native** — it reasons over the decoded audio + image directly; the reply is the `llm` stage.
- **ASR is by default a *second Gemma call*** (`_transcribe()` with a transcription system prompt). It runs only when `transcribe=true`, purely to show the HEARD text; the reply doesn't depend on it. Hence the diagram shows it as a parallel branch off `decode`.
- **Parakeet + cascade (newer gateways):** the gateway can also transcribe with Parakeet TDT (`asr_engine=parakeet`, ~25 ms instead of a few hundred), and with `llm_input=transcript` Gemma answers from that transcript instead of the audio. Both are switchable from the control centre's *Gateway pipeline* card.
- **No control/gesture output from the gateway** — `/converse` returns reply text only → TTS. The G1 wave is an on-robot reflex (`presence._farewell`), not a model output. (See §9.)
- Gateway `done` event carries `metrics.components` = `[audio_decode, asr, llm, tts]` plus `ttft_ms`, `generation_ms`, `tokens_per_sec`, `time_to_first_audio_ms`, `tts_audio_seconds`, `prefix_cache_hit`, `total_ms`.

---

## 4. Control Centre features

- **Presence feed** — live MJPEG. CV overlay (face boxes + `"1 face · 24% close (need 18%) · ARMED · greeted"`) shown **only when armed**; clean preview when idle. Arm / Disarm / **Hold-to-Talk** buttons + a live **mic waveform** canvas (green idle, magenta recording).
- **Pipeline diagram, per mode** — the diagram follows the conversation link. *HTTP*: decode → Gemma / ASR → kokoro, total = round-trip Σ. *WebSocket*: **Mic · open** (echo cancellation state + measured speaker delay) → gateway **VAD** (`endpoint` wait) → **Smart Turn** (inference ms, probability, done/unfinished) → **hold** (`reopen_grace`) → Gemma / ASR; total = *stop talking → first audio* (endpoint + gateway time to first audio; the stages overlap, so they are not summed); the Speaker shows barge-in on / half-duplex and turns amber when a reply was cut off. *Cascade* (Gemma answers from the transcript): a violet **ASR → Gemma “transcript”** arrow appears, the audio → Gemma arrow fades, and the ASR box reads `<engine> · feeds Gemma` (vs `· display only`, or `off`).
- **Pipeline diagram** — JS-built SVG. Per-stage latency + autoscaled bars on the gateway stages; animated flow + gateway glow during a turn; **equalizer animation on the Speaker** while talking; live **round-trip Σ** aggregate; per-component **hover tooltips**; stat tiles (total time / to first word / words-per-sec / to first sound).
- **Live streaming** — HEARD appears on transcript, REPLY streams token-by-token, and the per-stage boxes fill in the moment generation finishes (during playback, not after).
- **Logs** — stdout tee → SSE console (docker logs still works).
- **Prompt presets** — named presets (greet/converse/goodbye); `+ New`, Save, Activate (live, no restart), Delete. The "converse" field is George's system prompt/persona.
- **Conversation link (experimental)** — switch between the HTTP loop (one utterance per request, robot VAD) and the **streamed WebSocket mode**, where the gateway does the turn-taking (Silero VAD + Smart Turn) and the visitor can **barge in**. Toggles for barge-in, echo cancellation and its noise suppression / auto gain; mic gain; and live sliders for this robot's overrides of the gateway's turn detection (speech threshold, speech needed to start a turn, closing silence, pre-roll, the reply-hold graces, Smart Turn on/off and threshold). See §11.
- **Gateway pipeline** — two live dropdowns, applied from the next turn: who writes the HEARD transcript (off / Gemma / Parakeet), and what Gemma answers from (the audio, or a Parakeet transcript = cascade). Options the gateway hasn't loaded are greyed out; the ASR box in the diagram shows which engine ran.
- **Tuning** — live sliders: proximity gate, arm/disarm seconds, end-of-speech ms, VAD onset ×; **mic source** dropdown (G1 mic array / USB mic — switches on the next listen and recalibrates); **Run VAD calibration** button (array: runs live, even mid-conversation; USB: queued until idle, because the USB mic can't be opened twice).

---

## 5. Running it

### Laptop dev (no robot)
```bash
cd robot
uv venv
uv pip install -r requirements-dev.txt
$env:PRESENCE_WEB_PASSWORD="g1demo"; uv run python dev_run.py
# open http://localhost:8080   (user: anything, password: g1demo; empty password = open)
```
Laptop reality: native `sounddevice`/PortAudio won't load on this win-arm64 box, so **playback uses winsound** and the **mic is the browser Hold-to-Talk** (native VAD auto-disabled). The webcam, gateway, diagram, prompts, logs all work for real.

### On the robot
```bash
# from the laptop — push the working set (NOT dev_run.py / requirements-dev.txt):
scp control_center.py presence.py g1_gemma_client.py usb_cam.py usb_mic.py \
    streaming.py aec.py aec_server.py Dockerfile.aec \
    docker-compose.jetson.yml prompts.json unitree@192.168.123.164:~/g1demo/
# on the robot:
cd ~/g1demo
PRESENCE_WEB=1 PRESENCE_WEB_PASSWORD='pick-one' GEMMA_URL=http://<gateway-host>:2730 \
  docker compose -f docker-compose.jetson.yml up -d --no-deps --force-recreate gemma-client
docker logs --tail 12 gemma-g1-client      # expect: [web] control centre on :8080
# open http://192.168.123.164:8080
```
On the robot the real mic/VAD, kokoro→head speaker, V4L2 camera, and arm wave all run; **Run VAD calibration** actually calibrates.

### Browser mic over the network (secure-context)
`getUserMedia` only works on HTTPS or `localhost`, so the browser Hold-to-Talk is **blocked** when a remote browser hits `http://192.168.123.164:8080` (the robot uses its own mic anyway). To use the *browser* mic remotely, pick one:
1. **SSH tunnel** (zero config): `ssh -L 8080:localhost:8080 unitree@192.168.123.164` → open `http://localhost:8080`.
2. **Whitelist the origin** (per browser): Chrome/Edge `chrome://flags/#unsafely-treat-insecure-origin-as-secure` → add `http://192.168.123.164:8080` → Enabled → relaunch. (Firefox: `about:config` → `media.devices.insecure.enabled` + `media.getusermedia.insecure.enabled`.)
3. **HTTPS** (works everywhere, needs a cert) — not yet implemented; would add `PRESENCE_WEB_CERT/KEY` → Flask `ssl_context`.

---

## 6. Endpoints
`GET /` UI · `GET /video.mjpg` · `GET /last_image.jpg` · `GET /logs` (SSE) · `GET /state` (SSE) · `GET|POST /api/prompts` · `DELETE /api/prompts/<name>` · `POST /api/prompts/active` · `POST /api/control {arm|disarm|toggle}` · `POST /api/utterance` (raw WAV) · `GET|POST /api/stream {transport, barge_in, aec, aec_noise_suppression, aec_gain_control, aec_delay_ms, mic_gain, vad:{…}, reset_vad}` · `GET|POST /api/gateway {transcribe, asr_engine, llm_input}` · `GET|POST /api/tune` (GET also returns `mic_source`) · `POST /api/calibrate` · `POST /api/mic {"source": "array"|"usb"}`. All behind HTTP Basic (empty password = open).

---

## 7. Tuning knobs (env, all live-overridable)
- **Web:** `PRESENCE_WEB`(1), `PRESENCE_WEB_PORT`(8080), `PRESENCE_WEB_PASSWORD`(""=open), `PRESENCE_WEB_FPS`(14), `PRESENCE_WEB_MAX_CLIENTS`(8), `PRESENCE_PROMPTS_PATH`.
- **Presence:** `PRESENCE_PROX_FRAC`(0.18), `PRESENCE_ARM_SECS`(1.0), `PRESENCE_DISARM_SECS`(2.0), `PRESENCE_STAY_FRAC`(0.12 — looser face gate once a conversation is running), `PRESENCE_CONV_DISARM_SECS`(5.0 — absence before a mid-conversation goodbye), `PRESENCE_HAAR_NEIGHBORS`(4), `PRESENCE_HAAR_MIN_PX`(60).
- **Mic source:** `PRESENCE_MIC`(compose `array`, code `usb`), `PRESENCE_ARRAY_MIN_FLOOR`(0.004), `PRESENCE_ARRAY_ONSET_BOOST`(1.5 — array onset = floor × NOISE_MULT × this), `PRESENCE_ARRAY_LOCAL_IP`(192.168.123.164), `PRESENCE_MIC_READ_TIMEOUT`(1.0 s — a read that waits longer = mic stall).
- **Conversation link:** `PRESENCE_TRANSPORT`(`http` | `ws`), `PRESENCE_BARGE_IN`(1), `PRESENCE_AEC`(1), `PRESENCE_AEC_URL`(127.0.0.1:5005 — the `aec` sidecar), `PRESENCE_AEC_DELAY_MS`(auto), `PRESENCE_AEC_NS`(1), `PRESENCE_AEC_AGC`(1), `PRESENCE_STREAM_MIC_GAIN`(auto), `PRESENCE_STREAM_ARRAY_GAIN`(4.0).
- **Gateway pipeline:** `PRESENCE_TRANSCRIBE`(compose 1), `PRESENCE_ASR_ENGINE`("" = gateway default; `gemma` | `parakeet`), `PRESENCE_LLM_INPUT`("" = gateway default; `audio` | `transcript`). Needs a gateway with Parakeet loaded; older gateways ignore the extra fields.
- **VAD:** `PRESENCE_SILENCE_MS`(compose 900), `PRESENCE_NOISE_MULT`(1.7), `PRESENCE_CONTINUE_MULT`(1.3), `PRESENCE_SIL_MARGIN`(1.8), `PRESENCE_CALIB_PCTL`(50), `PRESENCE_MIN_FLOOR`(0.010, USB), `PRESENCE_USE_WEBRTCVAD`(0 — energy VAD default), `PRESENCE_RECALIB_SECS`(30), `PRESENCE_RECALIB_AVG_N`(5), `PRESENCE_FLOOR_TRIM`(0.9).
- **Wave/diag:** `PRESENCE_WAVE`(1), `PRESENCE_WAVE_ACTION`(25), `PRESENCE_WAVE_HOLD`(2.5), `PRESENCE_TRANSCRIBE`(compose 1), `PRESENCE_DEBUG`(0).
- **Camera/mic:** `CAM_BACKEND`(v4l2 on Linux), `CAM_DEV`(6)/`CAM_INDEX`(0), `CAM_W/H`, `MIC_INDEX`, `PRESENCE_DEV_GAIN`(1.0, laptop).
- **Prompts:** edited live in the UI (or `PRESENCE_GREET/CONVERSE/GOODBYE_INSTRUCTION` as seeds).

---

## 8. Gotchas / lessons (so we don't re-learn them)
- **SVG must be built with `createElementNS`** — static inline SVG in an HTML string gets mangled by the HTML parser (lowercases `linearGradient`, drops `<g>`).
- **`_draw_overlay` must not do `faces or ()`** — `faces` is a numpy array; `bool(ndarray)` raises "ambiguous truth value" when a face is found.
- **Camera lag = capture backlog.** A single `Grabber` thread is the sole reader; everything reads its latest snapshot (`FRAME_SOURCE`) so the driver buffer never piles up. If the feed freezes, restart (old process holding the handle) — kill, wait 3s for Windows to release, relaunch.
- **Headless screenshots time out** on the live MJPEG/SSE page (never idle) — use `getBBox`/`preview_inspect` to verify layout numerically; screenshots only work on a fresh load.
- **win-arm64**: no PortAudio → winsound playback + browser mic; webrtcvad has no wheel → energy VAD default.
- **Arrowheads**: lines end `GAP=12` before each node; the base-anchored 12-unit marker bridges the gap so the line meets the arrow's back-centre and the tip touches the box.
- Background `dev_run.py` over the tool harness reports exit 255 on kill — harmless.
- **Gateway returns `event: session` then drops** = vLLM is down or rejecting the model name, not a robot problem. Check `curl <gateway>/health` → `backend_info.reachable` and `model`. Fixed gateway-side in gemma-voice-agent PR #3 (multi-audio crash + model re-discovery); vLLM now `restart: unless-stopped`.
- **Gateway only reachable through a laptop VPN** (e.g. on a phone hotspot): reverse-tunnel it to the robot and point the client at the tunnel. From the laptop: `ssh -N -R 127.0.0.1:2731:<gateway-host>:2730 -L 127.0.0.1:8080:127.0.0.1:8080 unitree@<robot-ip>` (wrap in a reconnect loop — hotspots drop connections), then recreate the client with `GEMMA_URL=http://127.0.0.1:2731`. The container keeps that env across reboots; the tunnel does not.
- **Browser Hold-to-Talk needs a secure context** — use the `-L 8080` forward above and open `http://localhost:8080`.

---

## 9. NEXT SESSION — Gemma-driven gesture control

Today the wave is a local reflex; gestures are **not** model-driven. Plan to make them so:
1. **Prompt**: add to the converse system prompt that Gemma may emit a tag like `[wave]`, `[nod]`, `[point]` inline in its reply.
2. **Parse** the tag(s) out of the reply text (robot-side in `presence._converse_turn`, or gateway-side), speak the cleaned text, and **fire `G1ArmActionClient.ExecuteAction(<id>)`** — reuse the proven gesture map + mode gate (`SAFE_MODES={5}`) from `g1_greeter_shadow.py` (wave=25, release=99, plus shake_hand 27, etc.).
3. **Diagram**: add a "Gesture / Arm" node with an arrow from **Gemma** (a real control-output path), and publish a `gesture` event to the Hub to animate it.
4. Decide where parsing lives (robot vs gateway) — robot-side keeps the gateway generic; gateway-side could return a structured `gesture` field.
- Gateway repo: the root of this repo (`app/main.py` `/converse`). Arm action IDs verified on-robot in `g1_greeter_shadow.py`.

---

## 10. Changes at AI in the City (Breda, 2026-09-24/25)

Found and fixed live on the robot at the BUas Makerspace stand.

| Problem seen | Cause | Fix |
|---|---|---|
| Robot repeats part of a reply / skips a sentence | Every TTS sentence was written to one shared temp WAV at *enqueue* time; the playback queue held the path, so a later sentence overwrote an earlier queued one | Queue the audio bytes; write the temp file right before playing (`presence._converse_turn`) |
| Face centred in frame not detected | Global `equalizeHist` crushes the face when a big bright wall dominates the histogram | CLAHE (tiled local contrast) instead — still handles the backlit-window case (`FaceDetector`) |
| Goodbye + wave while the visitor is still there | Mid-conversation leave used the arrival gate (18% face, 2 s) — a glance away ended it | Looser leash once talking: `STAY_FRAC` 0.12, `CONV_DISARM_SECS` 5 |
| Detection freezes; arm/disarm doesn't recover | Blocking `stream.read()` on a wedged USB mic hung the whole presence loop mid-recording | Callback stream → queue with `MIC_READ_TIMEOUT`; a stall drops that listen and reopens; `close()` bounded |
| Calibrate during a conversation → "Device unavailable" | Button opened a 2nd stream on the busy USB mic | USB: queue until idle. Array: runs live on its own socket |
| USB mic hears the hall badly (8.6 dB SNR) | Cheap USB dongle | **G1 built-in mic array** (27 dB SNR) — see below |
| Array hears the whole hall; turns run to the 15 s cap | Floor calibrated in a quiet moment; crowd chatter sat above onset so the in-listen floor update (non-voiced frames only) never ran | Array onset × `ARRAY_ONSET_BOOST` (1.5); floor also learned from the quietest 10% of every finished recording (`_fold_utterance_floor`) |
| Robot loops "Hello, I'm George…" | Gemma couldn't make out the audio | Prompt now asks it to request a repeat instead of guessing |

**G1 mic array.** Streams 16 kHz mono s16le on UDP multicast `239.168.123.161:5555`
**only while the voice service's mic mode is on**: undocumented voice API **1008**
`{"mode": 1}` (off: `{"mode": 2}`; the app / remote L1+L2 "wake-up mode" flips the
same switch). `presence.enable_array_mic()` calls it on every arm (it doesn't
survive a power cycle). `PRESENCE_MIC=array` (compose default) reads the array;
2 consecutive stalls fall back to the USB mic and recalibrate. Switch live from
the control-centre **mic source** dropdown (`POST /api/mic`). The array needs no
PortAudio, multiple readers can join the group (so calibration runs alongside a
listen), and its floor is ~4× lower than the USB mic's.

**Prompt profile `ai-in-the-city`** (`prompts.json`, active): George at AI in the
City, Breda, for the BUas Makerspace; tells visitors to speak when the head light is
green (LED: green = listening, amber = thinking, blue = speaking); never invents
makerspace facts; asks for a repeat when unsure.

**Gateway (gemma-voice-agent PR #3):** past turns now go to vLLM as text only
(multiple audio clips in one request crashed vLLM 0.26's Gemma-4 audio encoder);
served-model name re-discovered after a vLLM restart; `vllm` restarts itself.

---

## 11. Streamed (WebSocket) mode, barge-in and echo cancellation — experimental

**Status: built and exercised in simulation only. It has not run on the robot.**

### What it is
`presence.STREAM["transport"] = "ws"` (control centre → *Conversation link*) swaps the conversation loop: after the usual HTTP greeting, the mic is streamed to the gateway's `/ws/converse` (`streaming.py`) instead of being cut into utterances by the on-robot energy VAD. The gateway then decides when a turn starts and ends (Silero VAD + Smart Turn, the same logic as the Reachy Mini backend) and can be interrupted mid-reply. The goodbye still goes through `/converse`. Everything else (presence detection, prompts, the pipeline diagram, HEARD/REPLY) works as before. If the WebSocket can't be opened (old gateway, network), that visitor falls back to the HTTP loop.

No new Python packages are needed on the robot for the link itself: `streaming.py` carries a small standard-library WebSocket client.

### Echo cancellation
With barge-in the mic is open while George talks, so his own voice must be removed from it or he interrupts himself. `aec.py` uses GStreamer's `webrtcdsp` + `webrtcechoprobe`, the elements the Reachy Mini SDK uses for software AEC. Two things differ from Reachy, both forced by the G1:

- **No local playback to tap.** Reachy plays through a sound card, so its echo probe sits in the playback pipeline. George's speaker is driven over DDS (`PlayStream`), so the player reports each clip as it is sent and `aec.py` lays it on the mic's sample clock.
- **It runs in a sidecar container.** The canceller needs GStreamer ≥ 1.24 (WebRTC "AEC3", which finds the speaker delay by itself). Jetson images are Ubuntu 20.04/22.04, whose GStreamer has the older canceller: in simulation that one only works when the reference is placed within ~20 ms of a 40 ms target, which a DDS speaker path is unlikely to hold. So `Dockerfile.aec` builds a small Debian image with a current GStreamer and `aec_server.py`; the client talks to it on `127.0.0.1:5005`.

```bash
# once, on the Orin (needs internet; pulls debian:trixie-slim)
docker compose -f docker-compose.jetson.yml build aec
docker compose -f docker-compose.jetson.yml up -d aec
docker logs gemma-aec        # expect: [aec] listening on 127.0.0.1:5005 (AEC3)
# then copy streaming.py + aec.py to ~/g1demo with the other client files and recreate gemma-client
```

Safety net: if echo cancellation is switched on but not reachable, the client keeps barge-in off (the gateway ignores the mic while George is replying or playing), so he can't interrupt himself. The status line in the card says which state he is in.

For reference, Reachy's real robot doesn't use this software path at all: its XMOS mic array cancels echo in hardware. Whether the G1's own array stream is already echo-cancelled is unknown; a quick check is to switch echo cancellation **off**, leave barge-in **on**, and see whether George cuts himself off.

### What to check on the robot, in this order
1. **WebSocket, barge-in off.** Does he hear turns and answer? If speech is missed, raise *mic gain* (the array is quiet) or lower *speech threshold*; if noise starts turns, do the opposite or raise *speech needed to start a turn*.
2. **`PlayStop`.** Barge-in cuts playback with `AudioClient.PlayStop("gemma")`. Untested on the G1.
3. **Echo cancellation + barge-in.** Watch the status line: it reports the measured *speaker delay*. George should finish his sentences when nobody speaks, and stop within about half a second when someone does.
4. **Crowd noise.** Silero is more robust than an energy gate, but it has never heard this mic in a hall. The HTTP loop remains the proven fallback.

### Simulated results (off-robot, real gateway)
A fake mic that hears a recorded visitor plus the fake speaker's output, delayed 150 ms:

| Setup | Result |
|---|---|
| No echo cancellation, barge-in on | George interrupted himself on every reply (9 false barge-ins in 24 s) |
| AEC3 (in-process or via the sidecar) | 4 sentences played with no false barge-in; a visitor talking over him stopped playback in ~0.5 s and was transcribed correctly |
| Old canceller, reference misaligned by > ~20 ms | no cancellation |
