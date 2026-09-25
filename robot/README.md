# Unitree G1 presence greeter (thin client, off-board Gemma)

The **robot-side** client for the Gemma voice agent. It runs on the G1's Jetson
Orin, talks to the gateway at the repo root over HTTP, and never runs a model
locally. It is fully additive: it doesn't change the root `docker-compose.yml`.

Built for (and hardened at) the **AI in the City** event in Breda, where it ran
as "George" at the BUas Makerspace stand.

## What it does

- **Hands-free presence greeting.** A USB camera + Haar face detector (CLAHE
  contrast) spots a visitor close to the robot. George greets them with a line
  grounded in what the camera sees, then holds a hands-free conversation.
- **Voice in.** The G1's built-in 4-mic array (default) or a USB mic, segmented
  by an auto-calibrated energy VAD. Each utterance plus a camera frame goes to the
  gateway's `POST /converse`; Gemma 4 E4B hears the audio directly.
- **Voice out.** The kokoro TTS audio streamed back is played sentence by sentence
  on the head speaker (`AudioClient.PlayStream` over DDS). The head LED shows the
  state: green = listening, amber = thinking, blue = speaking.
- **Goodbye + wave** when the visitor walks away (a predefined arm action, only
  in the standing mode). Otherwise the client never sends motion commands.
- **Web control centre** on `:8080`: live camera with detection overlay, pipeline
  diagram with per-stage latency, logs, live prompt presets, tuning sliders, mic
  source switch, VAD calibration, and a browser Hold-to-Talk.
- **F1 push-to-talk** is still available as a manual fallback; **F2** toggles
  presence mode.

```
Camera ─▶ face detector (gate) ──────┐ frame
G1 mic array / USB mic ─▶ VAD ───────┼─ audio ─▶ gateway /converse (Gemma 4 E4B, kokoro)
                                     │                 │ SSE: text + audio per sentence
G1 head speaker ◀─ DDS PlayStream ◀──┴─────────────────┘
```

## Files

| File | Role |
|---|---|
| `g1_gemma_client.py` | Entrypoint: DDS init, speaker/LED/arm clients, F1/F2, starts presence + web |
| `presence.py` | Face detection, VAD, greet → converse → goodbye state machine |
| `control_center.py` | In-process Flask control centre (single-page UI embedded) |
| `usb_cam.py`, `usb_mic.py` | Camera handle, USB mic discovery |
| `converse_mode.py` | `/converse` client used by F1 push-to-talk |
| `prompts.json` | Prompt presets (greet / converse / goodbye); edited live from the UI |
| `docker-compose.jetson.yml` | Robot deployment (Piper + client containers) |
| `dev_run.py`, `requirements-dev.txt` | Run the whole thing on a laptop without the robot |
| `preflight.py` | Robot/network sanity checks |
| `CONTROL_CENTRE.md` | Handoff doc: features, running, endpoints, knobs, gotchas, event changelog |
| `PRESENCE_VAD.md` | Deep-dive: presence detection + VAD design and tuning |

## Prerequisites on the robot

- Docker with the `g1-greeter-shadow:latest` image (carries `unitree_sdk2py`,
  `cyclonedds`, `piper-tts`, `httpx`, `espeak-ng`).
- A USB camera. The compose file maps a CyberTrack H3 by its stable
  `/dev/v4l/by-id/...` path; change it for another camera.
- The robot must reach the gateway. The DDS side (speaker, arm, mic array) is on
  `eth0` (`192.168.123.x`); `network_mode: host` gives the client both.

## Run

Copy this folder to the robot (e.g. `~/g1demo`), then:

```bash
cd ~/g1demo
GEMMA_URL=http://<gateway-host>:2730 PRESENCE_WEB_PASSWORD=<pick-one> \
  docker compose -f docker-compose.jetson.yml up -d
```

`GEMMA_URL` and `PRESENCE_WEB_PASSWORD` are required. The container restarts
with the robot and keeps these settings. Open `http://<robot-ip>:8080` (any user,
that password) and press **Arm** (or **F2** in the client's terminal). Presence mode
starts disarmed after every restart.

After editing a file, recreate the client (the code is bind-mounted read-only):

```bash
docker compose -f docker-compose.jetson.yml up -d --no-deps --force-recreate gemma-client
```

### Gateway only reachable through a VPN

If only a laptop on a VPN can reach the gateway, reverse-tunnel it to the robot
and point the client at the tunnel:

```bash
ssh -N -R 127.0.0.1:2731:<gateway-host>:2730 -L 127.0.0.1:8080:127.0.0.1:8080 unitree@<robot-ip>
```

Then recreate the client with `GEMMA_URL=http://127.0.0.1:2731`. The `-L` part
also makes `http://localhost:8080` a secure context, which the browser
Hold-to-Talk needs.

## Mic array

The G1's 4-mic array streams 16 kHz mono on UDP multicast
`239.168.123.161:5555`, but only while the voice service's mic mode is on.
The client switches it on (voice API 1008, `{"mode": 1}`) every time presence mode
is armed. At the event it measured 27 dB signal-to-noise against 8.6 dB for a
USB mic. It also hears more of the room, so its speech threshold sits higher
(`PRESENCE_ARRAY_ONSET_BOOST`). If the array stops streaming, the client falls
back to the USB mic. Switch between them with `PRESENCE_MIC=array|usb` or live
from the control centre.

## Key settings (env)

| Var | Default | Meaning |
|---|---|---|
| `GEMMA_URL` | required | gateway base URL |
| `PRESENCE_WEB_PASSWORD` | required | control-centre password |
| `PRESENCE_MIC` | `array` | mic source: `array` or `usb` |
| `PRESENCE_PROX_FRAC` | `0.18` | how close a face must be to greet (face width ÷ frame width) |
| `PRESENCE_STAY_FRAC` / `PRESENCE_CONV_DISARM_SECS` | `0.12` / `5.0` | looser gate once a conversation is running |
| `PRESENCE_SILENCE_MS` | `900` | pause that ends an utterance |
| `PRESENCE_NOISE_MULT` | `1.7` | speech threshold = noise floor × this |
| `PRESENCE_ARRAY_ONSET_BOOST` | `1.5` | extra threshold factor for the mic array |
| `PRESENCE_TRANSCRIBE` | `1` | show what the robot heard (extra Gemma call) |
| `PRESENCE_WAVE` | `1` | wave on goodbye |

The full list, with the reasoning behind each value, is in `CONTROL_CENTRE.md`
§7 and `PRESENCE_VAD.md`.
