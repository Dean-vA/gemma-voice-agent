# The Reachy speech pipeline, for comparison

The backend the Reachy Mini runs on when it plays Ruxithid, packaged so the G1 can talk
through it instead of through the gateway. It is the same code, unchanged. That lets the two
pipelines be compared end to end on the robot:

| | Gateway (repo root) | This pipeline |
|---|---|---|
| Turn-taking | `app/realtime.py`, a port of speech-to-speech's | speech-to-speech 1.0.0 itself |
| Hearing | Gemma 4 E4B hears the audio | Parakeet transcribes, TitaNet names known voices |
| Brain | Gemma 4 E4B (vLLM), sees the camera | Gemma 4 12B Q4_0 (llama.cpp), text only |
| Voice | kokoro | Qwen3-TTS, voice described in words |
| Protocol | `/ws/converse` | OpenAI-realtime WebSocket, `/v1/realtime` |

## Where the code comes from

`app/` is copied from the Reachy project by `sync.sh`; that project is the source of truth.
Only code is copied: voiceprints, the player roster, table logs, profiles and campaign notes
stay there, because this repo is public. Two options were added there for containers:
`backend.py --stt-url/--llm-url` (use services that are already running instead of starting
them) and `speaker_id/server.py serve --host`.

The speech server patches private parts of speech-to-speech (gesture tags, scene memory,
remembering only what was heard, the table log), so it is pinned to 1.0.0 and to the other
versions it was tuned with (`requirements.txt`).

## Running it

One container, one GPU (about 22 GB used): `Dockerfile` starts what `backend.py` starts on
Windows, in the same way. The image is `ghcr.io/dean-va/s2s-backend`, built and pushed by
`.github/workflows/s2s-images.yml` (on pushes to main or `feat/s2s-backend`, or by hand).

- **Portainer stack on "Student apps (cluster)":** `portainer-s2s.yml` at the repo root. That
  environment is a Swarm API in front of the adsai-rke2 cluster, so the stack publishes a
  node port and asks for the GPU as a Swarm generic resource. The G1 connects to
  `ws://<node>:32740/v1/realtime`.
- **Kubernetes directly:** `k8s/s2s.yaml`, same address.
- **A plain Docker host, or locally:** `docker compose -f s2s/docker-compose.yml up --build`;
  the G1 connects to `ws://<host>:2740/v1/realtime`.

The first start downloads about 15 GB of models into the cache volume.

## Differences from the Reachy's own setup

- No on-screen head, no PC audio (Discord) and so no DM-only mode: there is no screen or
  Discord on a server.
- Nobody is enrolled, so the transcript never names a speaker. A robot meeting strangers
  shouldn't call one of them by a player's name.
- The voice is George's (set with `--voice`), not the goblin's.
- `table_log.py` writes what was heard and said to the logs volume, one file per day.
  At a public event that is visitors' speech: clear the volume when it is no longer needed.
- llama.cpp is build b11429, the nearest published image to the Windows b11406.
