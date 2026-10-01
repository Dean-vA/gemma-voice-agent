"""Audio-native /converse client: mic audio in -> gateway (ASR+Gemma+TTS) ->
WAV-per-sentence out -> G1 head speaker via PlayStream. No local Whisper/Piper.
SSE framing matches stream_gemma: event: lines + JSON data: payloads."""
import os, time, json, base64, queue, threading, tempfile
import httpx

GEMMA_URL    = os.environ.get("GEMMA_URL", "http://localhost:2730").rstrip("/")
SESSION_ID   = os.environ.get("SESSION_ID", "")
TTS_ENGINE   = os.environ.get("TTS_ENGINE", "kokoro")   # only reachable engine
INSTRUCTION  = os.environ.get("INSTRUCTION", "")
HTTP_TIMEOUT = 60.0

def _extract_audio_b64(payload):
    """The audio event's JSON may carry the wav under a few possible keys."""
    for k in ("wav_base64", "audio", "data", "wav", "pcm"):
        v = payload.get(k)
        if isinstance(v, str) and v:
            return v
    return None

def run_converse(audio, wav_bytes, image_bytes, set_led, wav_to_pcm16k, gain, seq_ref):
    """POST mic audio to /converse, play each 'audio' SSE event as it arrives.
    Returns (heard_text, reply_text) for logging."""
    url   = f"{GEMMA_URL}/converse"
    files = {"audio": ("speech.wav", wav_bytes, "audio/wav")}
    data  = {"engine": TTS_ENGINE, "transcribe": "true"}
    if INSTRUCTION:
        data["instruction"] = INSTRUCTION
    if SESSION_ID:
        data["session_id"] = SESSION_ID
    if image_bytes:
        files["image"] = ("frame.jpg", image_bytes, "image/jpeg")
        print(f"  [cam] sending {len(image_bytes)} byte frame")
    else:
        print("  [cam] no image this turn")

    say_q = queue.Queue(); DONE = object()
    def worker():
        while True:
            item = say_q.get()
            if item is DONE:
                return
            try:
                path = os.path.join(tempfile.gettempdir(), "g1_converse.wav")
                with open(path, "wb") as f:
                    f.write(item)
                pcm, dur = wav_to_pcm16k(path, gain)   # handles 24k(kokoro)->16k
                seq_ref[0] += 1
                audio.PlayStream("gemma", str(seq_ref[0]), pcm)
                time.sleep(dur + 0.2)
            except Exception as e:
                print(f"[tts] play failed: {e}")
    threading.Thread(target=worker, daemon=True).start()

    reply_parts, heard = [], ""
    spoke = {"led": False}
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
                    heard = payload.get("text", "")
                    print(f"  heard: {heard!r}")
                elif event == "token":
                    tok = payload.get("text", "")
                    if tok:
                        reply_parts.append(tok)
                elif event == "audio":
                    b64 = _extract_audio_b64(payload)
                    if b64:
                        if not spoke["led"]:
                            set_led(audio, 0, 80, 160); spoke["led"] = True
                        try:
                            say_q.put(base64.b64decode(b64))
                        except Exception as e:
                            print(f"[audio] decode failed: {e}")
                    else:
                        print(f"[audio] event had no recognizable audio key: {list(payload)}")
                elif event == "done":
                    m = payload.get("metrics") or {}
                    if m.get("ttft_ms") is not None:
                        print(f"  [gateway] TTFT={m['ttft_ms']}ms backend={m.get('backend')}")
    say_q.put(DONE)
    return heard, payload_reply(reply_parts)

def payload_reply(parts):
    return "".join(parts)
