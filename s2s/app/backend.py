"""Run Ruxithid's self-hosted speech backend on this PC.

Starts three things and wires them together:
  1. the speaker-aware transcription service (speaker_id/server.py)
  2. llama.cpp serving the language model
  3. Hugging Face's speech-to-speech server, in this process
  4. his on-screen head (head.py), a body for him that needs no robot

It also mixes what this PC is playing (the players on Discord) into what he hears; see pc_audio.py.

The robot connects to it once you run `uv run ruxithid.py backend local`.

Run it with the backend environment's Python and leave it running:

    C:\\Users\\deanv\\ruxithid-backend\\.venv\\Scripts\\python backend.py
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import dm_only
import gestures
import heard
import memory
import pc_audio
import table_log

HERE = Path(__file__).parent

# In a container (the gemma-voice-agent repo's s2s/Dockerfile) these BACKEND_* settings say where
# that image keeps each part; on this PC they are unset and the paths below are used.

# The transcription service needs torch + NeMo; this environment already has both models cached.
STT_PYTHON = Path(os.environ.get("BACKEND_STT_PYTHON") or HERE.parent / "DnD-Session-Summaries" / ".venv" / "Scripts" / "python.exe")
STT_PORT = 8790

# llama.cpp's official CUDA build (release b11406, win-cuda-12.4-x64), unpacked here. It reads his
# prompt about 50% faster than the Vulkan build that `winget install llama.cpp` provides.
LLAMA_SERVER = Path(os.environ.get("BACKEND_LLAMA_SERVER") or r"C:\Users\deanv\ruxithid-backend\llama-cuda\llama-server.exe")
LLM_PORT = 8080  # preferred; another free port is used when Windows has this one reserved
LLM_MODEL = "ggml-org/gemma-4-12B-it-GGUF:Q4_0"
LLM_DEVICE = "CUDA0"
LLM_CONTEXT = 16384  # per slot; his prompt takes about 4,000 tokens of this
# In a long scene the speech server asks the model to summarise the older exchanges, in the
# background (see memory.py). With one slot that summary held up his next reply for about four
# seconds and threw away the model's memory of his prompt; with two, it runs beside the conversation.
LLM_SLOTS = 2
LLM_LOG = Path(os.environ.get("BACKEND_LLM_LOG") or LLAMA_SERVER.parent.parent / "llama.log")  # what llama.cpp prints, including how long each reply took it

PORT = 8765
# When someone pauses and Smart Turn thinks they haven't finished, the server waits this long for them
# to go on before he answers, and any talk resuming within it joins the same turn. At a live table
# the default 2000 ms let one turn swallow 50 seconds of everyone's chatter, so he never got a word in.
TURN_WAIT_MS = 1200
TURN_WAIT_BEFORE_WORK_MS = 400  # of that, how long to hold off before transcribing and thinking

# The app on this PC whose sound he hears, by the start of its program name. Nothing else this PC
# plays reaches him. Set to None to hear a whole output instead: PC_AUDIO_OUTPUT is part of that
# output's name ("HyperX"), or None for Windows' default output.
PC_AUDIO_APP: str | None = "Discord"
PC_AUDIO_OUTPUT: str | None = None
TTS_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
# Asking for "very high-pitched" or "shrill" tips this model into a female-sounding voice.
# This is the eighth of the options recorded in voice_samples/options/, a blend of a cold croak and
# a manic zealot leaning cold. The first voice he had is kept below it for going back.
VOICE = (
    "A cruel male goblin priest. A low, dry, cracked man's voice, hoarse and wheezing, close to a hiss. "
    "He speaks slowly, savouring each threat, with a feverish gleam and a dry cackle at the end of his lines."
)
ORIGINAL_VOICE = (
    "A small, wicked goblin. High-pitched, raspy, nasal male voice with a gravelly snarl. "
    "He speaks fast and clipped, sneering and proud, with menace and a hint of a cackle."
)


def wait_until_up(process: subprocess.Popen[bytes] | None, url: str, what: str, log: Path | None = None) -> None:
    """Wait for a service to answer. `process` is None for one this script didn't start (a container)."""
    while True:
        if process is not None and process.poll() is not None:
            raise SystemExit(f"{what} exited during startup; see {log or 'its output above'}.")
        try:
            urllib.request.urlopen(url, timeout=2)
            return
        except (urllib.error.URLError, OSError):
            time.sleep(2)


def usable_port(preferred: int) -> int:
    """The preferred port, or any free one when it can't be opened.

    Windows reserves blocks of ports at boot for Hyper-V and WSL, a different set each time
    (`netsh interface ipv4 show excludedportrange protocol=tcp`), and 8080 is sometimes in one.
    """
    for port in (preferred, 0):
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
            return int(probe.getsockname()[1])
    raise SystemExit("No free port for the language model.")


def start_stt() -> subprocess.Popen[bytes]:
    print("Loading the transcription and voiceprint models...")
    process = subprocess.Popen([str(STT_PYTHON), str(HERE / "speaker_id" / "server.py"), "--port", str(STT_PORT), "serve"])
    wait_until_up(process, f"http://127.0.0.1:{STT_PORT}/speakers", "The transcription service")
    return process


def start_head() -> subprocess.Popen[bytes]:
    return subprocess.Popen([sys.executable, str(HERE / "head.py")])


def start_llm(port: int) -> subprocess.Popen[bytes]:
    print(f"Loading the language model ({LLM_MODEL})...")
    process = subprocess.Popen(
        [
            str(LLAMA_SERVER),
            "-hf", LLM_MODEL,
            "--alias", "local-gemma",
            "--host", "127.0.0.1",
            "--port", str(port),
            "--device", LLM_DEVICE,
            "-ngl", "all",
            "-np", str(LLM_SLOTS),
            "-c", str(LLM_CONTEXT * LLM_SLOTS),
            "-fa", "on",
            "--no-mmproj",
            "--reasoning", "off",
            # Even with reasoning off, Gemma 4 sometimes opens its thinking channel and then says
            # nothing aloud; the speech server answers that with a 20s wait and a canned error.
            # Token 100 is `<|channel>` in this model's vocabulary, so forbid it outright.
            "--logit-bias", "100-inf",
        ],  # fmt: skip
        stdout=LLM_LOG.open("wb"),
        stderr=subprocess.STDOUT,
    )
    wait_until_up(process, f"http://127.0.0.1:{port}/health", "llama.cpp", LLM_LOG)
    return process


def fix_qwen_tts_rope_init() -> None:
    """Stop Qwen3-TTS's RoPE initialiser from crashing on its own audio codec.

    qwen-tts registers a "default" initialiser that reads `config.rope_theta`. transformers
    5.18 applies registered initialisers to every rotary embedding, including the Mimi
    codec's, whose config keeps that value in `rope_parameters` instead.
    """
    import torch
    from qwen_tts import _transformers_compat as compat
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    original = compat._default_rope_parameters

    def default_rope_parameters(config, device=None, seq_len=None, layer_type=None):  # type: ignore[no-untyped-def]
        if hasattr(config, "rope_theta"):
            return original(config, device, seq_len, layer_type)
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        positions = torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float)
        return 1.0 / (base ** (positions / dim)), 1.0

    # qwen-tts only registers its own when the key is free, so claiming it first is enough.
    ROPE_INIT_FUNCTIONS["default"] = default_rope_parameters


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--local-only", action="store_true", help="listen on 127.0.0.1 only (the robot can't connect)")
    parser.add_argument("--voice", default=VOICE, help="plain-English description of his voice")
    parser.add_argument("--pc-audio-app", default=PC_AUDIO_APP, help="the one app on this PC that he hears")
    parser.add_argument("--pc-audio-output", default=None, help="hear everything on this output instead (part of its name, or 'default')")
    parser.add_argument("--no-pc-audio", action="store_true", help="he hears only the robot's own microphone")
    parser.add_argument("--no-head", action="store_true", help="don't start his on-screen head")
    # In a container (see the gemma-voice-agent repo, s2s/) the transcription service and the
    # language model run in containers of their own; these point at them instead of starting them.
    parser.add_argument("--stt-url", help="use this running transcription service (http://host:port) instead of starting one")
    parser.add_argument("--llm-url", help="use this running llama.cpp server (http://host:port) instead of starting one")
    args = parser.parse_args()

    children: list[subprocess.Popen[bytes]] = []
    try:
        if args.stt_url:
            stt_url = args.stt_url.rstrip("/")
            print(f"Waiting for the transcription service at {stt_url}...")
            wait_until_up(None, f"{stt_url}/speakers", "The transcription service")
        else:
            children.append(start_stt())
            stt_url = f"http://127.0.0.1:{STT_PORT}"
        if args.llm_url:
            llm_url = args.llm_url.rstrip("/")
            print(f"Waiting for the language model at {llm_url}...")
            wait_until_up(None, f"{llm_url}/health", "llama.cpp")
        else:
            llm_port = usable_port(LLM_PORT)
            children.append(start_llm(llm_port))
            llm_url = f"http://127.0.0.1:{llm_port}"
        if not args.no_head:
            children.append(start_head())
        fix_qwen_tts_rope_init()
        gestures.install()
        memory.install()
        heard.install()
        table_log.install()
        if not args.no_pc_audio:
            if args.pc_audio_output:
                hearing = pc_audio.install(output=None if args.pc_audio_output == "default" else args.pc_audio_output)
            elif args.pc_audio_app:
                hearing = pc_audio.install(app=args.pc_audio_app)
            else:
                hearing = pc_audio.install(output=PC_AUDIO_OUTPUT)
            dm_only.install(hearing)
        from speech_to_speech.cli import main as serve

        sys.argv = [
            "speech-to-speech", "serve",
            "--host", "127.0.0.1" if args.local_only else "0.0.0.0",
            "--port", str(PORT),
            "--stt", "openai",
            "--openai_stt_base_url", f"{stt_url}/v1",
            "--llm_backend", "responses-api",
            "--model_name", "local-gemma",
            "--chat_size", str(memory.KEEP_TURNS),
            "--smart_turn_max_wait_ms", str(TURN_WAIT_MS),
            "--smart_turn_incomplete_delay_ms", str(TURN_WAIT_BEFORE_WORK_MS),
            "--responses_api_base_url", f"{llm_url}/v1",
            "--responses_api_api_key", "",
            "--tts", "qwen3",
            "--qwen3_tts_backend", "torch",
            "--qwen3_tts_model_name", TTS_MODEL,
            "--qwen3_tts_language", "English",
            "--qwen3_tts_instruct", args.voice,
        ]  # fmt: skip
        serve()
    finally:
        for child in children:
            child.terminate()


if __name__ == "__main__":
    main()
