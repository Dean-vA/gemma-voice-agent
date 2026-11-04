"""A written record of what was said at the table, for working out afterwards what went wrong.

Every line he heard (with the name the voice service gave the speaker), every line typed to him,
every reply he gave (with the feeling he opened it with and how long he took to start), replies he
chose not to give, lines left unanswered in DM-only mode, replies cancelled, and replies cut off
part-way. One file per day, logs/table-YYYY-MM-DD.txt in this folder, and the same lines in the
backend's terminal, starting `table:`.

It reads the events the speech server sends its client, so it records the robot and the on-screen
head alike. Call `install()` before starting speech-to-speech; it patches that package's WebSocket
transport, so it is tied to the version in the backend environment (1.0.0).
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any

LOG_DIR = Path(__file__).parent / "logs"

_state: dict[str, Any] = {"heard_at": None, "first_sound": None, "said": False, "feeling": None}


def note(kind: str, text: str = "") -> None:
    """Add one line to today's table log and show it in the terminal."""
    now = datetime.now()
    line = f"{now:%H:%M:%S}.{now.microsecond // 100000} {kind:<11} {text}".rstrip()
    print(f"table: {line}", flush=True)
    try:
        LOG_DIR.mkdir(exist_ok=True)
        with open(LOG_DIR / f"table-{now:%Y-%m-%d}.txt", "a", encoding="utf-8") as log:
            log.write(line + "\n")
    except OSError:
        pass  # a full disk or locked file must never stop him talking


def feeling(tag: str | None) -> None:
    """Called by gestures.py with the feeling a reply opened with (or "silent")."""
    _state["feeling"] = tag


def _text(item: Any) -> str:
    return " ".join(getattr(part, "text", None) or getattr(part, "transcript", None) or "" for part in getattr(item, "content", None) or []).strip()


def _observe(event: Any) -> None:
    kind = getattr(event, "type", None)
    now = time.monotonic()
    if kind == "conversation.item.input_audio_transcription.completed":
        note("heard", str(getattr(event, "transcript", "")).strip())
        _state["heard_at"] = now
    elif kind == "conversation.item.created":
        item = getattr(event, "item", None)
        if getattr(item, "role", None) == "user" and getattr(item, "type", None) == "message":
            text = _text(item)
            if text and not any(getattr(part, "type", None) == "input_audio" for part in getattr(item, "content", None) or []):
                note("typed", text)
                _state["heard_at"] = now
    elif kind == "response.created":
        # Not the feeling: the model may already have written this reply, and its feeling, before the server announces it.
        _state.update(first_sound=None, said=False)
    elif kind == "response.output_audio.delta":
        if _state["first_sound"] is None:
            _state["first_sound"] = now
    elif kind == "response.output_audio_transcript.done":
        _state["said"] = True
        heard_at, first = _state["heard_at"], _state["first_sound"]
        delay = f"  [{first - heard_at:.1f}s after the line]" if heard_at and first and 0 <= first - heard_at < 60 else ""
        tag = f"<{_state['feeling']}> " if _state["feeling"] else ""
        note("said", f"{tag}{str(getattr(event, 'transcript', '')).strip()}{delay}")
    elif kind == "response.done":
        status = getattr(getattr(event, "response", None), "status", None)
        if status == "cancelled":
            note("cancelled", "a reply was dropped before it was finished")
        elif status == "completed" and not _state["said"]:
            reason = "he chose silence" if _state["feeling"] in (None, "silent") else f"opened with <{_state['feeling']}> but said nothing"
            note("no reply", reason)
        _state["feeling"] = None


def install() -> None:
    from speech_to_speech.api.openai_realtime.transports import WebSocketTransport

    import gestures

    gestures.on_feeling = feeling

    send_events = WebSocketTransport.send_events

    async def send_events_noting(self: Any, events: list[Any]) -> None:
        for event in events:
            try:
                _observe(event)
            except Exception:  # the record is a convenience; never let it break a reply
                pass
        await send_events(self, events)

    WebSocketTransport.send_events = send_events_noting  # type: ignore[method-assign]
    note("started", "backend.py is up")
