"""A mode where Ruxithid hears the whole table but answers only the Dungeon Master.

For the stretches of a session where the players are talking among themselves and he should
speak only on the DM's cue. He still hears everyone, so when the DM does turn to him he knows
what was said; he just says nothing back to the players, and the model isn't even asked to.

A line is the DM's, and gets an answer, if any of these holds:
- it is labelled [DM]: typed from the control page, or a voice recognised as the DM's;
- it came through the DM's microphone rather than from Discord (pc_audio.py keeps count of
  which of the two was loud), which works whoever the voice is recognised as;
- neither was loud, so it was typed.
Everything else came from the players on Discord and is left unanswered.

Switched on and off through pc_audio.py's control address, by the control page or
`uv run ruxithid.py answer dm`. Call `install()` after gestures.install(): it wraps the same
handler from the outside. Tied to the speech-to-speech version in the backend environment (1.0.0).
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

import table_log

RECENT_SECONDS = 5.0  # how far back to look for who was talking when a line ends
SPOKE_SECONDS = 0.5  # the DM's mic must have been loud for this much of it for the line to count as the DM's

LABEL = re.compile(r"\s*\[([^\]]+)\]")


class Unanswered:
    """Stands in for the model's reply to a line he was not asked to answer."""

    def close(self) -> None:
        pass


def last_heard(api_input: Any) -> str | None:
    """The line a reply is being requested for, if the conversation ends on something said to him."""
    if not isinstance(api_input, list) or not api_input:
        return None
    item = api_input[-1]
    if not isinstance(item, dict) or item.get("role") != "user":
        return None
    content = item.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "input_text")
    return None


def is_the_dms(text: str, mic_seconds: float, discord_seconds: float) -> bool:
    label = LABEL.match(text)
    if label and label.group(1).strip().upper() == "DM":
        return True
    return mic_seconds >= SPOKE_SECONDS or discord_seconds < SPOKE_SECONDS


def install(pc_audio: Any) -> None:
    from speech_to_speech.LLM.responses_api_language_model import ResponsesApiModelHandler

    request = ResponsesApiModelHandler._request
    iter_stream_events = ResponsesApiModelHandler._iter_stream_events

    def request_unless_ignoring(self: Any, api_input: Any, optional_kwargs: dict[str, Any]) -> Any:
        text = last_heard(api_input) if pc_audio.dm_only else None
        if text is not None:
            mic_seconds, discord_seconds = pc_audio.who_was_loud(RECENT_SECONDS)
            if not is_the_dms(text, mic_seconds, discord_seconds):
                label = LABEL.match(text)
                who = label.group(1) if label else "an unrecognised voice"
                table_log.note("unanswered", f"DM-only: {who} (Discord loud for {discord_seconds:.1f}s, your mic for {mic_seconds:.1f}s)")
                return Unanswered()
        return request(self, api_input, optional_kwargs)

    def iter_events_unless_ignoring(self: Any, api_response: Any) -> Iterator[Any]:
        if isinstance(api_response, Unanswered):
            return iter(())  # no words and no gesture: a silent turn, and the line stays in his memory
        return iter_stream_events(self, api_response)

    ResponsesApiModelHandler._request = request_unless_ignoring  # type: ignore[method-assign]
    ResponsesApiModelHandler._iter_stream_events = iter_events_unless_ignoring  # type: ignore[method-assign]
