"""Make Ruxithid remember only what the table actually heard when he is cut off.

His reply is written well before he finishes saying it, so when someone talks over him, or the
DM cuts him off, his memory holds sentences nobody heard. The realtime protocol has a message
for this, `conversation.item.truncate` ("the listener heard up to here"), but the speech server
accepts it and does nothing.

The head (head.py) knows which words had been played when the sound stopped, so it sends that
message with the heard words added (`x_heard_text`, and the whole reply as `x_full_text`). This
module catches it inside the speech server and rewrites his last reply in the conversation to
just those words, ending in a dash so he knows he was interrupted. If the server had already
thrown the reply away (it does when a reply is cancelled while still being written), the heard
part is put back; if nothing was heard, the reply is removed.

Call `install()` before starting speech-to-speech. It patches private parts of that package,
so it is tied to the version in the backend environment (1.0.0).
"""

from __future__ import annotations

import re
from typing import Any

import table_log

FEELING = re.compile(r"\s*(<[a-z_]+>)\s*")
CUT_OFF = " —"  # how an interrupted line ends in his memory


def _plain(text: str) -> str:
    """Words only, for comparing what he said with what his memory holds."""
    return " ".join(re.sub(r"[^\w\s']", " ", text.lower()).split())


def _text_of(item: Any) -> str:
    return "".join(getattr(part, "text", "") or "" for part in getattr(item, "content", None) or [])


def _matches(stored: str, heard: str, full: str) -> bool:
    tag = FEELING.match(stored)
    speech = _plain(stored[tag.end():] if tag else stored)
    return speech == _plain(full) or bool(heard) and speech.startswith(_plain(heard))


def remember_only_heard(chat: Any, heard: str, full: str, finished: bool = False) -> str:
    """Rewrite his last reply in `chat` to the heard words. Returns what was done, for the log.

    `finished` says whether the server had finished writing the reply, in which case it is certainly
    in the conversation: if its text doesn't match word for word, his latest reply is the one.
    """
    from speech_to_speech.LLM.chat import (
        AssistantContent,
        RealtimeConversationItemAssistantMessage,
        make_assistant_message,
    )

    heard, full = " ".join(heard.split()), " ".join(full.split())
    with chat._lock:
        # The interrupter's own words may already be in (the server adds a provisional line while
        # someone is still talking), so look back over his last few replies rather than after the last line.
        replies = [i for i, item in enumerate(chat.buffer) if isinstance(item, RealtimeConversationItemAssistantMessage)]
        candidates = list(reversed(replies[-3:]))
        matching = [i for i in candidates if _matches(_text_of(chat.buffer[i]), heard, full)]
        for index in matching or (candidates[:1] if finished else []):
            stored = _text_of(chat.buffer[index])
            tag = FEELING.match(stored)
            if not heard:
                del chat.buffer[index]
                return "removed his reply: none of it was heard"
            feeling = f"{tag.group(1)} " if tag else ""
            chat.buffer[index].content = [AssistantContent(type="output_text", text=f"{feeling}{heard}{CUT_OFF}")]
            return f"cut his reply to the {len(heard.split())} of {len(full.split())} words that were heard"
    # Not in the conversation: the server dropped it when the reply was cancelled mid-way.
    if heard:
        chat.add_item(make_assistant_message(f"<still> {heard}{CUT_OFF}"))
        return f"put back the {len(heard.split())} words that were heard of a reply the server had dropped"
    return "nothing to change"


def install() -> None:
    from speech_to_speech.api.openai_realtime.service import RealtimeService

    parse_client_event = RealtimeService.parse_client_event

    def parse_noting_what_was_heard(self: Any, raw: Any) -> Any:
        if isinstance(raw, dict) and raw.get("type") == "conversation.item.truncate" and isinstance(raw.get("x_heard_text"), str):
            raw = dict(raw)
            heard, full, finished = raw.pop("x_heard_text"), str(raw.pop("x_full_text", "")), bool(raw.pop("x_finished", False))
            for state in list(self._conns.values()):  # the server holds one conversation at a time
                try:
                    table_log.note("cut off", f"{remember_only_heard(state.runtime_config.chat, heard, full, finished)}; they heard: {heard or '(nothing)'}")
                except Exception as exc:  # never let bookkeeping break the conversation
                    print(f"heard: could not update his memory ({exc})", flush=True)
        return parse_client_event(self, raw)

    RealtimeService.parse_client_event = parse_noting_what_was_heard  # type: ignore[method-assign]
