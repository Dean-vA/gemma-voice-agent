"""Turn feeling tags in Ruxithid's replies into robot gestures, inside the speech server.

Left to call the robot's `play_emotion` tool itself, the local model gestures *instead of*
speaking: in testing it produced a gesture with no words in 20 of 24 replies when encouraged
to gesture, sometimes several calls at once and sometimes malformed. So on the local backend
the model is given no tools at all. It is prompted (profiles/ruxithid/body_local.md) to open
each reply with a feeling in angle brackets, e.g. `<angry> You dare strike the Chosen?`, and
this module strips the tags from the spoken text and sends the robot the gesture call itself.

Call `install()` before starting speech-to-speech. It patches that package's Responses API
handler, so it is tied to the speech-to-speech version in the backend environment (1.0.0).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

# What the robot's play_emotion tool accepts. Any other word, such as "still" or "silent", means no gesture.
EMOTIONS = frozenset(
    "happy excited loving grateful success thinking attentive confused uncertain sad downcast lonely "
    "angry irritated displeased disgusted scared anxious surprised amazed calming relief impatient "
    "embarrassed bored tired sleepy yes yes_understanding no no_sad no_excited no_firm welcoming "
    "greeting goodbye go_away helpful".split()
)

# The faces his on-screen head can make (head.py) travel the same way as the robot's gestures.
# They are listed by the head's model bundle, read afresh so a new model needs no restart here.
HEAD_MOODS_FILE = Path(__file__).parent / "head" / "moods.json"


def gestures() -> frozenset[str]:
    try:
        return EMOTIONS | frozenset(json.loads(HEAD_MOODS_FILE.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return EMOTIONS | frozenset("scheming snarl cackle suspicious afraid agony commune".split())

# Told the feeling each reply opens with, if set (table_log.py keeps a record of the table).
on_feeling: Callable[[str], None] | None = None

TAG = re.compile(r"<(/?)([a-z_]+)\s*/?>")
TAG_SO_FAR = re.compile(r"</?[a-z_]*\s*/?")
# The model sometimes drops the opening bracket: `<thinking>displeased>Enough muttering.`
BARE_TAG = re.compile(r"([a-z_]+)>")
BARE_TAG_SO_FAR = re.compile(r"[a-z_]*")
LONGEST_TAG = 24  # characters to wait for a closing bracket before treating the text as ordinary speech


def split_feelings(chunks: Iterable[str]) -> Iterator[tuple[str, str]]:
    """Separate streamed reply text into one ("tag", word) and the ("text", speech) around it.

    Every `<word>`, `</word>` or `<word/>` is removed from the speech, wherever it falls and
    however the stream splits it, as is a bare `word>` ahead of the first spoken word. Only
    the first tag is reported; the caller decides whether it names a gesture.
    """
    held = ""  # the tail of the stream so far, kept back because it may be an unfinished tag
    tagged = False
    spoke = False
    for chunk in chunks:
        text, held = held + chunk, ""
        speech = ""
        while text:
            if not spoke and not speech.strip():
                lead = text.lstrip()
                bare = BARE_TAG.match(lead)
                if bare:
                    if not tagged:
                        tagged = True
                        yield "tag", bare.group(1)
                    text = lead[bare.end() :]
                    continue
                if lead and len(lead) <= LONGEST_TAG and BARE_TAG_SO_FAR.fullmatch(lead):
                    held = text
                    break
            bracket = text.find("<")
            if bracket == -1:
                speech += text
                break
            speech += text[:bracket]
            text = text[bracket:]
            tag = TAG.match(text)
            if tag:
                closing, word = tag.groups()
                if not closing and not tagged:
                    tagged = True
                    yield "tag", word
                text = text[tag.end() :]
            elif len(text) <= LONGEST_TAG and TAG_SO_FAR.fullmatch(text):
                held = text
                break
            else:
                speech += "<"
                text = text[1:]
        if speech:
            spoke = spoke or bool(speech.strip())
            yield "text", speech
    # What is still held when the reply ends is either a tag cut off mid-way (not speech,
    # so dropped) or a last plain word that was being watched in case it became `word>`.
    if held and not held.lstrip().startswith("<"):
        yield "text", held


def adapt(events: Iterable[Any], *, text_type: type, call_type: type, message_type: type, gesture: Callable[[str], Any]) -> Iterator[Any]:
    """Rewrite one reply's event stream: tags out of the speech, a gesture call in.

    The reply as it goes into the conversation history is rewritten to the plain form
    `<tag> speech`. The model imitates its own earlier replies, so letting a malformed one
    stand is how the format drifts.
    """
    tag: str | None = None
    spoken: list[str] = []
    after: list[Any] = []

    def text_chunks() -> Iterator[str]:
        for event in events:
            if isinstance(event, text_type):
                yield event.text
            elif not isinstance(event, call_type):  # the model has no tools here, so its own calls are noise
                after.append(event)

    for kind, value in split_feelings(text_chunks()):
        if kind == "tag":
            tag = value
            if on_feeling is not None:
                on_feeling(value)
            if value in gestures():
                yield gesture(value)
        else:
            spoken.append(value)
            yield text_type(text=value)

    speech = " ".join("".join(spoken).split())
    for event in after:
        if isinstance(event, message_type):
            parts = [part for part in event.content if part.type == "output_text"]
            for part in parts:
                part.text = ""
            if parts:
                parts[0].text = f"<{tag}> {speech}".strip() if tag else speech
        yield event


def install() -> None:
    from openai.types.responses import ResponseFunctionToolCall
    from speech_to_speech.LLM.base_openai_compatible_language_model import AssistantMessage, TextDelta, ToolCall
    from speech_to_speech.LLM.responses_api_language_model import ResponsesApiModelHandler
    from speech_to_speech.utils.utils import _generate_id

    def gesture(feeling: str) -> Any:
        return ToolCall(
            item=ResponseFunctionToolCall(
                type="function_call",
                name="play_emotion",
                arguments=json.dumps({"emotion": feeling}),
                call_id=_generate_id("call"),
                id=_generate_id("fc"),
                status="completed",
            )
        )

    original_stream_events = ResponsesApiModelHandler._iter_stream_events
    original_optional_kwargs = ResponsesApiModelHandler._build_optional_kwargs
    original_serialize = ResponsesApiModelHandler._serialize

    def iter_stream_events(self: Any, api_response: Any) -> Iterator[Any]:
        return adapt(
            original_stream_events(self, api_response),
            text_type=TextDelta,
            call_type=ToolCall,
            message_type=AssistantMessage,
            gesture=gesture,
        )

    def build_optional_kwargs(self: Any, req_tools: Any, req_tool_choice: Any) -> dict[str, Any]:
        return original_optional_kwargs(self, None, None)

    def serialize(self: Any, active_chat: Any) -> Any:
        """Keep the gesture calls out of what the model reads back: it never made them."""

        def kind(item: Any) -> Any:
            return item.get("type") if isinstance(item, dict) else getattr(item, "type", None)

        return [item for item in original_serialize(self, active_chat) if kind(item) not in ("function_call", "function_call_output")]

    ResponsesApiModelHandler._iter_stream_events = iter_stream_events  # type: ignore[method-assign]
    ResponsesApiModelHandler._build_optional_kwargs = build_optional_kwargs  # type: ignore[method-assign]
    ResponsesApiModelHandler._serialize = serialize  # type: ignore[method-assign]
