"""Give Ruxithid a longer, better memory of a long scene, inside the speech server.

Left alone, the speech server keeps 30 exchanges, then has the model squash everything but the
latest one into two paragraphs of at most five sentences, following instructions written for a
general voice assistant ("The user asked about..."). At the table that throws away who said
what, what was promised and what he was told, twenty-nine exchanges at a time.

With this installed:
- he holds KEEP_TURNS exchanges before anything is summarised,
- the newest KEEP_RECENT of them always stay word for word; only what is older is summarised,
- the summary is written as the memory of a character in a D&D scene: who did what, by name,
  with secrets, bargains, threats and wounds kept exactly,
- and it is put back in the form he hears everything else in, a [DM] note and a line of his own,
  so it doesn't pull his later replies out of shape.

Each summary also reads the one before it, so the memory rolls forward for as long as the scene runs.

Call `install()` before starting speech-to-speech, and pass it `--chat_size KEEP_TURNS`. It
patches private parts of that package, so it is tied to the version in the backend
environment (1.0.0).
"""

from __future__ import annotations

from typing import Any

KEEP_TURNS = 60  # exchanges he holds before the oldest are summarised
KEEP_RECENT = 30  # how many of the newest stay word for word when that happens

SUMMARY_INSTRUCTIONS = """\
You keep the memory of Ruxithid the Chosen, a goblin boss played as a character in a live
Dungeons & Dragons scene. Below is the older part of the scene.

Lines marked "User" are what he heard. A name in square brackets at the start says who spoke:
[DM] is the Dungeon Master describing what happens, any other name is one of the adventurers
facing him. Lines marked "Assistant" are what Ruxithid said; a word in angle brackets at the
start is only his expression. The scene may open with an earlier memory written like this one.
Carry forward everything in it that still matters.

Write what he must remember to carry on without contradicting himself. Output a single JSON
object with exactly two string fields:
  "user_summary"      - what was said and done to him, written to him as "you". Name who did
                        each thing. Keep the facts and secrets he was told, offers and bargains,
                        threats, lies he caught, insults that stung, attacks and wounds, and
                        anything the DM described that changed the situation.
  "assistant_summary" - what he himself said and did, in his own first person ("I"). Keep what
                        he revealed and what he refused to reveal, what he promised, demanded or
                        agreed to, and how he now feels about each adventurer.

Rules:
- Keep names, numbers, passwords, places and objects exactly, word for word.
- Leave out repeated taunts and anything that changes nothing.
- At most twelve plain sentences in each field, in the past tense.
- Emit only the JSON object: no markdown, no code fences, no extra keys.\
"""

HEARD_PREFIX = "[DM] Earlier in this scene: "
SAID_PREFIX = "<still> "  # a feeling tag, like every reply of his, so the format he imitates stays intact


def install() -> None:
    from speech_to_speech.LLM import base_openai_compatible_language_model as handlers
    from speech_to_speech.LLM import compaction_prompt
    from speech_to_speech.LLM.chat import Chat, CompactionResult

    # Summarise only what is older than the newest KEEP_RECENT exchanges. The original always
    # stops one exchange short of the end, so tell it the conversation is that much shorter.
    snapshot_for_compaction = Chat._snapshot_for_compaction

    def snapshot_older_turns(self: Any) -> Any:
        real = self._user_turn_count
        self._user_turn_count = max(0, real - (KEEP_RECENT - 1))
        try:
            return snapshot_for_compaction(self)
        finally:
            self._user_turn_count = real

    Chat._snapshot_for_compaction = snapshot_older_turns  # type: ignore[method-assign]

    compaction_prompt.COMPACTION_SYSTEM_PROMPT = SUMMARY_INSTRUCTIONS

    # The gesture calls (gestures.py) are not his words; keep them out of what gets summarised.
    render_transcript = compaction_prompt._render_transcript

    def render_without_gestures(snapshot: list[Any]) -> str:
        spoken = [item for item in snapshot if not (isinstance(item, dict) and item.get("type") in ("function_call", "function_call_output"))]
        return render_transcript(spoken)

    compaction_prompt._render_transcript = render_without_gestures

    build_compactor = compaction_prompt.build_compactor

    def build_scene_compactor(generate_fn: Any) -> Any:
        compact = build_compactor(generate_fn)

        def compact_scene(snapshot: list[Any]) -> Any:
            result = compact(snapshot)
            heard = result.user_summary.removeprefix(HEARD_PREFIX).removeprefix("[DM] ")
            said = result.assistant_summary.removeprefix(SAID_PREFIX)
            print(f"memory: summarised the older part of the scene.\n  heard: {heard}\n  said:  {said}", flush=True)
            return CompactionResult(user_summary=HEARD_PREFIX + heard, assistant_summary=SAID_PREFIX + said)

        return compact_scene

    handlers.build_compactor = build_scene_compactor  # type: ignore[assignment]
