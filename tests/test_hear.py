"""The transcript runs alongside the reply when Gemma hears the audio itself."""
import asyncio

import numpy as np
import pytest

from app import main
from app.backends.base import TurnState
from app.config import get_settings
from app.metrics import TurnTimer


class FakeBackend:
    def __init__(self, name: str = "vllm") -> None:
        self.name = name
        self.calls: list[str] = []

    async def stream(self, system, history, samples, instruction, max_new_tokens, user_image=None, user_text=None):
        if system == main._ASR_SYSTEM:      # Gemma as the transcriber
            self.calls.append("asr")
            await asyncio.sleep(0.05)
            yield "hello robot"
            return
        self.calls.append(f"reply:{user_text}")
        for word in ("Hi ", "there."):
            await asyncio.sleep(0.01)
            yield word


class FakeParakeet:
    def __init__(self, delay: float, text: str = "hello robot", fail: bool = False) -> None:
        self.delay, self.text, self.fail = delay, text, fail
        self.cancelled = False

    async def transcribe(self, samples) -> str:
        try:
            await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        if self.fail:
            raise RuntimeError("asr down")
        return self.text


@pytest.fixture
def gateway():
    def setup(backend=None, parakeet=None):
        main.app.state.settings = get_settings()
        main.app.state.backend = backend or FakeBackend()
        main.app.state.parakeet = parakeet
        main.SESSIONS.clear()
        return main.app.state
    return setup


def _events(**kwargs):
    async def run():
        out = []
        async for event, data in main.turn_events("s", np.zeros(16000, np.float32), "", TurnTimer("fake"),
                                                  **kwargs):
            out.append((event, data))
        return out
    return asyncio.run(run())


def _order(events):
    return [e for e, _ in events]


def test_reply_starts_before_a_slow_transcript(gateway):
    gateway(parakeet=FakeParakeet(delay=0.2))
    events = _events(transcribe=True, asr_engine="parakeet")
    order = _order(events)
    assert order.index("token") < order.index("transcript") < order.index("done")
    assert dict(events)["transcript"] == {"text": "hello robot"}
    assert main.SESSIONS["s"][0].text == "hello robot"      # stored with the words


def test_fast_transcript_still_shows_first(gateway):
    gateway(parakeet=FakeParakeet(delay=0.0))
    order = _order(_events(transcribe=True, asr_engine="parakeet"))
    assert order[0] == "transcript" and order.count("transcript") == 1


def test_failed_transcript_keeps_the_reply(gateway):
    gateway(parakeet=FakeParakeet(delay=0.0, fail=True))
    events = _events(transcribe=True, asr_engine="parakeet")
    assert dict(events)["transcript"] == {"text": ""}
    assert dict(events)["done"]["reply"] == "Hi there."


def test_cascade_still_waits_for_the_words(gateway):
    backend = FakeBackend()
    gateway(backend=backend, parakeet=FakeParakeet(delay=0.05))
    order = _order(_events(llm_input="transcript"))
    assert order[0] == "transcript"
    assert backend.calls == ["reply:hello robot"]


def test_gemma_transcriber_in_process_stays_serial(gateway):
    backend = FakeBackend(name="transformers")
    gateway(backend=backend)
    order = _order(_events(transcribe=True, asr_engine="gemma"))
    assert order[0] == "transcript"
    assert backend.calls == ["asr", "reply:None"]


def test_gemma_transcriber_on_vllm_runs_alongside(gateway):
    backend = FakeBackend(name="vllm")
    gateway(backend=backend)
    order = _order(_events(transcribe=True, asr_engine="gemma"))
    assert order.index("token") < order.index("transcript")


def test_dropped_turn_cancels_its_transcript(gateway):
    asr = FakeParakeet(delay=1.0)
    gateway(parakeet=asr)
    state = TurnState()

    async def run():
        gen = main.turn_events("s", np.zeros(16000, np.float32), "", TurnTimer("fake"),
                               transcribe=True, asr_engine="parakeet", state=state, commit=False)
        assert (await gen.__anext__())[0] == "token"
        await gen.aclose()                   # barge-in / reopened turn
        await asyncio.sleep(0)
    asyncio.run(run())
    assert asr.cancelled
    assert state.user_text == ""             # not overwritten later
