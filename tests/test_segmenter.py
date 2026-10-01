"""Segmenter state machine, driven by synthetic VAD probabilities (no model).

Run:  pip install pytest numpy pydantic-settings && pytest tests
"""
from __future__ import annotations

import numpy as np

from app.vad.segmenter import CHUNK_SAMPLES, Segmenter, SpeechStarted, SpeechStopped

PAD = 8000  # 500 ms pre-roll in samples


def run(seg: Segmenter, probs: list[float]):
    """Feed one chunk per probability; each chunk is filled with its stream index."""
    events = []
    for p in probs:
        idx = seg.total_samples // CHUNK_SAMPLES
        events += seg.feed(np.full(CHUNK_SAMPLES, idx, dtype=np.float32), p)
    return events


def test_silence_emits_nothing():
    assert run(Segmenter(), [0.05] * 100) == []


def test_start_is_deferred_until_min_speech():
    seg = Segmenter()  # 384 ms -> 12 chunks
    assert run(seg, [0.9] * 11) == []
    (ev,) = run(seg, [0.9])
    assert isinstance(ev, SpeechStarted) and not ev.candidate and ev.interrupt


def test_short_segment_is_discarded():
    seg = Segmenter()
    assert run(seg, [0.9] * 8 + [0.0] * 10) == []
    assert not seg.triggered


def test_closes_after_min_silence_with_preroll():
    seg = Segmenter()  # min silence 64 ms, counted from the end of the first quiet chunk
    events = run(seg, [0.0] * 30 + [0.9] * 20 + [0.0] * 3)
    started, stopped = events
    assert isinstance(started, SpeechStarted) and isinstance(stopped, SpeechStopped)
    assert stopped.start_sample == 30 * CHUNK_SAMPLES - PAD == started.start_sample
    assert stopped.end_sample == 53 * CHUNK_SAMPLES
    assert len(stopped.audio) == PAD + 23 * CHUNK_SAMPLES
    assert stopped.audio[-1] == 52
    assert stopped.last_speech_sample == 50 * CHUNK_SAMPLES


def test_two_silent_chunks_do_not_close():
    seg = Segmenter()
    events = run(seg, [0.9] * 15 + [0.0] * 2 + [0.9] * 5)
    assert [type(e) for e in events] == [SpeechStarted]
    assert seg.triggered


def test_hysteresis_keeps_segment_active_below_trigger():
    seg = Segmenter()  # trigger 0.6, active down to 0.45
    events = run(seg, [0.9] * 3 + [0.5] * 20)
    assert [type(e) for e in events] == [SpeechStarted]
    assert run(Segmenter(), [0.5] * 40) == []  # never reaches the trigger


def test_reopenable_turn_uses_continuation_bar():
    seg = Segmenter()
    seg.reopenable = lambda start: True
    assert run(seg, [0.9] * 5) == []
    assert seg.pending_reopen
    (ev,) = run(seg, [0.9])  # 192 ms -> 6 chunks
    assert isinstance(ev, SpeechStarted) and ev.candidate
    assert not seg.pending_reopen


def test_candidate_is_dropped_when_speech_does_not_confirm():
    seg = Segmenter()
    seg.reopenable = lambda start: True
    assert run(seg, [0.9] * 3 + [0.0] * 3) == []
    assert not seg.pending_reopen and not seg.triggered


def test_candidate_survives_turn_becoming_unreopenable():
    """Once tentative speech is a reopen candidate it keeps the lower bar."""
    seg = Segmenter()
    seg.reopenable = lambda start: True
    run(seg, [0.9] * 2)
    seg.reopenable = lambda start: False
    events = run(seg, [0.9] * 4)
    assert [type(e) for e in events] == [SpeechStarted] and events[0].candidate


def test_next_segment_preroll_starts_after_previous_close():
    seg = Segmenter()
    first = run(seg, [0.9] * 15 + [0.0] * 3)[-1]
    second = run(seg, [0.0] * 4 + [0.9] * 15 + [0.0] * 3)[-1]
    assert second.start_sample == first.end_sample  # gap shorter than the pre-roll is kept whole
    merged = np.concatenate([first.audio, second.audio])
    assert np.array_equal(merged[::CHUNK_SAMPLES], np.arange(len(merged) // CHUNK_SAMPLES))


def test_skip_advances_clock_and_drops_state():
    seg = Segmenter()
    run(seg, [0.9] * 5)
    seg.skip(10 * CHUNK_SAMPLES)
    assert not seg.triggered and seg.total_samples == 15 * CHUNK_SAMPLES
    stopped = run(seg, [0.9] * 15 + [0.0] * 3)[-1]
    assert stopped.start_sample == 15 * CHUNK_SAMPLES  # nothing from before the skip
