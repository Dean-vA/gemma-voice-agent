"""Speech segmentation over per-chunk VAD probabilities.

A port of the segmentation logic in huggingface/speech-to-speech (the Reachy
Mini backend): ``VAD/vad_iterator.py`` plus the deferred speech-start rules of
``VAD/vad_handler.py``. Kept free of the model (it is fed the Silero
probability for each chunk) so it can be unit-tested with synthetic sequences.

Rules, as upstream:
  * speech triggers at ``threshold`` and counts as active down to
    ``threshold - 0.15``
  * up to ``speech_pad_ms`` of audio before the trigger is prepended
  * the segment closes once ``min_silence_ms`` has passed since the first
    below-threshold chunk ended
  * ``SpeechStarted`` is deferred until ``min_speech_ms`` of active speech has
    accumulated — or ``min_speech_continuation_ms`` when the speech continues a
    turn that can still be reopened (``reopenable`` hook)
  * a segment that closes before reaching that bar is discarded
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512   # 32 ms; the frame size Silero VAD expects at 16 kHz
_HYSTERESIS = 0.15
# Upstream floor for the continuation bar (_SHORT_SEGMENT_MIN_FRAGMENT_MS).
_MIN_FRAGMENT_MS = 100


@dataclass
class VADConfig:
    threshold: float = 0.6
    min_silence_ms: int = 64
    min_speech_ms: int = 384
    min_speech_continuation_ms: int = 192
    speech_pad_ms: int = 500


@dataclass
class SpeechStarted:
    start_sample: int        # stream offset of the segment start (incl. pre-roll)
    candidate: bool          # a reopen candidate was pending for this speech
    interrupt: bool = True   # False when only confirmed as the segment closed


@dataclass
class SpeechStopped:
    audio: np.ndarray        # float32 mono: pre-roll + speech + closing silence
    start_sample: int
    end_sample: int
    last_speech_sample: int  # where speech actually ended (for endpoint latency)


class Segmenter:
    def __init__(self, config: VADConfig | None = None) -> None:
        self.configure(config or VADConfig())
        # Set by the turn manager: can speech starting at this stream offset
        # still reopen the previous turn?
        self.reopenable: Callable[[int], bool] = lambda start_sample: False
        self.total_samples = 0
        self._pre_speech: deque[np.ndarray] = deque()
        self._pre_speech_samples = 0
        self._reset_segment()

    def configure(self, cfg: VADConfig) -> None:
        """Apply new thresholds; safe mid-stream (takes effect from the next chunk)."""
        self.cfg = cfg
        self._min_silence_samples = int(SAMPLE_RATE * cfg.min_silence_ms / 1000)
        self._pad_samples = int(SAMPLE_RATE * cfg.speech_pad_ms / 1000)
        if cfg.min_speech_continuation_ms <= 0:
            self._continuation_ms = cfg.min_speech_ms
        else:
            self._continuation_ms = min(cfg.min_speech_ms, max(_MIN_FRAGMENT_MS, cfg.min_speech_continuation_ms))

    def _reset_segment(self) -> None:
        self.triggered = False
        self._temp_end = 0
        self._prefix: list[np.ndarray] = []
        self._buffer: list[np.ndarray] = []
        self._active_samples = 0
        self._started = False
        # Tentative speech heard while the previous turn is reopenable; the
        # turn manager holds that turn's reply until this resolves.
        self.candidate = False

    @property
    def pending_reopen(self) -> bool:
        """Unconfirmed speech that may yet reopen the previous turn."""
        return self.candidate and not self._started

    def reset(self) -> None:
        """Drop any in-progress segment and the pre-roll (stream offsets keep counting)."""
        self._pre_speech.clear()
        self._pre_speech_samples = 0
        self._reset_segment()

    def skip(self, n_samples: int) -> None:
        """Advance the stream clock over audio that is deliberately not analysed."""
        self.total_samples += n_samples
        self.reset()

    def _remember_pre_speech(self, chunk: np.ndarray) -> None:
        if self._pad_samples <= 0:
            return
        self._pre_speech.append(chunk)
        self._pre_speech_samples += len(chunk)
        while self._pre_speech and self._pre_speech_samples > self._pad_samples:
            first = self._pre_speech[0]
            excess = self._pre_speech_samples - self._pad_samples
            if excess >= len(first):
                self._pre_speech.popleft()
                self._pre_speech_samples -= len(first)
            else:
                self._pre_speech[0] = first[excess:]
                self._pre_speech_samples -= excess

    def _segment_samples(self) -> int:
        return sum(len(c) for c in self._prefix) + sum(len(c) for c in self._buffer)

    def _min_active_ms(self, start_sample: int) -> float:
        if self.candidate or self.reopenable(start_sample):
            return self._continuation_ms
        return self.cfg.min_speech_ms

    def feed(self, chunk: np.ndarray, prob: float) -> list[SpeechStarted | SpeechStopped]:
        events: list[SpeechStarted | SpeechStopped] = []
        window = len(chunk)
        self.total_samples += window
        low = self.cfg.threshold - _HYSTERESIS
        closed: list[np.ndarray] | None = None
        closed_active = 0
        last_speech = 0

        if not self.triggered:
            if prob < self.cfg.threshold:
                self._remember_pre_speech(chunk)
                return events
            self.triggered = True
            self._prefix = list(self._pre_speech)
            self._pre_speech.clear()
            self._pre_speech_samples = 0
            self._buffer.append(chunk)
            self._active_samples = window
        else:
            self._buffer.append(chunk)
            if prob >= low:
                self._active_samples += window
                self._temp_end = 0
            else:
                if not self._temp_end:
                    self._temp_end = self.total_samples
                if self.total_samples - self._temp_end >= self._min_silence_samples:
                    closed = [*self._prefix, *self._buffer]
                    closed_active = self._active_samples
                    last_speech = self._temp_end - window

        if closed is None:
            # Deferred start: only once enough active speech has accumulated.
            if not self._started:
                start = self.total_samples - self._segment_samples()
                if not self.candidate and self.reopenable(start):
                    self.candidate = True
                active_ms = self._active_samples * 1000 / SAMPLE_RATE
                if active_ms >= self._min_active_ms(start):
                    self._started = True
                    events.append(SpeechStarted(start, self.candidate))
            return events

        audio = np.concatenate(closed)
        start = self.total_samples - len(audio)
        active_ms = closed_active * 1000 / SAMPLE_RATE
        if self._started or active_ms >= self._min_active_ms(start):
            if not self._started:
                events.append(SpeechStarted(start, self.candidate, interrupt=False))
            events.append(SpeechStopped(audio, start, self.total_samples, last_speech))
        self._reset_segment()
        return events
