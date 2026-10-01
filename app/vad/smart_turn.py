"""Smart Turn v3 end-of-turn classifier (pipecat-ai/smart-turn-v3) on CPU.

Only run when Silero has just found a speech->silence boundary, so it adds no
per-chunk cost. It looks at the last 8 s of the utterance and returns the
probability that the speaker has finished their turn (vs. pausing mid-thought).
"""
from __future__ import annotations

import numpy as np

from app.vad.segmenter import SAMPLE_RATE

MODEL_REPO_ID = "pipecat-ai/smart-turn-v3"
MODEL_FILENAME = "smart-turn-v3.2-cpu.onnx"
MAX_AUDIO_SECONDS = 8


class SmartTurn:
    def __init__(self, warmup: bool = True) -> None:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from transformers import WhisperFeatureExtractor

        path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME)
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
        self._input = self.session.get_inputs()[0].name
        self._features = WhisperFeatureExtractor(chunk_length=MAX_AUDIO_SECONDS)
        if warmup:
            self.predict(np.zeros(SAMPLE_RATE, dtype=np.float32))

    def predict(self, audio: np.ndarray) -> float:
        """Probability that ``audio`` (16 kHz mono float32) is a completed turn."""
        max_samples = MAX_AUDIO_SECONDS * SAMPLE_RATE
        audio = np.asarray(audio, dtype=np.float32)
        if audio.size > max_samples:
            audio = audio[-max_samples:]
        elif audio.size < max_samples:
            audio = np.pad(audio, (max_samples - audio.size, 0))  # model expects left-padding
        feats = self._features(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="np", padding="max_length",
            max_length=max_samples, truncation=True, do_normalize=True,
        )
        out = self.session.run(None, {self._input: np.asarray(feats.input_features, dtype=np.float32)})
        return float(np.asarray(out[0]).reshape(-1)[0])
