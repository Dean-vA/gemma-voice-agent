"""Silero VAD on CPU through onnxruntime.

The ONNX model file comes from the ``silero-vad`` pip package, but we run it
with a small numpy wrapper instead of the package's own (torch-tensor) one, so
streaming VAD never touches torch or the GPU. One ``SileroVAD`` (the ONNX
session) is shared by the process; each connection gets its own
``SileroStream`` holding the recurrent state.
"""
from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path

import numpy as np

from app.vad.segmenter import CHUNK_SAMPLES, SAMPLE_RATE

_CONTEXT = 64  # samples of the previous chunk the model expects prepended at 16 kHz


def _model_path() -> Path:
    # Locate the packaged model without importing silero_vad (which imports torch).
    spec = find_spec("silero_vad")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError("Silero VAD needs the `silero-vad` package (pip install silero-vad)")
    path = Path(next(iter(spec.submodule_search_locations))) / "data" / "silero_vad.onnx"
    if not path.is_file():
        raise FileNotFoundError(f"Silero VAD model not found at {path}")
    return path


class SileroVAD:
    def __init__(self) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(_model_path()), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)

    def stream(self) -> "SileroStream":
        return SileroStream(self)


class SileroStream:
    def __init__(self, vad: SileroVAD) -> None:
        self._vad = vad
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, _CONTEXT), dtype=np.float32)

    def prob(self, chunk: np.ndarray) -> float:
        """Speech probability for one 512-sample float32 chunk."""
        if chunk.shape[0] != CHUNK_SAMPLES:
            raise ValueError(f"Silero VAD expects {CHUNK_SAMPLES}-sample chunks, got {chunk.shape[0]}")
        x = np.concatenate([self._context, chunk[None, :].astype(np.float32, copy=False)], axis=1)
        out, self._state = self._vad.session.run(
            None, {"input": x, "state": self._state, "sr": self._vad._sr}
        )
        self._context = x[:, -_CONTEXT:]
        return float(out.reshape(-1)[0])
