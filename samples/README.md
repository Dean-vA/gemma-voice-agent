# Sample audio

Drop **16 kHz mono WAV** clips here (≤ 30 s) for `scripts/benchmark.py`.

Quick way to make one from any audio/video file with ffmpeg:

```bash
ffmpeg -i input.mp3 -ac 1 -ar 16000 samples/hello.wav
```

The browser console (http://localhost:8000) records at 16 kHz automatically, so
you only need files here for the CLI benchmark.
