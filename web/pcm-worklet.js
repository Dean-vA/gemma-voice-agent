// Mic capture off the main thread: batches the 128-sample render quanta into
// ~32 ms mono Float32 blocks and posts them to the page.
class PcmCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.block = new Float32Array(Math.round(sampleRate * 0.032));
    this.fill = 0;
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    let off = 0;
    while (off < ch.length) {
      const n = Math.min(ch.length - off, this.block.length - this.fill);
      this.block.set(ch.subarray(off, off + n), this.fill);
      this.fill += n; off += n;
      if (this.fill === this.block.length) {
        this.port.postMessage(this.block.slice(0));
        this.fill = 0;
      }
    }
    return true;
  }
}
registerProcessor("pcm-capture", PcmCapture);
