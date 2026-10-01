// Gemma 4 E4B voice console — robot simulator.
// Captures mic audio at 16 kHz mono, draws a live waveform, and reports per-turn
// latency against a robot-conversation budget. Two input modes:
//   push-to-talk  one WAV per turn over HTTP, reply streamed back as SSE
//   continuous    mic streamed over /ws/converse; the server does VAD, turn
//                 detection and barge-in and we just react to its events

const TARGET_SR = 16000;
const STREAM_KEEP_S = 60;            // sent mic audio kept for "replay my speech"
const GAUGE_MAX_MS = 1500;          // gauge full-scale; zones at 300 / 800

let sessionId = null;
let audioCtx = null, stream = null, sourceNode = null, captureNode = null, analyser = null;
let recording = false;
let captured = [];
let captureSampleRate = TARGET_SR;

// Continuous mode: the open /ws/converse socket, plus the PCM we've sent
// (int16 chunks) so a turn's audio can be cut out by the server's offsets.
let ws = null;
let sentChunks = [], sentBase = 0, sentTotal = 0;   // sentBase = stream offset of sentChunks[0]
const sess = { turns: 0, ttftSum: 0, tpsSum: 0 };

// Webcam image input: when "Vision" is on, one frame is grabbed per spoken turn.
let camStream = null;        // active MediaStream while Vision is on
const CAM_MAX_DIM = 768;     // downscale longest side; Gemma sees ~768px anyway

const $ = (id) => document.getElementById(id);

// ---------- health ----------
async function refreshHealth() {
  try {
    const h = await (await fetch("/health")).json();
    setBadge("badge-backend", "backend", h.backend);
    setBadge("badge-quant", "quant", h.quant_mode);
    const gpu = h.gpu || {};
    setBadge("badge-gpu", "gpu", gpu.cuda ? shortGpu(gpu.device_name) : "cpu");
    const tts = h.tts || {};
    setBadge("badge-tts", "tts", tts.reachable ? (tts.engine || "on") : "off");
    $("status-dot").className = "dot ok";
    refreshEngines();
    refreshAsrEngines();
  } catch {
    $("status-dot").className = "dot err";
  }
}

// Populate the TTS engine dropdown from the reachable engines.
async function refreshEngines() {
  try {
    const data = await (await fetch("/tts/engines")).json();
    const sel = $("tts-select");
    const prev = sel.value;
    const usable = data.engines.filter((e) => e.reachable);
    sel.innerHTML = "";
    if (!usable.length) {
      sel.innerHTML = '<option value="">— no voices —</option>';
      return false;
    }
    for (const e of usable) {
      const o = document.createElement("option");
      o.value = e.name;
      o.textContent = `🔉 ${e.name}`;
      sel.appendChild(o);
    }
    sel.value = usable.some((e) => e.name === prev) ? prev
      : (usable.some((e) => e.name === data.default) ? data.default : usable[0].name);
    return true;
  } catch { /* leave as-is */ return false; }
}
// Offer the transcription engines the gateway has loaded (gemma is always there).
async function refreshAsrEngines() {
  try {
    const data = await (await fetch("/asr/engines")).json();
    const sel = $("asr-select"), prev = sel.dataset.touched ? sel.value : data.default;
    sel.innerHTML = "";
    for (const e of data.engines.filter((e) => e.available)) {
      const o = document.createElement("option");
      o.value = e.name; o.textContent = `📝 ${e.name}`;
      sel.appendChild(o);
    }
    sel.value = [...sel.options].some((o) => o.value === prev) ? prev : "gemma";
    const input = $("input-select");
    if (!input.dataset.touched) input.value = data.llm_input || "audio";
  } catch { /* leave as-is */ }
}
function setBadge(id, k, v) { $(id).innerHTML = `${k} <b>${v}</b>`; }
function shortGpu(name) { return (name || "").replace("NVIDIA GeForce ", "").replace("NVIDIA ", "") || "gpu"; }

// ---------- waveform ----------
const waveCanvas = $("wave");
const wctx = waveCanvas.getContext("2d");
function sizeCanvas() {
  const r = waveCanvas.getBoundingClientRect();
  waveCanvas.width = Math.max(2, r.width) * devicePixelRatio;
  waveCanvas.height = 56 * devicePixelRatio;
}
addEventListener("resize", sizeCanvas);

function drawWave() {
  requestAnimationFrame(drawWave);
  const w = waveCanvas.width, h = waveCanvas.height;
  wctx.clearRect(0, 0, w, h);
  const mid = h / 2;
  const live = (recording || ws) && analyser;
  let data;
  if (live) { data = new Uint8Array(analyser.fftSize); analyser.getByteTimeDomainData(data); }

  wctx.lineWidth = 2 * devicePixelRatio;
  wctx.strokeStyle = live
    ? "oklch(0.73 0.175 42)"      // signal coral when transmitting
    : "oklch(0.40 0.02 274)";     // dim idle line
  wctx.beginPath();
  const n = live ? data.length : 120;
  for (let i = 0; i < n; i++) {
    const x = (i / (n - 1)) * w;
    const v = live ? (data[i] / 128 - 1) : 0;
    const y = mid + v * mid * 0.9;
    i ? wctx.lineTo(x, y) : wctx.moveTo(x, y);
  }
  wctx.stroke();
}

// ---------- capture ----------
async function ensureMic() {
  if (audioCtx) return;
  stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
  });
  // Ask for a 16 kHz context so the browser does the (properly filtered)
  // resampling; fall back to the device rate + resampleTo16k where a context
  // can't differ from the mic's rate.
  const AC = window.AudioContext || window.webkitAudioContext;
  try {
    audioCtx = new AC({ sampleRate: TARGET_SR });
    sourceNode = audioCtx.createMediaStreamSource(stream);
  } catch {
    if (audioCtx) audioCtx.close();
    audioCtx = new AC();
    sourceNode = audioCtx.createMediaStreamSource(stream);
  }
  captureSampleRate = audioCtx.sampleRate;
  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 1024;
  sourceNode.connect(analyser);
  await audioCtx.audioWorklet.addModule("/web/pcm-worklet.js");
  captureNode = new AudioWorkletNode(audioCtx, "pcm-capture");
  captureNode.port.onmessage = (e) => onAudio(e.data);
  sourceNode.connect(captureNode);
}

function onAudio(frame) {
  if (ws) { if (ws.readyState === WebSocket.OPEN) streamFrame(frame); return; }
  if (recording) captured.push(frame);
}

async function startRecording() {
  await ensureMic();
  if (audioCtx.state === "suspended") await audioCtx.resume();
  captured = []; recording = true;
  $("btn-talk").classList.add("recording");
  setMic("listening", true);
}

async function stopRecording() {
  if (!recording) return;
  recording = false;
  $("btn-talk").classList.remove("recording");
  setMic("processing", true);

  const total = captured.reduce((n, c) => n + c.length, 0);
  if (total < captureSampleRate * 0.2) { setMic("idle", false); return; }
  const down = resampleTo16k(mergeChunks(captured, total), captureSampleRate);
  if (evalMode) await evalSubmit(encodeWav(down, TARGET_SR));
  else { await sendAudio(encodeWav(down, TARGET_SR)); setMic("idle", false); }
}

function setMic(text, live) { const el = $("mic-state"); el.textContent = text; el.classList.toggle("live", !!live); }

function mergeChunks(chunks, total) { const out = new Float32Array(total); let o = 0; for (const c of chunks) { out.set(c, o); o += c.length; } return out; }
function resampleTo16k(buffer, srcRate) {
  if (srcRate === TARGET_SR) return buffer;
  const ratio = srcRate / TARGET_SR, newLen = Math.round(buffer.length / ratio), out = new Float32Array(newLen);
  for (let i = 0; i < newLen; i++) {
    const idx = i * ratio, lo = Math.floor(idx), hi = Math.min(lo + 1, buffer.length - 1), frac = idx - lo;
    out[i] = buffer[lo] * (1 - frac) + buffer[hi] * frac;
  }
  return out;
}
function floatToInt16(samples) {
  const out = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; i++) { const s = Math.max(-1, Math.min(1, samples[i])); out[i] = s < 0 ? s * 0x8000 : s * 0x7fff; }
  return out;
}
function encodeWav(samples, sampleRate) { return encodeWavInt16(floatToInt16(samples), sampleRate); }
function encodeWavInt16(samples, sampleRate) {
  const buf = new ArrayBuffer(44 + samples.length * 2), view = new DataView(buf);
  const w = (off, s) => { for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i)); };
  w(0, "RIFF"); view.setUint32(4, 36 + samples.length * 2, true); w(8, "WAVE"); w(12, "fmt ");
  view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true); view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true); view.setUint16(34, 16, true); w(36, "data");
  view.setUint32(40, samples.length * 2, true);
  for (let i = 0, off = 44; i < samples.length; i++, off += 2) view.setInt16(off, samples[i], true);
  return new Blob([view], { type: "audio/wav" });
}

// ---------- audio playback (sequential live queue + on-demand replay) ----------
// All audio lives in JS memory only, so it's gone on page refresh.
const playQueue = [];
let playing = false;
let curPlayer = null;          // the Audio element currently speaking
const idleState = () => (ws ? ["listening", true] : ["idle", false]);
function setPlaying(on) {
  if (playing === on) return;
  playing = on;
  wsSend({ type: "playback", active: on });   // lets the server tell barge-in from a new turn
}
function enqueueAudio(b64) {
  const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  const blob = new Blob([bytes], { type: "audio/wav" });
  playQueue.push(URL.createObjectURL(blob));
  if (!playing) playNext();
  return blob;
}
function playNext() {
  if (!playQueue.length) { curPlayer = null; setPlaying(false); setMic(...idleState()); return; }
  setPlaying(true);
  setMic("speaking", true);
  const url = playQueue.shift();
  const a = curPlayer = new Audio(url);
  a.onended = a.onerror = () => { if (curPlayer !== a) return; URL.revokeObjectURL(url); playNext(); };
  a.play().catch(() => { if (curPlayer === a) playNext(); });
}
// Stop speaking now and drop anything queued (barge-in).
function flushPlayback() {
  for (const url of playQueue.splice(0)) URL.revokeObjectURL(url);
  if (curPlayer) { const a = curPlayer; curPlayer = null; a.pause(); URL.revokeObjectURL(a.src); }
  setPlaying(false);
}
function playBlobs(blobs) {
  let i = 0;
  const next = () => {
    if (i >= blobs.length) return;
    const url = URL.createObjectURL(blobs[i++]);
    const a = new Audio(url);
    a.onended = a.onerror = () => { URL.revokeObjectURL(url); next(); };
    a.play().catch(next);
  };
  next();
}
function attachReplay(el, blobs) {
  if (!blobs || !blobs.length) return;
  const b = document.createElement("button");
  b.className = "replay"; b.textContent = "▶"; b.title = "Replay audio";
  b.onclick = () => playBlobs(blobs);
  el.querySelector(".who").appendChild(b);
}

// ---------- send + stream ----------
let curUser = null, curAssistant = null, curAudio = [];
async function sendAudio(wavBlob) {
  $("empty")?.remove();
  // Vision on -> grab one webcam frame for this turn.
  const sentImage = $("toggle-vision").checked ? await captureFrame() : null;
  curUser = addMessage("user", "🎤 spoken audio");
  attachReplay(curUser, [wavBlob]);              // replay your own speech
  if (sentImage) addImageToMsg(curUser, sentImage);
  curAssistant = addMessage("assistant", "");
  curAudio = [];

  const form = new FormData();
  form.append("audio", wavBlob, "turn.wav");
  if (sessionId) form.append("session_id", sessionId);
  form.append("instruction", $("instruction").value || "");
  form.append("transcribe", $("toggle-transcribe").checked ? "true" : "false");
  form.append("asr_engine", $("asr-select").value || "");
  form.append("llm_input", $("input-select").value || "");
  form.append("engine", $("tts-select").value || "");
  if (sentImage) form.append("image", sentImage, "frame.jpg");

  const endpoint = $("toggle-speak").checked ? "/converse" : "/chat/stream";
  const resp = await fetch(endpoint, { method: "POST", body: form });
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n"); buffer = events.pop();
    for (const block of events) handleEvent(block);
  }
  curAssistant.classList.add("done");
  attachReplay(curAssistant, curAudio);          // replay robot speech
  if (!playing && !playQueue.length) setMic("idle", false);
}

function handleEvent(block) {
  let ev = "message", data = "";
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) ev = line.slice(6).trim();
    else if (line.startsWith("data:")) data += line.slice(5).trim();
  }
  if (!data) return;
  dispatch(ev, JSON.parse(data));
}

// Turn events, shared by the SSE (push-to-talk) and WebSocket (continuous) paths.
function dispatch(ev, p) {
  if (ev === "session") { sessionId = p.session_id; $("session-id").textContent = sessionId.slice(0, 12); }
  else if (ev === "transcript") { curUser.querySelector(".body").textContent = p.text || "(no speech detected)"; scrollDown(); }
  else if (ev === "token") { curAssistant.querySelector(".body").textContent += p.text; scrollDown(); }
  else if (ev === "audio") { curAudio.push(enqueueAudio(p.wav_base64)); }
  else if (ev === "done") updateMetrics(p.metrics);
}

// ---------- continuous mode (server-side VAD over /ws/converse) ----------
function wsSend(obj) { if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj)); }
function sendConfig() {
  wsSend({
    type: "config", session_id: sessionId,
    instruction: $("instruction").value || "",
    transcribe: $("toggle-transcribe").checked,
    asr_engine: $("asr-select").value || "",
    llm_input: $("input-select").value || "",
    respond: !evalMode,                 // ASR eval only wants the turn boundaries
    speak: $("toggle-speak").checked,
    engine: $("tts-select").value || "",
  });
}

function streamFrame(frame) {
  const pcm = floatToInt16(resampleTo16k(frame, captureSampleRate));
  ws.send(pcm.buffer);
  sentChunks.push(pcm); sentTotal += pcm.length;
  while (sentChunks.length && sentTotal - sentBase - sentChunks[0].length > STREAM_KEEP_S * TARGET_SR) {
    sentBase += sentChunks.shift().length;
  }
}
// WAV of what we sent between two stream offsets (ms), for the replay button.
function sentSlice(startMs, endMs) {
  const a = Math.max(Math.round(startMs * TARGET_SR / 1000), sentBase);
  const b = Math.min(Math.round(endMs * TARGET_SR / 1000), sentTotal);
  if (b <= a) return null;
  const out = new Int16Array(b - a);
  let pos = sentBase;
  for (const c of sentChunks) {
    const lo = Math.max(a, pos), hi = Math.min(b, pos + c.length);
    if (hi > lo) out.set(c.subarray(lo - pos, hi - pos), lo - a);
    pos += c.length;
  }
  return encodeWavInt16(out, TARGET_SR);
}

let pendingImage = null;      // webcam frame sent for the turn being spoken
async function startContinuous() {
  await ensureMic();
  if (audioCtx.state === "suspended") await audioCtx.resume();
  sentChunks = []; sentBase = sentTotal = 0;
  const sock = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/converse`);
  sock.binaryType = "arraybuffer";
  sock.onopen = () => { ws = sock; sendConfig(); setMic("listening", true); };
  // Handlers await (webcam grab), so run events strictly in arrival order.
  let chain = Promise.resolve(), lastError = "";
  sock.onmessage = (e) => {
    const p = JSON.parse(e.data);
    if (p.type === "error") lastError = p.message;
    chain = chain.then(() => onServerEvent(p)).catch((err) => console.error("[ws]", err));
  };
  sock.onclose = () => {
    if (ws === sock) ws = null;
    if (vad.checked) { vad.checked = false; vad.dispatchEvent(new Event("change")); }
    if (lastError) setMic(lastError, false);
  };
}
function stopContinuous() {
  const sock = ws; ws = null;
  if (sock) sock.close();
  flushPlayback();
  setMic("idle", false);
}

function finishAssistant() {
  if (!curAssistant) return;
  curAssistant.classList.add("done");
  attachReplay(curAssistant, curAudio);
}

async function onServerEvent(p) {
  const ev = p.type;
  if (evalMode && ev !== "session" && ev !== "error") {
    // Hands-free reading: the server only tells us where each sentence
    // started and ended; cut that out of what we streamed and score it.
    if (ev === "speech_started") setMic("hearing you", true);
    else if (ev === "speech_stopped") {
      const wav = sentSlice(p.audio_start_ms, p.audio_end_ms);
      if (wav) evalCapture(wav);
    }
    return;
  }
  if (ev === "speech_started") {
    flushPlayback();
    setMic("hearing you", true);
    sendConfig();                                 // pick up toggle/instruction edits
    pendingImage = $("toggle-vision").checked ? await captureFrame() : null;
    if (pendingImage) wsSend({ type: "image", data: await blobToBase64(pendingImage) });
  } else if (ev === "speech_stopped") {
    $("empty")?.remove();
    curUser = addMessage("user", "🎤 spoken audio");
    const wav = sentSlice(p.audio_start_ms, p.audio_end_ms);
    if (wav) attachReplay(curUser, [wav]);
    if (pendingImage) addImageToMsg(curUser, pendingImage);
    curAssistant = addMessage("assistant", "");
    curAudio = [];
    setMic("processing", true);
  } else if (ev === "cancelled") {
    flushPlayback();
    if (curAssistant && !curAssistant.classList.contains("done")) {
      curAssistant.classList.add("interrupted");
      finishAssistant();
    }
    curUser = curAssistant = null;
  } else if (ev === "error") {
    console.error("[ws]", p.message);
    setMic(p.message, false);
  } else if (ev === "session" || curAssistant) {
    dispatch(ev, p);
    if (ev === "done") { finishAssistant(); if (!playing && !playQueue.length) setMic(...idleState()); }
  }
}

function blobToBase64(blob) {
  return new Promise((resolve) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result.slice(r.result.indexOf(",") + 1));
    r.readAsDataURL(blob);
  });
}

// ---------- ASR eval mode ----------
// Read each test case in ASR_SCRIPT (asr-script.js) aloud with the talk button.
// The clip is saved with its reference text (samples/asr/ on the gateway, for
// scripts/asr_eval.py) and transcribed by every loaded engine, so their word
// error rate and latency can be compared side by side.
let evalMode = false, evalIndex = 0, evalEngines = ["gemma"];
const evalResults = {};   // id -> { blob, runs: { engine: {text, ms, errors, ref_words, wer} | {error} } }

// The conversation panel has three modes: "chat", "asr" (read-aloud ASR eval)
// and "latency" (replay clips, time the first spoken audio).
let mode = "chat";
async function setMode(name) {
  mode = name;
  const on = evalMode = name === "asr";
  $("mode-chat").classList.toggle("active", name === "chat");
  $("mode-eval").classList.toggle("active", on);
  $("mode-latency").classList.toggle("active", name === "latency");
  $("transcript").hidden = name !== "chat";
  $("eval-view").hidden = !on;
  $("lat-view").hidden = name !== "latency";
  // The latency eval replays saved clips; the mic has no part in it.
  if (name === "latency" && vad.checked) { vad.checked = false; vad.dispatchEvent(new Event("change")); }
  talkBtn.disabled = name === "latency" || vad.checked;
  if (ws) { flushPlayback(); sendConfig(); setMic("listening", true); }   // Continuous stays on; replies switch off/on
  else talkBtn.innerHTML = on ? "🎙 Hold to read <kbd>space</kbd>" : "🎙 Hold to talk <kbd>space</kbd>";
  if (name === "latency") latRender();
  if (on) {
    try {
      const data = await (await fetch("/asr/engines")).json();
      evalEngines = data.engines.filter((e) => e.available).map((e) => e.name);
    } catch { /* keep the last known list */ }
    evalRender();
  }
}

// Hands-free: take the clip for the sentence on screen, move straight on to the
// next one, and score clips one after another in the background.
let evalQueue = Promise.resolve();
function evalCapture(blob) {
  const item = ASR_SCRIPT[evalIndex];
  if (evalIndex < ASR_SCRIPT.length - 1) evalGo(evalIndex + 1);
  evalQueue = evalQueue.then(() => evalSubmit(blob, item)).catch((e) => console.error("[eval]", e));
}

async function evalSubmit(blob, item = ASR_SCRIPT[evalIndex]) {
  const entry = evalResults[item.id] = { blob, runs: {} };
  evalRender();
  setMic("saving clip", true);
  const form = new FormData();
  form.append("id", item.id); form.append("text", item.text); form.append("audio", blob, `${item.id}.wav`);
  const saved = await fetch("/eval/asr/clips", { method: "POST", body: form }).catch(() => null);
  if (!saved || !saved.ok) { setMic("could not save clip", false); return; }
  // One engine at a time, so they don't compete for the GPU while being timed.
  for (const engine of evalEngines) {
    setMic(`transcribing · ${engine}`, true);
    const f = new FormData();
    f.append("audio", blob, "clip.wav"); f.append("engine", engine); f.append("reference", item.text);
    try {
      const r = await fetch("/transcribe", { method: "POST", body: f });
      entry.runs[engine] = r.ok ? await r.json() : { error: `HTTP ${r.status}` };
    } catch (e) { entry.runs[engine] = { error: String(e.message || e) }; }
    evalRender();
  }
  ws ? setMic("listening", true) : setMic("saved", false);
}

const esc = (s) => s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
const evalWords = (s) => s.toLowerCase().replace(/[^a-z0-9' ]+/g, " ").split(/\s+/).filter(Boolean);
// Highlight hypothesis words that aren't in the reference. A visual hint only:
// the WER figure comes from the gateway's proper alignment.
function evalMarkDiff(hyp, ref) {
  const known = new Set(evalWords(ref));
  return hyp.split(/(\s+)/).map((tok) => {
    const w = evalWords(tok)[0];
    return w && !known.has(w) && !/^\d/.test(w) ? `<b>${esc(tok)}</b>` : esc(tok);
  }).join("");
}

function evalRender() {
  const item = ASR_SCRIPT[evalIndex];
  const done = ASR_SCRIPT.filter((s) => evalResults[s.id]).length;
  $("eval-prompt").textContent = item.text;
  $("eval-meta").textContent = `${evalIndex + 1} / ${ASR_SCRIPT.length} · ${item.group} · ${item.id} · ${done} recorded`;
  $("eval-play").disabled = !evalResults[item.id];
  $("eval-prev").disabled = evalIndex === 0;
  $("eval-next").disabled = evalIndex === ASR_SCRIPT.length - 1;

  $("eval-rows").innerHTML = ASR_SCRIPT.map((s, i) => {
    const r = evalResults[s.id];
    const cls = [i === evalIndex ? "current" : "", r ? "" : "todo"].join(" ").trim();
    const lines = [`<div class="ref">${esc(s.text)}</div>`], wer = ["&nbsp;"], lat = ["&nbsp;"];
    if (r) for (const e of evalEngines) {
      const run = r.runs[e], tag = `<span class="eval-engine">${e}</span>`;
      if (!run) { lines.push(`<div class="hyp">${tag} …</div>`); wer.push("…"); lat.push("…"); }
      else if (run.error) { lines.push(`<div class="hyp">${tag} <b>${esc(run.error)}</b></div>`); wer.push("—"); lat.push("—"); }
      else {
        lines.push(`<div class="hyp">${tag} ${evalMarkDiff(run.text, s.text) || "<b>(empty)</b>"}</div>`);
        wer.push(`${(run.wer * 100).toFixed(0)}%`); lat.push(`${run.ms.toFixed(0)} ms`);
      }
    }
    return `<tr class="${cls}" data-i="${i}"><td class="id">${s.id}</td><td>${lines.join("")}</td>`
         + `<td class="num">${wer.join("<br>")}</td><td class="num">${lat.join("<br>")}</td></tr>`;
  }).join("");

  $("eval-summary").innerHTML = evalEngines.map((e) => {
    const runs = Object.values(evalResults).map((r) => r.runs[e]).filter((x) => x && !x.error);
    if (!runs.length) return `<div class="eval-card"><div class="eval-engine">${e}</div><div class="v">— <small>no clips yet</small></div></div>`;
    const errors = runs.reduce((n, x) => n + x.errors, 0), refWords = runs.reduce((n, x) => n + x.ref_words, 0);
    const ms = runs.map((x) => x.ms).sort((a, b) => a - b), exact = runs.filter((x) => x.errors === 0).length;
    return `<div class="eval-card"><div class="eval-engine">${e} · ${runs.length} clips · ${exact} word-perfect</div>`
         + `<div class="v">${(100 * errors / Math.max(1, refWords)).toFixed(1)}% <small>WER</small> `
         + `&nbsp; ${ms[Math.floor((ms.length - 1) / 2)].toFixed(0)} <small>ms median</small></div></div>`;
  }).join("");
}

function evalGo(i) {
  evalIndex = Math.max(0, Math.min(ASR_SCRIPT.length - 1, i));
  evalRender();
  $("eval-rows").querySelector("tr.current")?.scrollIntoView({ block: "nearest" });
}

// ---------- speech latency eval ----------
// Replays the clips recorded in ASR eval through POST /converse in the two
// pipeline configurations and compares the time to the first spoken audio:
//   audio       Gemma answers from the speech itself, then TTS
//   transcript  Parakeet transcribes, Gemma answers from the text, then TTS
// Times are the gateway's own (request received -> first sentence synthesized),
// so they exclude upload and playback. The transcript pass that the console
// shows in "hears audio" mode is switched off here: it would only add delay.
const LAT_CONFIGS = [["audio", "Hears audio"], ["transcript", "Reads transcript"]];
let latClips = null;            // [{id, text}] from the gateway
const latResults = {};          // id -> { audio: [run], transcript: [run] }, run = {first, asr, ttft, tts1}
let latRunning = false;

async function converseOnce(blob, llmInput) {
  const form = new FormData();
  form.append("audio", blob, "clip.wav");
  form.append("llm_input", llmInput);
  form.append("transcribe", "false");
  form.append("engine", $("tts-select").value || "");
  const resp = await fetch("/converse", { method: "POST", body: form });
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  const text = await resp.text();           // the timings come from the gateway, so no need to stream
  let sid = null, metrics = null;
  for (const block of text.split("\n\n")) {
    let ev = "", data = "";
    for (const line of block.split("\n")) {
      if (line.startsWith("event:")) ev = line.slice(6).trim();
      else if (line.startsWith("data:")) data += line.slice(5).trim();
    }
    if (ev === "session") sid = JSON.parse(data).session_id;
    else if (ev === "done") metrics = JSON.parse(data).metrics;
  }
  if (sid) { const f = new FormData(); f.append("session_id", sid); fetch("/reset", { method: "POST", body: f }); }
  if (!metrics || metrics.time_to_first_audio_ms == null) throw new Error("no audio in reply");
  return { first: metrics.time_to_first_audio_ms, asr: metrics.asr_ms || 0, ttft: metrics.ttft_ms,
           tts1: (metrics.tts_segments[0] || {}).client_ms || 0 };
}

const latMedian = (xs) => { const s = [...xs].sort((a, b) => a - b); return s.length ? s[Math.floor((s.length - 1) / 2)] : null; };
const latP95 = (xs) => { const s = [...xs].sort((a, b) => a - b); return s.length ? s[Math.min(s.length - 1, Math.ceil(s.length * 0.95) - 1)] : null; };

async function latLoadClips() {
  try {
    const manifest = await (await fetch("/eval/asr/clips")).json();
    latClips = Object.entries(manifest).map(([id, c]) => ({ id, text: c.text })).sort((a, b) => a.id.localeCompare(b.id));
  } catch { latClips = []; }
}

async function latRun() {
  if (latRunning) { latRunning = false; return; }      // the button doubles as Stop
  if (!$("tts-select").value) { $("lat-status").textContent = "No TTS voice is reachable, so there is no speech to time."; return; }
  await latLoadClips();
  if (!latClips.length) { latRender(); return; }
  const runs = Number($("lat-runs").value);
  for (const c of latClips) delete latResults[c.id];
  latRunning = true; $("lat-run").textContent = "■ Stop";
  try {
    const blobs = {};
    const blobOf = async (id) => blobs[id] || (blobs[id] = await (await fetch(`/eval/asr/clips/${id}.wav`)).blob());
    $("lat-status").textContent = "warming up…";
    for (const [cfg] of LAT_CONFIGS) await converseOnce(await blobOf(latClips[0].id), cfg).catch(() => {});
    for (let i = 0; i < latClips.length && latRunning; i++) {
      const clip = latClips[i], entry = latResults[clip.id] = { audio: [], transcript: [] };
      for (let r = 0; r < runs && latRunning; r++) {
        // Alternate which configuration goes first so neither always benefits from a warm cache.
        const order = (i + r) % 2 ? [...LAT_CONFIGS].reverse() : LAT_CONFIGS;
        for (const [cfg] of order) {
          $("lat-status").textContent = `clip ${i + 1} / ${latClips.length} · run ${r + 1} / ${runs} · ${cfg}`;
          try { entry[cfg].push(await converseOnce(await blobOf(clip.id), cfg)); }
          catch (e) { entry.error = String(e.message || e); }
        }
        latRender();
      }
    }
    $("lat-status").textContent = latRunning ? "done" : "stopped";
  } finally { latRunning = false; $("lat-run").textContent = "▶ Run"; }
}

function latRender() {
  if (latClips === null) { latLoadClips().then(latRender); return; }
  const fmt = (v) => (v == null ? "—" : `${v.toFixed(0)} ms`);
  if (!latClips.length) {
    $("lat-rows").innerHTML = `<tr class="todo"><td colspan="5">No recorded clips yet. Record some in ASR eval first.</td></tr>`;
    $("lat-summary").innerHTML = "";
    return;
  }
  $("lat-rows").innerHTML = latClips.map((c) => {
    const r = latResults[c.id];
    const a = r ? latMedian(r.audio.map((x) => x.first)) : null, t = r ? latMedian(r.transcript.map((x) => x.first)) : null;
    const diff = a != null && t != null ? t - a : null;
    const cls = diff == null ? "" : diff < 0 ? "faster" : "slower";
    return `<tr class="${r ? "" : "todo"}"><td class="id">${c.id}</td><td>${esc(c.text)}${r && r.error ? ` <b>${esc(r.error)}</b>` : ""}</td>`
         + `<td class="num">${fmt(a)}</td><td class="num">${fmt(t)}</td>`
         + `<td class="num ${cls}">${diff == null ? "—" : (diff > 0 ? "+" : "") + diff.toFixed(0) + " ms"}</td></tr>`;
  }).join("");

  $("lat-summary").innerHTML = LAT_CONFIGS.map(([cfg, label]) => {
    const all = Object.values(latResults).flatMap((r) => r[cfg]);
    if (!all.length) return `<div class="eval-card"><div class="eval-engine">${label}</div><div class="v">— <small>not run yet</small></div></div>`;
    const med = (k) => latMedian(all.map((x) => x[k]));
    // ttft is measured from the start of the request, so it already contains the ASR pass.
    return `<div class="eval-card"><div class="eval-engine">${label} · ${all.length} runs</div>`
         + `<div class="v">${med("first").toFixed(0)} <small>ms to first speech (median)</small> &nbsp; ${latP95(all.map((x) => x.first)).toFixed(0)} <small>p95</small></div>`
         + `<div class="parts">asr ${med("asr").toFixed(0)} ms · llm first token ${(med("ttft") - med("asr")).toFixed(0)} ms · first sentence tts ${med("tts1").toFixed(0)} ms</div></div>`;
  }).join("");
}

// ---------- UI ----------
function addMessage(role, text) {
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  el.innerHTML = `<span class="who">${role === "user" ? "you" : "robot"}</span><span class="body"></span>`;
  el.querySelector(".body").textContent = text;
  $("transcript").appendChild(el); scrollDown();
  return el;
}
function scrollDown() { const t = $("transcript"); t.scrollTop = t.scrollHeight; }

// ---------- webcam image input ----------
async function openCam() {
  const v = $("cam-video");
  try {
    if (!navigator.mediaDevices?.getUserMedia)
      throw new Error("getUserMedia unavailable (needs https or localhost)");
    camStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    v.srcObject = camStream;
    $("cam-tray").hidden = false;
    setMic("camera…", true);

    const track = camStream.getVideoTracks()[0];
    console.log("[cam] device:", track?.label, "| state:", track?.readyState,
                "| muted:", track?.muted, "| settings:", track?.getSettings?.());
    navigator.mediaDevices.enumerateDevices().then((d) =>
      console.log("[cam] video inputs:", d.filter((x) => x.kind === "videoinput").map((x) => x.label || "(unnamed)"))
    ).catch(() => {});

    v.onplaying = () => setMic(`live ${v.videoWidth}×${v.videoHeight}`, true);
    if (track) track.onmute = () => setMic("camera muted (in use?)", false);
    await v.play().catch((e) => console.warn("[cam] play() failed:", e));

    // Watchdog: granted but no frames within 2.5s -> device isn't delivering video.
    setTimeout(() => {
      if (camStream && !v.videoWidth) {
        setMic("no frames — covered/in use?", false);
        console.warn("[cam] no frames after 2.5s. muted:", track?.muted,
          "readyState:", track?.readyState,
          "→ test the Windows Camera app; check privacy settings + other apps holding the camera.");
      }
    }, 2500);
  } catch (e) {
    closeCam();
    $("toggle-vision").checked = false;          // reflect the failure on the toggle
    setMic("no camera", false);
    console.error("[cam] getUserMedia failed:", e);
    alert("Webcam error: " + (e.message || e) +
      "\n\nCheck the camera permission (icon in the address bar) and that no other app " +
      "(Teams / Zoom / OBS / Camera) is holding the camera.");
  }
}
function closeCam() {
  if (camStream) { camStream.getTracks().forEach((t) => t.stop()); camStream = null; }
  $("cam-tray").hidden = true;
}
// Grab one JPEG frame from the live preview; resolves null if no frame is available.
function captureFrame() {
  return new Promise((resolve) => {
    const v = $("cam-video");
    if (!camStream || !v.videoWidth || !v.videoHeight) { resolve(null); return; }
    const scale = Math.min(1, CAM_MAX_DIM / Math.max(v.videoWidth, v.videoHeight));
    const w = Math.round(v.videoWidth * scale), h = Math.round(v.videoHeight * scale);
    const c = document.createElement("canvas"); c.width = w; c.height = h;
    c.getContext("2d").drawImage(v, 0, 0, w, h);
    c.toBlob((b) => resolve(b), "image/jpeg", 0.85);
  });
}
function addImageToMsg(el, blob) {
  const img = document.createElement("img");
  img.className = "msg-img";
  img.src = URL.createObjectURL(blob);     // own URL, independent of the chip thumbnail
  el.appendChild(img);                     // sibling of .body, so a transcript update can't wipe it
}

function updateMetrics(m) {
  const ms = (v) => (v == null ? "—" : `${v.toFixed(0)} ms`);
  $("m-ttft").innerHTML = m.ttft_ms == null ? "—" : `${m.ttft_ms.toFixed(0)}<span class="u">ms</span>`;
  $("m-tps").textContent = m.tokens_per_sec ? m.tokens_per_sec.toFixed(1) : "—";
  $("m-total").textContent = ms(m.total_ms);
  $("m-pre").textContent = ms(m.preprocess_ms);
  $("m-audio").textContent = m.audio_seconds != null ? `${m.audio_seconds.toFixed(1)} s` : "—";
  $("m-tokens").textContent = m.output_tokens ?? "—";
  $("m-audio1").textContent = m.time_to_first_audio_ms != null ? `${m.time_to_first_audio_ms.toFixed(0)} ms` : "—";
  $("m-asr").textContent = ms(m.asr_ms);
  $("m-tts").textContent = m.tts_total_ms != null ? `${m.tts_total_ms.toFixed(0)} ms` : "—";
  renderBreakdown(m.components || []);
  updateGauge(m.ttft_ms);

  sess.turns += 1; sess.ttftSum += m.ttft_ms || 0; sess.tpsSum += m.tokens_per_sec || 0;
  $("m-turns").textContent = sess.turns;
  $("m-avg-ttft").textContent = `${(sess.ttftSum / sess.turns).toFixed(0)} ms`;
  $("m-avg-tps").textContent = (sess.tpsSum / sess.turns).toFixed(1);
}

// Render the per-component latency breakdown as labeled proportional bars.
function renderBreakdown(components) {
  const host = $("m-breakdown");
  if (!components.length) { host.innerHTML = `<div class="bd-empty">—</div>`; return; }
  const max = Math.max(...components.map((c) => c.ms || 0), 1);
  host.innerHTML = components.map((c) => {
    const pct = Math.max(2, ((c.ms || 0) / max) * 100);
    let sub = "";
    if (c.name === "llm" && c.ttft_ms != null) sub = `ttft ${c.ttft_ms.toFixed(0)}`;
    else if (c.name === "asr" && c.engine) sub = c.engine;
    else if (c.name === "tts") {
      sub = c.calls != null ? `${c.calls}×` : "";
      if (c.server_ms != null) sub += `${sub ? " · " : ""}net ${(c.ms - c.server_ms).toFixed(0)}`;
    }
    return `<div class="bd-row" title="${c.name}: ${(c.ms || 0).toFixed(1)} ms">`
      + `<span class="bd-name">${c.name}${sub ? `<span class="bd-sub"> ${sub}</span>` : ""}</span>`
      + `<span class="bd-bar"><span style="width:${pct}%"></span></span>`
      + `<span class="bd-ms">${(c.ms || 0).toFixed(0)}</span></div>`;
  }).join("");
}

function updateGauge(ttft) {
  if (ttft == null) return;
  const pct = Math.min(ttft / GAUGE_MAX_MS, 1) * 100;
  $("gauge-marker").style.left = `calc(${pct}% - 1.5px)`;
  const v = $("ttft-verdict");
  if (ttft <= 300) { v.textContent = "snappy"; v.className = "verdict v-good"; }
  else if (ttft <= 800) { v.textContent = "usable"; v.className = "verdict v-warn"; }
  else { v.textContent = "too slow"; v.className = "verdict v-bad"; }
}

// ---------- wiring ----------
const talkBtn = $("btn-talk"), vad = $("toggle-vad");
const ptt = () => !vad.checked && mode !== "latency";
talkBtn.addEventListener("mousedown", () => ptt() && startRecording());
talkBtn.addEventListener("mouseup", () => ptt() && stopRecording());
talkBtn.addEventListener("mouseleave", () => recording && ptt() && stopRecording());
talkBtn.addEventListener("touchstart", (e) => { e.preventDefault(); ptt() && startRecording(); }, { passive: false });
talkBtn.addEventListener("touchend", (e) => { e.preventDefault(); ptt() && stopRecording(); });

// Keys typed into a text field belong to that field, not to push-to-talk.
const typing = (e) => ["INPUT", "TEXTAREA", "SELECT"].includes(e.target.tagName) && e.target.type !== "checkbox";
addEventListener("keydown", (e) => {
  if (typing(e)) return;
  if (e.code === "Space" && ptt()) {
    // Swallow every press, auto-repeats included: otherwise holding space
    // scrolls the page or "clicks" whichever button has focus.
    e.preventDefault();
    if (!e.repeat) startRecording();
  } else if (evalMode && !recording) {
    if (e.code === "ArrowRight") { e.preventDefault(); evalGo(evalIndex + 1); }
    else if (e.code === "ArrowLeft") { e.preventDefault(); evalGo(evalIndex - 1); }
  }
});
addEventListener("keyup", (e) => {
  if (e.code === "Space" && ptt() && !typing(e)) { e.preventDefault(); stopRecording(); }
});

vad.addEventListener("change", async (e) => {
  if (e.target.checked) {
    talkBtn.innerHTML = "🔴 Continuous — listening";
    talkBtn.disabled = true;
    try { await startContinuous(); }
    catch (err) { console.error("[ws] start failed:", err); e.target.checked = false; e.target.dispatchEvent(new Event("change")); }
  } else {
    talkBtn.innerHTML = evalMode ? "🎙 Hold to read <kbd>space</kbd>" : "🎙 Hold to talk <kbd>space</kbd>";
    talkBtn.disabled = mode === "latency";
    stopContinuous();
  }
});

$("toggle-vision").addEventListener("change", (e) => { e.target.checked ? openCam() : closeCam(); });
$("asr-select").addEventListener("change", (e) => { e.target.dataset.touched = "1"; });

function clearTranscript() {
  $("transcript").innerHTML = `<div class="empty" id="empty"><span class="mark">◍</span>Hold the button (or <kbd>space</kbd>) and speak. The robot brain replies in text, latency on the right.</div>`;
  sess.turns = 0; sess.ttftSum = 0; sess.tpsSum = 0;
  $("m-turns").textContent = "0"; $("m-avg-ttft").textContent = "—"; $("m-avg-tps").textContent = "—";
  $("m-breakdown").innerHTML = `<div class="bd-empty">—</div>`;
}

$("btn-reset").addEventListener("click", async () => {
  if (sessionId) { const f = new FormData(); f.append("session_id", sessionId); await fetch("/reset", { method: "POST", body: f }); }
  closeCam(); $("toggle-vision").checked = false;
  sessionId = null;
  $("session-id").textContent = "no session";
  clearTranscript();
  sendConfig();                                 // continuous mode: server issues a fresh session
  // Carry the active persona into the fresh session (no-op when never set).
  if (appliedPersona) applyPersona();
});

// ---------- persona ----------
// Per-session system prompt override. Presets come from /web/personas.js;
// the textarea stays editable so any prompt can be applied. Applying clears
// the server-side history (a persona switch mid-conversation bleeds voices).
let appliedPersona = "";                       // "" = server default

function initPersona() {
  const sel = $("persona-select");
  for (const [key, p] of Object.entries(PERSONAS)) {
    const o = document.createElement("option");
    o.value = key; o.textContent = p.label;
    sel.appendChild(o);
  }
  sel.onchange = () => { $("persona-text").value = PERSONAS[sel.value].prompt; };
}

async function applyPersona() {
  const prompt = $("persona-text").value.trim();
  const form = new FormData();
  if (sessionId) form.append("session_id", sessionId);
  form.append("system_prompt", prompt);
  const data = await (await fetch("/persona", { method: "POST", body: form })).json();
  sessionId = data.session_id;
  $("session-id").textContent = sessionId.slice(0, 12);
  appliedPersona = prompt;
  const preset = Object.values(PERSONAS).find((p) => p.prompt.trim() === prompt);
  $("persona-active").textContent = prompt ? (preset ? preset.label : "custom") : "default";
  clearTranscript();                           // server cleared history; mirror it
  sendConfig();
}

$("btn-persona").addEventListener("click", applyPersona);

$("mode-chat").addEventListener("click", () => setMode("chat"));
$("mode-eval").addEventListener("click", () => setMode("asr"));
$("mode-latency").addEventListener("click", () => setMode("latency"));
$("lat-run").addEventListener("click", latRun);
$("input-select").addEventListener("change", (e) => {
  e.target.dataset.touched = "1";
  // A cascade is only worth it with the fast ASR: switch to Parakeet when it's loaded.
  const asr = $("asr-select");
  if (e.target.value === "transcript" && [...asr.options].some((o) => o.value === "parakeet")) { asr.value = "parakeet"; asr.dataset.touched = "1"; }
  sendConfig();
});
$("eval-prev").addEventListener("click", () => evalGo(evalIndex - 1));
$("eval-next").addEventListener("click", () => evalGo(evalIndex + 1));
$("eval-play").addEventListener("click", () => {
  const r = evalResults[ASR_SCRIPT[evalIndex].id];
  if (r) playBlobs([r.blob]);
});
$("eval-rows").addEventListener("click", (e) => { const tr = e.target.closest("tr"); if (tr) evalGo(Number(tr.dataset.i)); });
initPersona();

sizeCanvas();
drawWave();
refreshHealth();
setInterval(refreshHealth, 15000);

// TTS engines load their model on container start, so they're unreachable for
// the first few seconds after this page loads. Poll quickly until at least one
// voice appears, then fall back to the 15s health cadence — otherwise the
// dropdown sits on "— no voices —" until the first message happens to warm it.
(async function waitForVoices() {
  for (let i = 0; i < 40; i++) {          // ~60s ceiling for a cold start
    if (await refreshEngines()) return;
    await new Promise((r) => setTimeout(r, 1500));
  }
})();
