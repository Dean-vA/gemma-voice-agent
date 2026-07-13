// Gemma 4 E4B voice console — robot simulator.
// Captures mic audio, resamples to 16 kHz mono WAV client-side, streams the
// reply over SSE, draws a live waveform, and reports per-turn latency against a
// robot-conversation budget. Push-to-talk + optional VAD loop.

const TARGET_SR = 16000;
const VAD_SILENCE_MS = 700;
const VAD_RMS_THRESHOLD = 0.012;
const VAD_MIN_SPEECH_MS = 300;
const GAUGE_MAX_MS = 1500;          // gauge full-scale; zones at 300 / 800

let sessionId = null;
let audioCtx = null, stream = null, sourceNode = null, procNode = null, analyser = null;
let recording = false;
let captured = [];
let captureSampleRate = 48000;

let speechMs = 0, silenceMs = 0, sawSpeech = false;
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
  const live = recording && analyser;
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
  stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } });
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  captureSampleRate = audioCtx.sampleRate;
  sourceNode = audioCtx.createMediaStreamSource(stream);
  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 1024;
  sourceNode.connect(analyser);
  procNode = audioCtx.createScriptProcessor(4096, 1, 1);
  procNode.onaudioprocess = onAudio;
  sourceNode.connect(procNode);
  procNode.connect(audioCtx.destination);
}

function onAudio(e) {
  if (!recording) return;
  const input = e.inputBuffer.getChannelData(0);
  captured.push(new Float32Array(input));
  if (!$("toggle-vad").checked) return;
  let sum = 0;
  for (let i = 0; i < input.length; i++) sum += input[i] * input[i];
  const rms = Math.sqrt(sum / input.length);
  const frameMs = (input.length / captureSampleRate) * 1000;
  if (rms > VAD_RMS_THRESHOLD) { speechMs += frameMs; silenceMs = 0; if (speechMs > VAD_MIN_SPEECH_MS) sawSpeech = true; }
  else { silenceMs += frameMs; if (sawSpeech && silenceMs > VAD_SILENCE_MS) stopRecording(true); }
}

async function startRecording() {
  await ensureMic();
  if (audioCtx.state === "suspended") await audioCtx.resume();
  captured = []; speechMs = silenceMs = 0; sawSpeech = false; recording = true;
  $("btn-talk").classList.add("recording");
  setMic("listening", true);
}

async function stopRecording(autoSend) {
  if (!recording) return;
  recording = false;
  $("btn-talk").classList.remove("recording");
  setMic("processing", true);

  const total = captured.reduce((n, c) => n + c.length, 0);
  if (total < captureSampleRate * 0.2) { setMic("idle", false); if ($("toggle-vad").checked && autoSend) maybeRelisten(); return; }
  const down = resampleTo16k(mergeChunks(captured, total), captureSampleRate);
  await sendAudio(encodeWav(down, TARGET_SR));
  if ($("toggle-vad").checked) maybeRelisten(); else setMic("idle", false);
}

function maybeRelisten() { if ($("toggle-vad").checked) setTimeout(() => { if (!recording) startRecording(); }, 150); }
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
function encodeWav(samples, sampleRate) {
  const buf = new ArrayBuffer(44 + samples.length * 2), view = new DataView(buf);
  const w = (off, s) => { for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i)); };
  w(0, "RIFF"); view.setUint32(4, 36 + samples.length * 2, true); w(8, "WAVE"); w(12, "fmt ");
  view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true); view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true); view.setUint16(34, 16, true); w(36, "data");
  view.setUint32(40, samples.length * 2, true);
  let off = 44;
  for (let i = 0; i < samples.length; i++, off += 2) { const s = Math.max(-1, Math.min(1, samples[i])); view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true); }
  return new Blob([view], { type: "audio/wav" });
}

// ---------- audio playback (sequential live queue + on-demand replay) ----------
// All audio lives in JS memory only, so it's gone on page refresh.
const playQueue = [];
let playing = false;
function enqueueAudio(b64) {
  const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  const blob = new Blob([bytes], { type: "audio/wav" });
  playQueue.push(URL.createObjectURL(blob));
  if (!playing) playNext();
  return blob;
}
function playNext() {
  if (!playQueue.length) { playing = false; setMic("idle", false); return; }
  playing = true;
  setMic("speaking", true);
  const url = playQueue.shift();
  const a = new Audio(url);
  a.onended = a.onerror = () => { URL.revokeObjectURL(url); playNext(); };
  a.play().catch(() => playNext());
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
  const p = JSON.parse(data);
  if (ev === "session") { sessionId = p.session_id; $("session-id").textContent = sessionId.slice(0, 12); }
  else if (ev === "transcript") { curUser.querySelector(".body").textContent = p.text || "(no speech detected)"; scrollDown(); }
  else if (ev === "token") { curAssistant.querySelector(".body").textContent += p.text; scrollDown(); }
  else if (ev === "audio") { curAudio.push(enqueueAudio(p.wav_base64)); }
  else if (ev === "done") updateMetrics(p.metrics);
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
const ptt = () => !vad.checked;
talkBtn.addEventListener("mousedown", () => ptt() && startRecording());
talkBtn.addEventListener("mouseup", () => ptt() && stopRecording(false));
talkBtn.addEventListener("mouseleave", () => recording && ptt() && stopRecording(false));
talkBtn.addEventListener("touchstart", (e) => { e.preventDefault(); ptt() && startRecording(); }, { passive: false });
talkBtn.addEventListener("touchend", (e) => { e.preventDefault(); ptt() && stopRecording(false); });

addEventListener("keydown", (e) => {
  if (e.code === "Space" && !e.repeat && ptt() && e.target.tagName !== "INPUT") { e.preventDefault(); startRecording(); }
});
addEventListener("keyup", (e) => {
  if (e.code === "Space" && ptt() && e.target.tagName !== "INPUT") { e.preventDefault(); stopRecording(false); }
});

vad.addEventListener("change", async (e) => {
  if (e.target.checked) {
    talkBtn.innerHTML = "🔴 Continuous — listening";
    talkBtn.disabled = true;
    await startRecording();
  } else {
    talkBtn.innerHTML = "🎙 Hold to talk <kbd>space</kbd>";
    talkBtn.disabled = false;
    if (recording) stopRecording(false);
    setMic("idle", false);
  }
});

$("toggle-vision").addEventListener("change", (e) => { e.target.checked ? openCam() : closeCam(); });

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
}

$("btn-persona").addEventListener("click", applyPersona);
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
