const statusEl = document.getElementById("status");
const progressCountEl = document.getElementById("progressCount");
const participantSelect = document.getElementById("participantSelect");
const levelEl = document.getElementById("level");
const scriptIndexEl = document.getElementById("scriptIndex");
const scriptStateEl = document.getElementById("scriptState");
const scriptTextEl = document.getElementById("scriptText");
const recordBtn = document.getElementById("recordBtn");
const durationEl = document.getElementById("duration");
const durationHintEl = document.getElementById("durationHint");
const player = document.getElementById("player");
const prevBtn = document.getElementById("prevBtn");
const saveBtn = document.getElementById("saveBtn");
const resetBtn = document.getElementById("resetBtn");
const nextBtn = document.getElementById("nextBtn");
const BrowserAudioContext = window.AudioContext || window.webkitAudioContext;

let participants = [];
let scripts = [];
let records = {};
let currentIndex = 0;
let audioContext = null;
let sourceNode = null;
let processorNode = null;
let stream = null;
let chunks = [];
let recordingStart = 0;
let levelTimer = null;
let isRecording = false;
let lastWavBlob = null;
let lastDurationSec = 0;

function canUseDirectMic() {
  return Boolean(window.isSecureContext && navigator.mediaDevices?.getUserMedia);
}

function selectedParticipantId() {
  return participantSelect.value;
}

function currentScript() {
  return scripts[currentIndex] || null;
}

function setStatus(text) {
  statusEl.textContent = text;
}

async function readJsonResponse(response) {
  const text = await response.text();
  try {
    return JSON.parse(text);
  } catch (error) {
    const preview = text.replace(/\s+/g, " ").slice(0, 220);
    throw new Error(`Backend returned non-JSON HTTP ${response.status}: ${preview}`);
  }
}

async function loadConfig() {
  const response = await fetch("/collection/config");
  const data = await readJsonResponse(response);
  if (!data.ok) throw new Error(data.error || "Collection config failed.");
  participants = data.participants;
  scripts = data.scripts;
  participantSelect.replaceChildren(...participants.map((participant) => {
    const option = document.createElement("option");
    option.value = participant.id;
    option.textContent = participant.name;
    return option;
  }));
  const savedParticipant = localStorage.getItem("collectParticipantId");
  if (participants.some((item) => item.id === savedParticipant)) {
    participantSelect.value = savedParticipant;
  }
  currentIndex = Number(localStorage.getItem(`collectIndex:${selectedParticipantId()}`) || 0);
  currentIndex = Math.max(0, Math.min(currentIndex, scripts.length - 1));
  await loadProgress();
  setStatus("Ready to collect clean retraining audio.");
}

async function loadProgress() {
  const participantId = selectedParticipantId();
  if (!participantId) return;
  const response = await fetch(`/collection/progress?participant_id=${encodeURIComponent(participantId)}`);
  const data = await readJsonResponse(response);
  if (!data.ok) throw new Error(data.error || "Progress failed.");
  records = data.records || {};
  updateView();
}

function updateView() {
  const script = currentScript();
  const total = scripts.length;
  const completed = scripts.filter((item) => records[item.id]?.status === "recorded").length;
  progressCountEl.textContent = `${completed}/${total}`;
  prevBtn.disabled = currentIndex <= 0;
  nextBtn.disabled = currentIndex >= total - 1;

  if (!script) {
    scriptIndexEl.textContent = "Sentence -";
    scriptTextEl.textContent = "No scripts found.";
    recordBtn.disabled = true;
    saveBtn.disabled = true;
    resetBtn.disabled = true;
    return;
  }

  const record = records[script.id];
  scriptIndexEl.textContent = `Sentence ${currentIndex + 1} of ${total} | ${script.wordCount} words`;
  scriptStateEl.textContent = record?.status === "recorded" ? "Recorded" : "Not recorded";
  scriptStateEl.classList.toggle("recorded", record?.status === "recorded");
  scriptTextEl.textContent = script.text;
  resetBtn.disabled = record?.status !== "recorded";
  saveBtn.disabled = !lastWavBlob;
  localStorage.setItem(`collectIndex:${selectedParticipantId()}`, String(currentIndex));
}

function updateDurationHint(sec) {
  durationHintEl.classList.remove("good", "warn");
  if (sec >= 8 && sec <= 18) {
    durationHintEl.textContent = "Good natural pace.";
    durationHintEl.classList.add("good");
  } else if (sec > 0 && sec < 5) {
    durationHintEl.textContent = "A bit short. Read naturally, not too fast.";
    durationHintEl.classList.add("warn");
  } else if (sec > 20) {
    durationHintEl.textContent = "A bit long. It can still be saved.";
    durationHintEl.classList.add("warn");
  } else {
    durationHintEl.textContent = "Aim for natural reading.";
  }
}

function downsample(buffer, sourceRate, targetRate) {
  if (sourceRate === targetRate) return buffer;
  const ratio = sourceRate / targetRate;
  const newLength = Math.round(buffer.length / ratio);
  const result = new Float32Array(newLength);
  for (let i = 0; i < newLength; i++) {
    const pos = i * ratio;
    const left = Math.floor(pos);
    const right = Math.min(left + 1, buffer.length - 1);
    const frac = pos - left;
    result[i] = buffer[left] * (1 - frac) + buffer[right] * frac;
  }
  return result;
}

function encodeWav(samples, sampleRate) {
  const bytesPerSample = 2;
  const buffer = new ArrayBuffer(44 + samples.length * bytesPerSample);
  const view = new DataView(buffer);

  function writeString(offset, text) {
    for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i));
  }

  writeString(0, "RIFF");
  view.setUint32(4, 36 + samples.length * bytesPerSample, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * bytesPerSample, true);
  view.setUint16(32, bytesPerSample, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, samples.length * bytesPerSample, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([view], { type: "audio/wav" });
}

function mergeChunks() {
  const total = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const merged = new Float32Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }
  return merged;
}

async function startRecording() {
  if (!canUseDirectMic()) {
    setStatus("Recording needs HTTPS and microphone permission.");
    return;
  }
  lastWavBlob = null;
  player.removeAttribute("src");
  chunks = [];
  saveBtn.disabled = true;
  updateDurationHint(0);
  stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
  });
  audioContext = new BrowserAudioContext();
  sourceNode = audioContext.createMediaStreamSource(stream);
  processorNode = audioContext.createScriptProcessor(4096, 1, 1);
  processorNode.onaudioprocess = (event) => {
    const input = event.inputBuffer.getChannelData(0);
    chunks.push(new Float32Array(input));
    let peak = 0;
    for (let i = 0; i < input.length; i++) peak = Math.max(peak, Math.abs(input[i]));
    levelEl.style.width = `${Math.min(100, Math.round(peak * 120))}%`;
  };
  sourceNode.connect(processorNode);
  processorNode.connect(audioContext.destination);
  recordingStart = performance.now();
  levelTimer = setInterval(() => {
    const sec = (performance.now() - recordingStart) / 1000;
    durationEl.textContent = `${sec.toFixed(1)}s`;
    updateDurationHint(sec);
  }, 100);
  isRecording = true;
  recordBtn.textContent = "Stop";
  recordBtn.classList.add("recording");
  setStatus("Recording...");
}

async function stopRecording() {
  if (processorNode) processorNode.disconnect();
  if (sourceNode) sourceNode.disconnect();
  if (stream) stream.getTracks().forEach((track) => track.stop());
  if (levelTimer) clearInterval(levelTimer);
  levelEl.style.width = "0%";

  const merged = mergeChunks();
  const resampled = downsample(merged, audioContext.sampleRate, 16000);
  lastDurationSec = resampled.length / 16000;
  lastWavBlob = encodeWav(resampled, 16000);
  player.src = URL.createObjectURL(lastWavBlob);
  durationEl.textContent = `${lastDurationSec.toFixed(1)}s`;
  updateDurationHint(lastDurationSec);
  await audioContext.close();

  isRecording = false;
  recordBtn.textContent = "Record";
  recordBtn.classList.remove("recording");
  saveBtn.disabled = false;
  setStatus("Recording ready. Replay it, then save.");
}

async function saveRecording() {
  const script = currentScript();
  const participantId = selectedParticipantId();
  if (!script || !lastWavBlob || !participantId) return;
  saveBtn.disabled = true;
  setStatus("Saving recording...");
  const url = `/collection/record?participant_id=${encodeURIComponent(participantId)}&script_id=${encodeURIComponent(script.id)}&duration_sec=${encodeURIComponent(lastDurationSec.toFixed(3))}`;
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "audio/wav" },
    body: lastWavBlob,
  });
  const data = await readJsonResponse(response);
  if (!data.ok) throw new Error(data.error || "Save failed.");
  records[script.id] = {
    status: "recorded",
    sample_id: data.record.id,
    audio_path: data.record.audio_path,
    duration_sec: data.record.duration_sec,
    updated_at: data.record.created_at,
  };
  lastWavBlob = null;
  player.removeAttribute("src");
  setStatus("Saved. Moving to next missing sentence.");
  const nextMissing = scripts.findIndex((item, index) => index > currentIndex && records[item.id]?.status !== "recorded");
  if (nextMissing >= 0) {
    currentIndex = nextMissing;
  } else {
    const anyMissing = scripts.findIndex((item) => records[item.id]?.status !== "recorded");
    if (anyMissing >= 0) currentIndex = anyMissing;
  }
  updateView();
}

async function resetRecording() {
  const script = currentScript();
  const participantId = selectedParticipantId();
  if (!script || !participantId) return;
  const response = await fetch("/collection/reset-recording", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ participant_id: participantId, script_id: script.id }),
  });
  const data = await readJsonResponse(response);
  if (!data.ok) throw new Error(data.error || "Reset failed.");
  delete records[script.id];
  lastWavBlob = null;
  player.removeAttribute("src");
  setStatus("Ready to re-record this sentence.");
  updateView();
}

recordBtn.addEventListener("click", () => {
  const action = isRecording ? stopRecording() : startRecording();
  action.catch((error) => {
    setStatus(`Microphone error: ${error}`);
    isRecording = false;
    recordBtn.textContent = "Record";
    recordBtn.classList.remove("recording");
    recordBtn.disabled = false;
  });
});

saveBtn.addEventListener("click", () => saveRecording().catch((error) => setStatus(String(error))));
resetBtn.addEventListener("click", () => resetRecording().catch((error) => setStatus(String(error))));
prevBtn.addEventListener("click", () => {
  currentIndex = Math.max(0, currentIndex - 1);
  lastWavBlob = null;
  player.removeAttribute("src");
  updateView();
});
nextBtn.addEventListener("click", () => {
  currentIndex = Math.min(scripts.length - 1, currentIndex + 1);
  lastWavBlob = null;
  player.removeAttribute("src");
  updateView();
});
participantSelect.addEventListener("change", async () => {
  localStorage.setItem("collectParticipantId", selectedParticipantId());
  currentIndex = Number(localStorage.getItem(`collectIndex:${selectedParticipantId()}`) || 0);
  currentIndex = Math.max(0, Math.min(currentIndex, scripts.length - 1));
  lastWavBlob = null;
  player.removeAttribute("src");
  await loadProgress();
});

loadConfig().catch((error) => {
  setStatus(String(error));
  scriptTextEl.textContent = "Could not load collection.";
});
