const statusEl = document.getElementById("status");
const levelEl = document.getElementById("level");
const modelSelect = document.getElementById("modelSelect");
const speakerSelect = document.getElementById("speakerSelect");
const exampleSelect = document.getElementById("exampleSelect");
const playExampleBtn = document.getElementById("playExampleBtn");
const runExampleBtn = document.getElementById("runExampleBtn");
const nextPromptBtn = document.getElementById("nextPromptBtn");
const promptIdEl = document.getElementById("promptId");
const targetTextEl = document.getElementById("targetText");
const recordBtn = document.getElementById("recordBtn");
const durationEl = document.getElementById("duration");
const runtimeEl = document.getElementById("runtime");
const transcriptEl = document.getElementById("transcript");
const player = document.getElementById("player");
const BrowserAudioContext = window.AudioContext || window.webkitAudioContext;

let allExamples = [];
let currentPrompt = null;
let audioContext = null;
let sourceNode = null;
let processorNode = null;
let stream = null;
let chunks = [];
let recordingStart = 0;
let levelTimer = null;
let isRecording = false;
let lastWavBlob = null;

function canUseDirectMic() {
  return Boolean(window.isSecureContext && navigator.mediaDevices?.getUserMedia);
}

// Selected model id; the server maps it to a model dir + decode backend.
function currentModel() {
  return modelSelect && modelSelect.value ? modelSelect.value : "deployed-beam";
}

async function checkHealth() {
  try {
    const response = await fetch("/health");
    const data = await response.json();
    statusEl.textContent = `${data.jetsonHost} | ${data.defaultModel || data.modelDir}`;

    if (modelSelect && Array.isArray(data.models)) {
      modelSelect.innerHTML = "";
      for (const m of data.models) {
        const opt = document.createElement("option");
        opt.value = m.id;
        opt.textContent = m.label || m.id;
        modelSelect.appendChild(opt);
      }
      if (data.defaultModel) modelSelect.value = data.defaultModel;
    }
  } catch (error) {
    statusEl.textContent = "Backend is not reachable.";
  }
}

async function loadExamples() {
  try {
    const response = await fetch("/examples");
    const data = await response.json();
    if (!data.ok || !data.examples.length) throw new Error("No examples found.");
    allExamples = data.examples;
    populateSpeakerSelect();
  } catch (error) {
    targetTextEl.textContent = `Example error: ${error}`;
  }
}

async function loadRandomPrompt() {
  try {
    const response = await fetch("/prompts?random=1");
    const data = await response.json();
    if (!data.ok || !data.prompt) throw new Error("No prompt found.");
    setPrompt(data.prompt);
  } catch (error) {
    transcriptEl.textContent = `Prompt error: ${error}`;
  }
}

function populateSpeakerSelect() {
  const speakers = [...new Set(allExamples.map((example) => example.speaker || "Unknown"))].sort();
  speakerSelect.replaceChildren(...speakers.map((speaker) => {
    const option = document.createElement("option");
    option.value = speaker;
    option.textContent = `${speaker} (${allExamples.filter((example) => (example.speaker || "Unknown") === speaker).length})`;
    return option;
  }));
  updateExampleSelect();
}

function updateExampleSelect() {
  const speaker = speakerSelect.value;
  const examples = allExamples.filter((example) => (example.speaker || "Unknown") === speaker);
  exampleSelect.replaceChildren(...examples.map((example) => {
    const option = document.createElement("option");
    option.value = example.id;
    option.textContent = `${example.id} - ${example.text}`;
    return option;
  }));
  const enabled = examples.length > 0;
  exampleSelect.disabled = !enabled;
  playExampleBtn.disabled = !enabled;
  runExampleBtn.disabled = !enabled;
  if (enabled) setPrompt(examples[0]);
}

function selectedExample() {
  return allExamples.find((example) => example.id === exampleSelect.value) || null;
}

function setPrompt(prompt) {
  currentPrompt = {
    id: prompt.id,
    speaker: prompt.speaker || "",
    text: prompt.text || "",
  };
  promptIdEl.textContent = `${currentPrompt.id}${currentPrompt.speaker ? ` | ${currentPrompt.speaker}` : ""}`;
  targetTextEl.textContent = currentPrompt.text;
  transcriptEl.textContent = "Ready to record.";
  runtimeEl.textContent = "-";
}

function playSelectedExample() {
  const example = selectedExample();
  if (!example) return;
  setPrompt(example);
  player.src = `/example-audio?id=${encodeURIComponent(example.id)}`;
  player.play();
}

async function runSelectedExample() {
  const example = selectedExample();
  if (!example) return;
  setPrompt(example);
  await sendExampleToJetson(example.id);
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
    for (let i = 0; i < text.length; i++) {
      view.setUint8(offset + i, text.charCodeAt(i));
    }
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
    transcriptEl.textContent = "Recording needs HTTPS and microphone permission.";
    return;
  }

  transcriptEl.textContent = "Listening...";
  runtimeEl.textContent = "-";
  lastWavBlob = null;
  chunks = [];
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
    durationEl.textContent = `${((performance.now() - recordingStart) / 1000).toFixed(1)}s`;
  }, 100);

  isRecording = true;
  recordBtn.textContent = "Stop & Run ASR";
  recordBtn.classList.add("recording");
}

async function stopRecordingAndSend() {
  if (processorNode) processorNode.disconnect();
  if (sourceNode) sourceNode.disconnect();
  if (stream) stream.getTracks().forEach((track) => track.stop());
  if (levelTimer) clearInterval(levelTimer);
  levelEl.style.width = "0%";

  const merged = mergeChunks();
  const resampled = downsample(merged, audioContext.sampleRate, 16000);
  lastWavBlob = encodeWav(resampled, 16000);
  player.src = URL.createObjectURL(lastWavBlob);
  durationEl.textContent = `${(resampled.length / 16000).toFixed(1)}s`;
  await audioContext.close();

  isRecording = false;
  recordBtn.textContent = "Record";
  recordBtn.classList.remove("recording");
  await sendRecordingToJetson();
}

async function sendRecordingToJetson() {
  if (!lastWavBlob) return;
  recordBtn.disabled = true;
  transcriptEl.textContent = "Running ASR on Jetson...";
  const started = performance.now();
  try {
    const response = await fetch(`/transcribe?model=${encodeURIComponent(currentModel())}`, {
      method: "POST",
      headers: { "Content-Type": "audio/wav" },
      body: lastWavBlob,
    });
    const data = await readJsonResponse(response);
    runtimeEl.textContent = `${((performance.now() - started) / 1000).toFixed(2)}s`;
    transcriptEl.textContent = data.ok ? (data.transcript || "(empty transcript)") : (data.error || "ASR failed.");
    if (!data.ok) console.error(data);
  } catch (error) {
    transcriptEl.textContent = String(error);
  } finally {
    recordBtn.disabled = false;
  }
}

async function sendExampleToJetson(exampleId) {
  runExampleBtn.disabled = true;
  transcriptEl.textContent = `Running ${exampleId} on Jetson...`;
  const started = performance.now();
  try {
    const response = await fetch(
      `/transcribe-example?id=${encodeURIComponent(exampleId)}&model=${encodeURIComponent(currentModel())}`,
      { method: "POST" },
    );
    const data = await readJsonResponse(response);
    runtimeEl.textContent = `${((performance.now() - started) / 1000).toFixed(2)}s`;
    transcriptEl.textContent = data.ok ? (data.transcript || "(empty transcript)") : (data.error || "ASR failed.");
    if (!data.ok) console.error(data);
  } catch (error) {
    transcriptEl.textContent = String(error);
  } finally {
    runExampleBtn.disabled = false;
  }
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

recordBtn.addEventListener("click", () => {
  const action = isRecording ? stopRecordingAndSend() : startRecording();
  action.catch((error) => {
    transcriptEl.textContent = `Microphone error: ${error}`;
    isRecording = false;
    recordBtn.textContent = "Record";
    recordBtn.classList.remove("recording");
    recordBtn.disabled = false;
  });
});
speakerSelect.addEventListener("change", updateExampleSelect);
exampleSelect.addEventListener("change", () => {
  const example = selectedExample();
  if (example) setPrompt(example);
});
playExampleBtn.addEventListener("click", playSelectedExample);
runExampleBtn.addEventListener("click", runSelectedExample);
nextPromptBtn.addEventListener("click", loadRandomPrompt);

checkHealth();
loadExamples();
