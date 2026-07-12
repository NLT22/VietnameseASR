#!/usr/bin/env python3
"""
Local real-time streaming ASR UI with Silero VAD endpointing.

Words appear while you speak; each sentence is finalized on a pause and a fresh
decoder session starts.

  python3 live_ui/server.py [--model-dir DIR] [--host 0.0.0.0] [--port 8100]
"""
import argparse
import asyncio
import os
import time
from collections import deque

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from stream_decoder import SAMPLE_RATE, StreamingSession
from vad import FRAME, Endpointer, SileroVAD

try:
    from speaker_id import SpeakerEmbedder, load_enrolled, identify, ENROLL_JSON
    _HAS_SPK = True
except Exception:
    _HAS_SPK = False

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.abspath(
    os.path.join(HERE, "..", "deploy", "jetson_nano",
                 "model_medium_epoch30_avg10"))  # medium (M) 5 spk; real 1.84% held-out 1.80%
                 # backup: model_small_epoch50_avg10 (small, 2.72%/2.28%)

app = FastAPI()
CFG = {"model_dir": DEFAULT_MODEL, "beam": 4, "vad": True,
       "threshold": 0.5, "min_silence_ms": 700, "spk": None}


@app.get("/")
async def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"))


@app.get("/health")
async def health():
    spk = CFG.get("spk")
    return {"ok": True, "modelDir": os.path.basename(CFG["model_dir"]),
            "beam": CFG["beam"], "sampleRate": SAMPLE_RATE,
            "vad": CFG["vad"], "minSilenceMs": CFG["min_silence_ms"],
            "speakerId": bool(spk),
            "enrolled": sorted(spk[1].keys()) if spk else [],
            "decoder": "classic beam_search (incremental)"}


class LiveSession:
    """VAD-gated continuous dictation over one WebSocket."""

    # Speech onset is detected a few frames late; replay this much audio so the
    # first phoneme is not clipped.
    PREROLL_FRAMES = 8   # 8 * 32 ms = 256 ms
    # The endpoint only fires after min_silence_ms. Feeding that whole silence
    # to the decoder makes it hallucinate tokens ("...dừng lại và t"), so stop
    # feeding once silence begins -- but keep a short pad for final consonants.
    TAIL_FRAMES = 6      # 6 * 32 ms = 192 ms

    def __init__(self):
        self.vad = SileroVAD() if CFG["vad"] else None
        self.ep = Endpointer(threshold=CFG["threshold"],
                             min_silence_ms=CFG["min_silence_ms"]) if CFG["vad"] else None
        self.preroll = deque(maxlen=self.PREROLL_FRAMES)
        self.buf = np.zeros(0, dtype=np.float32)
        self.asr = None
        self.speaking = False
        self.seg = []                    # raw waveform of the current segment
        self.spk = CFG.get("spk")        # (embedder, enrolled, threshold) or None

    def _new_asr(self):
        return StreamingSession(CFG["model_dir"], CFG["beam"])

    def _label(self):
        """Identify the speaker of the just-finished segment, or None."""
        if not self.spk or not self.seg:
            return None
        embedder, enrolled, thr = self.spk
        who, score = identify(embedder.embed(np.concatenate(self.seg)), enrolled, thr)
        return {"speaker": who, "score": round(score, 3)}

    def feed(self, pcm):
        """Returns a list of (event, payload) to send to the client."""
        if self.vad is None:                      # VAD off: one long session
            if self.asr is None:
                self.asr = self._new_asr()
            return [("partial", self.asr.accept_waveform(pcm))]

        out = []
        self.buf = np.concatenate([self.buf, pcm])
        while len(self.buf) >= FRAME:
            frame, self.buf = self.buf[:FRAME], self.buf[FRAME:]
            prob = self.vad(frame)
            ev = self.ep.update(prob)

            if ev == "start":
                self.speaking = True
                self.asr = self._new_asr()
                out.append(("speech", True))
                self.seg = list(self.preroll)    # segment audio starts at onset
                for f in self.preroll:           # replay the onset
                    self.asr.accept_waveform(f)
                self.preroll.clear()

            if self.speaking:
                # Feed speech, plus a short tail; withhold the long silence run.
                if prob >= self.ep.lo or self.ep.run_silence <= self.TAIL_FRAMES:
                    self.seg.append(frame)
                    text = self.asr.accept_waveform(frame)
                    if ev != "end":
                        out.append(("partial", text))
            else:
                self.preroll.append(frame)

            if ev == "end":
                text = self.asr.finish()
                label = self._label()
                self.asr = None
                self.speaking = False
                self.seg = []
                out.append(("speech", False))
                out.append(("segment", text, label))
        return out

    def finish(self):
        if self.asr is None:
            return None, None
        text = self.asr.finish()
        label = self._label()
        self.asr = None
        self.seg = []
        return text, label


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    loop = asyncio.get_running_loop()
    live = await loop.run_in_executor(None, LiveSession)
    await sock.send_json({"type": "ready"})
    t0 = time.time()
    audio_s = 0.0
    try:
        while True:
            msg = await sock.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                pcm = np.frombuffer(msg["bytes"], dtype=np.int16)
                pcm = pcm.astype(np.float32) / 32768.0   # [-1,1], matches lhotse
                audio_s += len(pcm) / SAMPLE_RATE
                # ONNX inference blocks; keep the event loop responsive
                events = await loop.run_in_executor(None, live.feed, pcm)
                for ev in events:
                    kind = ev[0]
                    if kind == "speech":
                        await sock.send_json({"type": "speech", "active": ev[1]})
                    elif kind == "partial":
                        await sock.send_json({"type": "partial", "text": ev[1],
                                              "audioSeconds": round(audio_s, 2)})
                    elif kind == "segment":
                        msg_out = {"type": "segment", "text": ev[1],
                                   "audioSeconds": round(audio_s, 2)}
                        if ev[2]:
                            msg_out.update(ev[2])   # speaker, score
                        await sock.send_json(msg_out)
            elif msg.get("text") == "stop":
                tail, label = await loop.run_in_executor(None, live.finish)
                dt = time.time() - t0
                out = {"type": "final", "text": tail or "",
                       "audioSeconds": round(audio_s, 2),
                       "rtf": round(dt / audio_s, 3) if audio_s else None}
                if label:
                    out.update(label)
                await sock.send_json(out)
                break
    except WebSocketDisconnect:
        pass
    except Exception as e:                        # surface errors, never hang
        try:
            await sock.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default=DEFAULT_MODEL)
    p.add_argument("--beam", type=int, default=4)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8100)
    p.add_argument("--no-vad", action="store_true",
                   help="disable endpointing; one growing hypothesis")
    p.add_argument("--vad-threshold", type=float, default=0.5)
    p.add_argument("--min-silence-ms", type=int, default=700,
                   help="pause length that ends a sentence")
    p.add_argument("--no-speaker-id", action="store_true",
                   help="disable per-segment speaker labelling")
    p.add_argument("--speaker-threshold", type=float, default=0.5,
                   help="min cosine to accept an enrolled speaker (else 'unknown')")
    a = p.parse_args()

    CFG.update(model_dir=os.path.abspath(a.model_dir), beam=a.beam,
               vad=not a.no_vad, threshold=a.vad_threshold,
               min_silence_ms=a.min_silence_ms)
    if not os.path.isdir(CFG["model_dir"]):
        raise SystemExit(f"Missing model dir: {CFG['model_dir']}")

    spk_on = "off"
    if not a.no_speaker_id and _HAS_SPK and os.path.isfile(ENROLL_JSON):
        enrolled = load_enrolled()
        CFG["spk"] = (SpeakerEmbedder(), enrolled, a.speaker_threshold)
        spk_on = f"{len(enrolled)} enrolled, threshold {a.speaker_threshold}"

    print(f"model : {CFG['model_dir']}")
    print(f"beam  : {a.beam}  (classic beam_search, incremental)")
    print(f"vad   : {'silero, endpoint after %d ms silence' % a.min_silence_ms if CFG['vad'] else 'off'}")
    print(f"spkid : {spk_on}")
    print(f"open  : http://localhost:{a.port}")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
