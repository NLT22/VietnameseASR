#!/usr/bin/env python3
"""
Build a HELD-OUT SPEAKER eval set.

The project's WER numbers are measured on the same 618 recordings used for
training ("hoc vet"), so they cannot see speaker generalization at all -- which
is what actually matters when someone speaks into the mic.

This synthesizes KNOWN sentences in voices that appear NOWHERE in training:
Gwen-TTS built-in speakers (training clones only ever used the 5 project
speakers as references). Same sentences the models memorized, new voices.

Caveat: these are TTS voices from the same engine used to make the training
clones, so the clone-trained model has a mild same-vocoder advantage. A real
human recording is still the gold standard.
"""
import csv
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

# Recipe root: local/tts/<this file> -> VietnameseASR/
VI = str(Path(__file__).resolve().parents[2])
# Gwen-TTS checkout (ref_info.json + reference wavs). Override with GWEN_TTS_DIR.
GT = os.environ.get("GWEN_TTS_DIR", str(Path(VI).parents[2] / "gwen-tts"))
OUT = f"{VI}/heldout_speaker_eval"
N_SENTS = 25

# Unseen voices: never used as a clone reference during training.
VOICES = ["yen_nhi", "khanh_toan"]

GEN_CFG = dict(
    temperature=0.3, top_k=20, top_p=0.9, max_new_tokens=4096,
    repetition_penalty=2.0, subtalker_do_sample=True,
    subtalker_temperature=0.1, subtalker_top_k=20, subtalker_top_p=1.0,
)
TARGET_RMS = 0.09
PEAK_CEIL = 0.97


def loudness_norm(w):
    w = w.astype(np.float32)
    r = np.sqrt((w ** 2).mean())
    if r > 0:
        w = w * (TARGET_RMS / r)
    pk = np.abs(w).max()
    if pk > PEAK_CEIL:
        w = w * (PEAK_CEIL / pk)
    return w


def main():
    import json
    os.makedirs(OUT, exist_ok=True)
    ref_info = json.load(open(f"{GT}/data/ref_info.json"))

    rows = list(csv.DictReader(open(f"{VI}/transcripts_matched_u20/test.tsv"), delimiter="\t"))
    # shortest sentences first: easiest, isolates speaker effect from length
    rows.sort(key=lambda r: len(r["text"].split()))
    picked = rows[:N_SENTS]
    print(f"{len(picked)} sentences, {len(picked[0]['text'].split())}-"
          f"{len(picked[-1]['text'].split())} words")

    try:
        import flash_attn  # noqa: F401
        attn = "flash_attention_2"
    except Exception:
        attn = "sdpa"
    model = Qwen3TTSModel.from_pretrained(
        "g-group-ai-lab/gwen-tts-0.6B",
        device_map="cuda:0", dtype=torch.bfloat16, attn_implementation=attn,
    )

    manifest = []
    for voice in VOICES:
        info = ref_info[voice]
        ref_audio = f"{GT}/{info['audio_path']}"
        ref_text = info["text"]
        for r in picked:
            wavs, sr = model.generate_voice_clone(
                text=r["text"], ref_audio=ref_audio, ref_text=ref_text, **GEN_CFG,
            )
            w = loudness_norm(wavs[0])
            name = f"{voice}__{r['utt_id']}"
            # 16 kHz mono for the ASR frontend
            if sr != 16000:
                import librosa
                w = librosa.resample(w, orig_sr=sr, target_sr=16000)
            sf.write(f"{OUT}/{name}.wav", w, 16000, subtype="PCM_16")
            manifest.append({"utt_id": name, "voice": voice,
                             "src_utt": r["utt_id"], "text": r["text"]})
        print(f"  {voice}: {len(picked)} clips")

    with open(f"{OUT}/manifest.tsv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["utt_id", "voice", "src_utt", "text"], delimiter="\t")
        w.writeheader()
        w.writerows(manifest)
    print(f"wrote {len(manifest)} clips -> {OUT}/")


if __name__ == "__main__":
    main()
