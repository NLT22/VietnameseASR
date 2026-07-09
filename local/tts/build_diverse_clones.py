#!/usr/bin/env python3
"""
Cross-speaker clones with DIVERSE reference voices (incl. female).

The earlier clone set used only the 5 project speakers -- all male -- so the
model generalizes to new male voices (8.9% WER on khanh_toan) but collapses on
female ones (76.5% on yen_nhi). This clones every training sentence into
Gwen-TTS built-in voices that cover the gap.

HELD OUT (never used here, reserved for eval): yen_nhi (F), khanh_toan (M).

Same pipeline as build_crossspeaker.py: reference cleanup -> batched clone ->
loudness normalize -> discard clips >= --max-duration. Resumable.
"""
import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# Recipe root: local/tts/<this file> -> VietnameseASR/
VI = str(Path(__file__).resolve().parents[2])
# Gwen-TTS checkout (supplies ref_info.json + reference wavs). Override with
# GWEN_TTS_DIR; defaults to a sibling of the icefall checkout.
GT = os.environ.get("GWEN_TTS_DIR", str(Path(VI).parents[2] / "gwen-tts"))
REF_TMP = os.path.join(tempfile.gettempdir(), "gwen_refs_diverse")

# 3 female + 1 male. yen_nhi / khanh_toan deliberately excluded (eval set).
TRAIN_VOICES = ["my_van", "ai_vy", "dieu_linh", "tran_lam"]

GEN_CFG = dict(
    temperature=0.3, top_k=20, top_p=0.9, max_new_tokens=4096,
    repetition_penalty=2.0, subtalker_do_sample=True,
    subtalker_temperature=0.1, subtalker_top_k=20, subtalker_top_p=1.0,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source-tsv", default=f"{VI}/transcripts_matched_u20/train.tsv")
    p.add_argument("--corpus-root", default=VI)
    p.add_argument("--out-audio-subdir", default="audio_crossspk_diverse")
    p.add_argument("--out-tsv", default=f"{VI}/transcripts_crossspk_diverse/train.tsv")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--target-rms", type=float, default=0.09)
    p.add_argument("--peak-ceil", type=float, default=0.97)
    p.add_argument("--max-duration", type=float, default=20.0)
    return p.parse_args()


def prepare_ref(src, dst, target_peak=0.95):
    w, sr = sf.read(src)
    if w.ndim > 1:
        w = w.mean(1)
    w = w.astype(np.float32)
    pk = np.abs(w).max()
    if pk > 0:
        thr = 0.02 * pk
        idx = np.where(np.abs(w) > thr)[0]
        if len(idx):
            pad = int(0.05 * sr)
            w = w[max(0, idx[0] - pad): idx[-1] + pad]
        w = w / np.abs(w).max() * target_peak
    sf.write(dst, w, sr, subtype="PCM_16")
    return dst


def loudness_norm(w, target_rms, peak_ceil):
    w = w.astype(np.float32)
    r = np.sqrt((w ** 2).mean())
    if r > 0:
        w = w * (target_rms / r)
    pk = np.abs(w).max()
    if pk > peak_ceil:
        w = w * (peak_ceil / pk)
    return w


def read_tsv(p):
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(p, rows):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["utt_id", "speaker", "audio_path", "text"],
                           delimiter="\t")
        w.writeheader()
        w.writerows(rows)


def main():
    a = parse_args()
    root = Path(a.corpus_root)
    out_root = root / a.out_audio_subdir
    ref_info = json.load(open(f"{GT}/data/ref_info.json"))

    src = read_tsv(a.source_tsv)
    if a.limit:
        src = src[: a.limit]

    todo = []
    for r in src:
        for v in TRAIN_VOICES:
            uid = f"{r['utt_id']}_v{v}"
            todo.append((r, v, uid, out_root / v / f"{uid}.wav"))
    pending = [t for t in todo if not t[3].exists()]
    print(f"sentences={len(src)} voices={TRAIN_VOICES}")
    print(f"clips: {len(todo)} total, {len(pending)} pending")

    if pending:
        os.makedirs(REF_TMP, exist_ok=True)
        try:
            import flash_attn  # noqa: F401
            attn = "flash_attention_2"
        except Exception:
            attn = "sdpa"
        from qwen_tts import Qwen3TTSModel
        model = Qwen3TTSModel.from_pretrained(
            "g-group-ai-lab/gwen-tts-0.6B",
            device_map="cuda:0", dtype=torch.bfloat16, attn_implementation=attn,
        )
        clean = {}
        for v in TRAIN_VOICES:
            info = ref_info[v]
            clean[v] = (prepare_ref(f"{GT}/{info['audio_path']}", f"{REF_TMP}/{v}.wav"),
                        info["text"])

        by_v = {}
        for it in pending:
            by_v.setdefault(it[1], []).append(it)

        skipped = done = 0
        for v, items in by_v.items():
            ra, rt = clean[v]
            for b in range(0, len(items), a.batch_size):
                chunk = items[b : b + a.batch_size]
                texts = [r["text"] for (r, _, _, _) in chunk]
                wavs, sr = model.generate_voice_clone(
                    text=texts if len(texts) > 1 else texts[0],
                    ref_audio=[ra] * len(texts) if len(texts) > 1 else ra,
                    ref_text=[rt] * len(texts) if len(texts) > 1 else rt,
                    **GEN_CFG,
                )
                for (r, _, uid, dst), wav in zip(chunk, wavs):
                    done += 1
                    w = loudness_norm(wav, a.target_rms, a.peak_ceil)
                    if sr != 16000:
                        import librosa
                        w = librosa.resample(w, orig_sr=sr, target_sr=16000)
                    if len(w) / 16000 >= a.max_duration:
                        skipped += 1
                        continue
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(str(dst), w, 16000, subtype="PCM_16")
                if done % 200 < a.batch_size:
                    print(f"  [{done}/{len(pending)}] {v}  skipped_long={skipped}")
        print(f"discarded {skipped} clips >= {a.max_duration}s")

    rows = []
    for r, v, uid, dst in todo:
        if not dst.exists():
            continue
        info = sf.info(str(dst))
        if info.frames / info.samplerate >= a.max_duration:
            continue
        rows.append({"utt_id": uid, "speaker": v,
                     "audio_path": str(dst.relative_to(root)), "text": r["text"]})
    write_tsv(a.out_tsv, rows)
    print(f"wrote {len(rows)} diverse-voice clone rows -> {a.out_tsv}")


if __name__ == "__main__":
    main()
