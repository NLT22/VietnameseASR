#!/usr/bin/env python3
"""
Cross-speaker augmentation for VietnameseASR via Gwen-TTS.

Each source sentence is originally spoken by ONE person. This generates the
SAME sentence in the OTHER speakers' voices, so every sentence gets covered by
every speaker. Real originals are kept; synthetic cross-speaker copies are
added.

Pipeline per clip:
  reference cleanup (trim silence + peak-normalize)  ->  clone  ->  loudness
  normalize (target RMS + peak ceiling).

Resumable: existing output wavs are skipped, so a long run can be interrupted
and restarted. Output TSV = original rows + cross-speaker rows, ready to drop
in as a training transcript (prepare_manifests.py resolves audio_path against
--corpus-root).

Usage:
  python build_crossspeaker.py --limit 3            # smoke test
  python build_crossspeaker.py                      # full run, all 5 speakers
  python build_crossspeaker.py --speakers Dung,Khoi,Quan,Trung   # only original 4
"""
import argparse
import csv
import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# Recipe root: local/tts/<this file> -> VietnameseASR/
VIET = str(Path(__file__).resolve().parents[2])
VC = os.path.join(VIET, "datasets", "vi_asr_corpus")
REF_TMP = os.path.join(tempfile.gettempdir(), "gwen_refs")

# One cleaned reference per speaker (vi_asr_corpus recordings — cleaner than
# VietnameseASR, esp. Dung). (audio_path, exact_ref_text).
REFS = {
    "Khoi": (f"{VC}/audio/train/Khoi/Khoi_000039.wav",
             "tôi ghi âm dữ liệu thử nghiệm vào lúc tám giờ ba mươi phút trong phòng "
             "yên tĩnh để kiểm tra hệ thống nhận dạng tiếng nói hôm nay"),
    "Quan": (f"{VC}/audio/train/Quan/Quan_000013.wav",
             "hôm nay tôi kiểm tra hệ thống nhận dạng tiếng nói bằng dữ liệu thử nghiệm "
             "và ghi âm trong phòng yên tĩnh vào lúc tám giờ ba mươi phút"),
    "Dung": (f"{VC}/audio/train/Dung/Dung_000002.wav",
             "hôm nay tôi kiểm tra hệ thống nhận dạng tiếng nói bằng dữ liệu thử nghiệm "
             "và ghi âm trong phòng yên tĩnh vào lúc tám giờ ba mươi phút"),
    "Trung": (f"{VC}/audio/train/Trung/Trung_000014.wav",
              "tôi ghi âm dữ liệu thử nghiệm vào lúc tám giờ ba mươi phút trong phòng "
              "yên tĩnh để kiểm tra hệ thống nhận dạng tiếng nói hôm nay"),
    "Hieu": (f"{VC}/audio/train/HIEU/HIEU_000013.wav",
             "hôm nay tôi kiểm tra hệ thống nhận dạng tiếng nói bằng dữ liệu thử nghiệm "
             "và ghi âm trong phòng yên tĩnh vào lúc tám giờ ba mươi phút"),
}

GEN_CFG = dict(
    temperature=0.3, top_k=20, top_p=0.9, max_new_tokens=4096,
    repetition_penalty=2.0, subtalker_do_sample=True,
    subtalker_temperature=0.1, subtalker_top_k=20, subtalker_top_p=1.0,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source-tsv", default=f"{VIET}/transcripts_matched_u20/train.tsv",
                   help="Transcript to augment (utt_id\\tspeaker\\taudio_path\\ttext)")
    p.add_argument("--corpus-root", default=VIET,
                   help="Root that output audio_path is relative to")
    p.add_argument("--out-audio-subdir", default="audio_crossspk")
    p.add_argument("--out-tsv", default=f"{VIET}/transcripts_crossspk/train.tsv")
    p.add_argument("--speakers", default="Dung,Khoi,Quan,Trung,Hieu",
                   help="Target voices to clone into (default: all 5, incl. Hieu)")
    p.add_argument("--limit", type=int, default=0,
                   help="Only process first N source rows (0 = all)")
    p.add_argument("--target-rms", type=float, default=0.09)
    p.add_argument("--peak-ceil", type=float, default=0.97)
    p.add_argument("--max-duration", type=float, default=20.0,
                   help="Discard clones >= this many seconds (long audio hurt "
                        "training in earlier experiments)")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Clips generated per batched forward pass (per speaker). "
                        "Higher = faster but more VRAM; 1 = one-at-a-time.")
    return p.parse_args()


def prepare_ref(src, dst, target_peak=0.95):
    w, sr = sf.read(src)
    if w.ndim > 1:
        w = w.mean(1)
    w = w.astype(np.float32)
    peak = np.abs(w).max()
    if peak > 0:
        thr = 0.02 * peak
        idx = np.where(np.abs(w) > thr)[0]
        if len(idx):
            pad = int(0.05 * sr)
            w = w[max(0, idx[0] - pad): idx[-1] + pad]
        w = w / np.abs(w).max() * target_peak
    sf.write(dst, w, sr, subtype="PCM_16")
    return dst


def loudness_norm(w, target_rms, peak_ceil):
    w = w.astype(np.float32)
    rms = np.sqrt((w ** 2).mean())
    if rms > 0:
        w = w * (target_rms / rms)
    pk = np.abs(w).max()
    if pk > peak_ceil:
        w = w * (peak_ceil / pk)
    return w


def read_tsv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["utt_id", "speaker", "audio_path", "text"],
                           delimiter="\t")
        w.writeheader()
        w.writerows(rows)


def main():
    args = parse_args()
    speakers = [s.strip() for s in args.speakers.split(",") if s.strip()]
    for s in speakers:
        if s not in REFS:
            raise SystemExit(f"No reference configured for speaker '{s}'")

    corpus_root = Path(args.corpus_root)
    out_audio_root = corpus_root / args.out_audio_subdir
    src_rows = read_tsv(args.source_tsv)
    if args.limit:
        src_rows = src_rows[:args.limit]

    # Count work up front.
    todo = []
    for r in src_rows:
        for tgt in speakers:
            if tgt == r["speaker"]:
                continue  # real original already exists for this voice
            new_utt = f"{r['utt_id']}_x{tgt}"
            dst = out_audio_root / tgt / f"{new_utt}.wav"
            todo.append((r, tgt, new_utt, dst))
    total = len(todo)
    pending = [t for t in todo if not t[3].exists()]
    print(f"source rows: {len(src_rows)} | target voices: {speakers}")
    print(f"cross-speaker clips: {total} total, {len(pending)} pending "
          f"(resuming, {total - len(pending)} already done)")
    if not pending:
        print("Nothing to generate.")
    else:
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
        clean_refs = {s: prepare_ref(REFS[s][0], f"{REF_TMP}/{s}_ref_clean.wav")
                      for s in speakers}

        # Batch per speaker: all clips for a speaker share one reference, so a
        # batch is text=[...] with the same ref repeated. Bigger batches use the
        # GPU better; --batch-size 1 reproduces the original one-at-a-time path.
        by_spk = {}
        for item in pending:
            by_spk.setdefault(item[1], []).append(item)

        skipped_long = 0
        done = 0
        for tgt, items in by_spk.items():
            ref_a, ref_t = clean_refs[tgt], REFS[tgt][1]
            for b in range(0, len(items), args.batch_size):
                chunk = items[b:b + args.batch_size]
                texts = [r["text"] for (r, _, _, _) in chunk]
                wavs, sr = model.generate_voice_clone(
                    text=texts if len(texts) > 1 else texts[0],
                    ref_audio=[ref_a] * len(texts) if len(texts) > 1 else ref_a,
                    ref_text=[ref_t] * len(texts) if len(texts) > 1 else ref_t,
                    **GEN_CFG,
                )
                kept = 0
                for (r, _, new_utt, dst), wav in zip(chunk, wavs):
                    done += 1
                    w = loudness_norm(wav, args.target_rms, args.peak_ceil)
                    dur = len(w) / sr
                    if dur >= args.max_duration:
                        skipped_long += 1
                        continue
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(str(dst), w, sr, subtype="PCM_16")
                    kept += 1
                print(f"  [{done}/{len(pending)}] {tgt} batch of {len(chunk)} (kept {kept})")
        print(f"Discarded {skipped_long} clones >= {args.max_duration}s")

    # (Re)build combined TSV: originals + every cross-speaker clip that exists
    # and is under the duration limit (guards against long files from older runs).
    rows = list(src_rows)
    excluded_long = 0
    for r, tgt, new_utt, dst in todo:
        if not dst.exists():
            continue
        info = sf.info(str(dst))
        if info.frames / info.samplerate >= args.max_duration:
            excluded_long += 1
            continue
        rows.append({
            "utt_id": new_utt,
            "speaker": tgt,
            "audio_path": str(dst.relative_to(corpus_root)),
            "text": r["text"],
        })
    if excluded_long:
        print(f"Excluded {excluded_long} existing over-length clips from TSV")
    write_tsv(args.out_tsv, rows)
    print(f"Wrote {len(rows)} rows ({len(rows) - len(src_rows)} cross-speaker) "
          f"-> {args.out_tsv}")


if __name__ == "__main__":
    main()
