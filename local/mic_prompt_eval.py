#!/usr/bin/env python3
"""Prompted microphone test for inspecting ASR hypotheses.

This script shows dataset sentences/phrases to read aloud, records each attempt
with Silero VAD, decodes with the same JIT path used by mic_streaming_asr.py, and
writes the model hypotheses to TSV/Markdown for later inspection.
"""

from __future__ import annotations

import argparse
import csv
import math
import queue
import re
import shutil
import sys
from collections import Counter, OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import k2
import sounddevice as sd
import soundfile as sf
import torch
from silero_vad import load_silero_vad
from torch.nn.utils.rnn import pad_sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mic_streaming_asr import (  # noqa: E402
    BEAM_SIZE,
    FULL_MODEL,
    MIN_SPEECH_CHUNKS,
    SAMPLE_RATE,
    SILENCE_CHUNKS_TO_END,
    TOKENS_PATH,
    VAD_CHUNK,
    VAD_THRESHOLD_ON,
    compute_fbank,
    jit_beam_search,
    jit_greedy_search,
    token_ids_to_text,
)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--transcript", default="transcripts/train.tsv")
    parser.add_argument("--model", default=FULL_MODEL)
    parser.add_argument("--tokens", default=TOKENS_PATH)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Delete old output under mic_prompt_eval/ before starting.",
    )
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--mode", choices=["sentences", "phrases", "both"], default="both")
    parser.add_argument("--phrase-len", type=int, default=5)
    parser.add_argument("--max-phrases", type=int, default=12)
    parser.add_argument("--decode-method", choices=["beam", "greedy"], default="beam")
    parser.add_argument("--beam", type=int, default=BEAM_SIZE)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--max-record-seconds", type=float, default=20.0)
    parser.add_argument("--vad-threshold", type=float, default=VAD_THRESHOLD_ON)
    parser.add_argument("--silence-chunks", type=int, default=SILENCE_CHUNKS_TO_END)
    parser.add_argument("--min-speech-chunks", type=int, default=MIN_SPEECH_CHUNKS)
    parser.add_argument(
        "--no-save-audio",
        action="store_true",
        help="Do not save recorded wav files.",
    )
    return parser.parse_args()


def unique_texts(transcript: Path) -> List[str]:
    texts: "OrderedDict[str, None]" = OrderedDict()
    with transcript.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            text = " ".join(row["text"].split())
            if text:
                texts.setdefault(text, None)
    return list(texts)


def extract_phrases(texts: Iterable[str], phrase_len: int, max_phrases: int) -> List[str]:
    counts: Counter[str] = Counter()
    for text in texts:
        words = text.split()
        for i in range(0, max(0, len(words) - phrase_len + 1)):
            counts[" ".join(words[i : i + phrase_len])] += 1

    phrases = []
    seen = set()
    for phrase, _ in counts.most_common():
        normalized = phrase.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        phrases.append(phrase)
        if len(phrases) >= max_phrases:
            break
    return phrases


def slugify(text: str, max_len: int = 48) -> str:
    text = re.sub(r"\s+", "_", text.strip().lower())
    text = re.sub(r"[^0-9a-zA-Z_]+", "", text)
    return text[:max_len].strip("_") or "prompt"


def wait_enter(message: str) -> None:
    try:
        input(message)
    except EOFError:
        pass


def record_once(
    audio_queue: "queue.Queue",
    vad_model,
    args: argparse.Namespace,
) -> torch.Tensor:
    print("Recording starts when speech is detected. Stop by staying silent.")
    sample_buf = torch.zeros(0)
    raw_buf = torch.zeros(0)
    pre_raw: List[torch.Tensor] = []
    utterance_active = False
    speech_count = 0
    silence_count = 0
    max_samples = int(args.max_record_seconds * SAMPLE_RATE)

    while True:
        try:
            chunk_np = audio_queue.get(timeout=5.0)
        except queue.Empty:
            print("No microphone audio received yet...")
            continue

        chunk = torch.from_numpy(chunk_np)
        sample_buf = torch.cat([sample_buf, chunk])

        while sample_buf.size(0) >= VAD_CHUNK:
            window = sample_buf[:VAD_CHUNK]
            sample_buf = sample_buf[VAD_CHUNK:]

            vad_prob = vad_model(window, SAMPLE_RATE).item()
            is_speech = vad_prob >= args.vad_threshold

            if is_speech:
                silence_count = 0
                speech_count += 1
                pre_raw.append(window)
                if len(pre_raw) > args.min_speech_chunks:
                    pre_raw.pop(0)

                if not utterance_active and speech_count >= args.min_speech_chunks:
                    utterance_active = True
                    raw_buf = torch.cat(pre_raw)
                    pre_raw = []
                    print("REC...")
                elif utterance_active:
                    raw_buf = torch.cat([raw_buf, window])
            else:
                speech_count = 0
                pre_raw = []
                if utterance_active:
                    silence_count += 1
                    raw_buf = torch.cat([raw_buf, window])

            if utterance_active and raw_buf.numel() >= max_samples:
                print("Reached max record length.")
                return raw_buf

            if utterance_active and silence_count >= args.silence_chunks:
                return raw_buf


@torch.no_grad()
def decode_wave(
    model,
    wave: torch.Tensor,
    token_table,
    args: argparse.Namespace,
    device: torch.device,
) -> str:
    tail_padding = torch.zeros(int(0.3 * SAMPLE_RATE), device=device)
    feats = compute_fbank(torch.cat([wave.to(device), tail_padding]), device)
    features = pad_sequence(
        [feats],
        batch_first=True,
        padding_value=math.log(1e-10),
    )
    feature_lens = torch.tensor([feats.size(0)], device=device)
    encoder_out, encoder_out_lens = model.encoder(
        features=features,
        feature_lengths=feature_lens,
    )
    if args.decode_method == "greedy":
        token_ids = jit_greedy_search(model, encoder_out, encoder_out_lens, device)
    else:
        token_ids = jit_beam_search(model, encoder_out, args.beam, device)
    return token_ids_to_text(token_ids, token_table)


def write_outputs(output_dir: Path, rows: List[Dict[str, str]]) -> None:
    tsv_path = output_dir / "results.tsv"
    md_path = output_dir / "summary.md"

    with tsv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "timestamp",
            "prompt_type",
            "prompt_index",
            "repeat_index",
            "prompt",
            "hypothesis",
            "audio_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    grouped: Dict[Tuple[str, str], List[Dict[str, str]]] = OrderedDict()
    for row in rows:
        grouped.setdefault((row["prompt_type"], row["prompt"]), []).append(row)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Mic Prompt Evaluation\n\n")
        f.write(f"- Created: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"- Total recordings: {len(rows)}\n")
        f.write(f"- Results TSV: `{tsv_path.name}`\n\n")
        for (prompt_type, prompt), items in grouped.items():
            f.write(f"## {prompt_type}: {prompt}\n\n")
            for item in items:
                f.write(
                    f"- Repeat {item['repeat_index']}: `{item['hypothesis']}`"
                    f" (`{item['audio_path']}`)\n"
                )
                f.write("\n")


def clear_old_output(output_dir: Path) -> None:
    root = (ROOT / "mic_prompt_eval").resolve()
    target = output_dir.resolve()
    if not target.exists():
        return
    if target == root:
        for child in target.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        return
    if root not in target.parents:
        raise ValueError(f"--clear-output can only delete paths under {root}: {target}")
    shutil.rmtree(target)


def main() -> None:
    args = get_args()
    transcript = ROOT / args.transcript
    model_path = ROOT / args.model
    tokens = ROOT / args.tokens
    output_dir = (
        ROOT
        / (
            args.output_dir
            or f"mic_prompt_eval/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    )
    if args.clear_output:
        clear_old_output(output_dir)

    audio_dir = output_dir / "audio"
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    texts = unique_texts(transcript)
    if not texts:
        raise SystemExit(f"No texts found in {transcript}")

    prompts: List[Tuple[str, str]] = []
    if args.mode in ("sentences", "both"):
        prompts.extend(("sentence", text) for text in texts)
    if args.mode in ("phrases", "both"):
        prompts.extend(
            ("phrase", phrase)
            for phrase in extract_phrases(texts, args.phrase_len, args.max_phrases)
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {model_path.relative_to(ROOT)}")
    print(f"Output: {output_dir.relative_to(ROOT)}")
    print(f"Prompts: {len(prompts)}, repeat={args.repeat}")

    model = torch.jit.load(str(model_path), map_location=device)
    model.eval().to(device)
    token_table = k2.SymbolTable.from_file(str(tokens))

    vad_model = load_silero_vad()
    vad_model.eval()

    if args.device is not None:
        sd.default.device = args.device
    dev_info = sd.query_devices(kind="input")
    print(f"Mic: [{dev_info['index']}] {dev_info['name']}")

    audio_queue: "queue.Queue" = queue.Queue()

    def mic_callback(indata, frames, time_info, status):
        if status:
            print(f"[mic] {status}", file=sys.stderr)
        audio_queue.put(indata[:, 0].copy())

    rows: List[Dict[str, str]] = []
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=VAD_CHUNK,
        device=args.device,
        callback=mic_callback,
    ):
        for prompt_index, (prompt_type, prompt) in enumerate(prompts, start=1):
            print("\n" + "=" * 80)
            print(f"{prompt_type.upper()} {prompt_index}/{len(prompts)}")
            print(prompt)
            print("=" * 80)
            for repeat_index in range(1, args.repeat + 1):
                print(f"\nRepeat {repeat_index}/{args.repeat}")
                wait_enter("Press Enter, then read the prompt aloud...")
                wave = record_once(audio_queue, vad_model, args)
                hyp = decode_wave(model, wave, token_table, args, device)

                audio_rel = ""
                if not args.no_save_audio:
                    audio_name = (
                        f"{prompt_index:03d}_{repeat_index:02d}_"
                        f"{prompt_type}_{slugify(prompt)}.wav"
                    )
                    audio_path = audio_dir / audio_name
                    sf.write(str(audio_path), wave.cpu().numpy(), SAMPLE_RATE)
                    audio_rel = audio_path.relative_to(output_dir).as_posix()

                row = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "prompt_type": prompt_type,
                    "prompt_index": str(prompt_index),
                    "repeat_index": str(repeat_index),
                    "prompt": prompt,
                    "hypothesis": hyp,
                    "audio_path": audio_rel,
                }
                rows.append(row)
                write_outputs(output_dir, rows)
                print(f"HYP: {hyp}")
                print(f"Saved progress to: {output_dir.relative_to(ROOT)}")

    write_outputs(output_dir, rows)
    print(f"\nDone. Results: {output_dir / 'results.tsv'}")
    print(f"Summary: {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
