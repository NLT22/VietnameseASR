#!/usr/bin/env python3
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_COLLECTION_ROOT = ROOT / "collection"


def get_args():
    parser = argparse.ArgumentParser(
        description="Transcribe collected retraining recordings after collection."
    )
    parser.add_argument("--collection-root", default=str(DEFAULT_COLLECTION_ROOT))
    parser.add_argument("--remote-dir", default=str(ROOT.parent))
    parser.add_argument("--model-dir", default="model_medium_epoch30_avg10")
    parser.add_argument("--asr-mode", choices=["streaming", "nonstream"], default="streaming")
    parser.add_argument("--provider", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-active-paths", type=int, default=20)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_records(collection_root):
    metadata_path = collection_root / "metadata.jsonl"
    if not metadata_path.is_file():
        return []
    records = []
    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("event") == "recorded":
                records.append(item)
    return records


def transcribe(args, wav_path):
    script = "transcribe_streaming_wav.py" if args.asr_mode == "streaming" else "transcribe_wav.py"
    python_bin = Path(args.remote_dir) / ".venv" / "bin" / "python3"
    if not python_bin.is_file():
        python_bin = Path(sys.executable)
    cmd = [
        str(python_bin),
        script,
        "--model-dir",
        args.model_dir,
        "--threads",
        str(args.threads),
        "--provider",
        args.provider,
        "--max-active-paths",
        str(args.max_active_paths),
    ]
    if args.fp32:
        cmd.append("--fp32")
    cmd.append(str(wav_path))
    proc = subprocess.run(
        cmd,
        cwd=args.remote_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=180,
    )
    transcript = proc.stdout.strip().splitlines()[-1].strip() if proc.stdout.strip() else ""
    return proc.returncode, transcript, proc.stdout, proc.stderr


def main():
    args = get_args()
    collection_root = Path(args.collection_root).resolve()
    records = load_records(collection_root)
    if args.limit > 0:
        records = records[: args.limit]

    out_jsonl = collection_root / "metadata_asr.jsonl"
    out_pairs = collection_root / "asr_pairs.tsv"
    if args.dry_run:
        print(f"Would transcribe {len(records)} recordings from {collection_root}")
        for item in records[:5]:
            print(collection_root / item["audio_path"])
        return

    new_pairs = not out_pairs.exists()
    with out_jsonl.open("a", encoding="utf-8") as json_f, out_pairs.open(
        "a", encoding="utf-8", newline=""
    ) as pairs_f:
        writer = csv.writer(pairs_f, delimiter="\t")
        if new_pairs:
            writer.writerow(
                [
                    "asr_output",
                    "target_text",
                    "participant_id",
                    "script_id",
                    "sample_id",
                    "audio_path",
                ]
            )
        for index, item in enumerate(records, start=1):
            wav_path = collection_root / item["audio_path"]
            rc, transcript, stdout, stderr = transcribe(args, wav_path)
            asr_record = {
                **item,
                "asr_output": transcript,
                "asr_returncode": rc,
                "asr_stdout": stdout,
                "asr_stderr": stderr,
                "model_dir": args.model_dir,
                "provider": args.provider,
                "asr_mode": args.asr_mode,
            }
            json_f.write(json.dumps(asr_record, ensure_ascii=False) + "\n")
            if rc == 0:
                writer.writerow(
                    [
                        transcript,
                        item["text"],
                        item["participant_id"],
                        item["script_id"],
                        item["id"],
                        item["audio_path"],
                    ]
                )
            print(f"[{index}/{len(records)}] rc={rc} {item['participant_id']} {item['script_id']}")


if __name__ == "__main__":
    main()
