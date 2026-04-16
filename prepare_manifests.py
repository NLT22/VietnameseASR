import csv
import argparse
from pathlib import Path

from lhotse import Recording, RecordingSet, SupervisionSegment, SupervisionSet

ROOT = Path(__file__).resolve().parent


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/manifests"),
        help="Directory to write train/dev/test Lhotse manifests.",
    )
    parser.add_argument(
        "--transcript-dir",
        type=Path,
        default=Path("transcripts"),
        help="Directory containing train/dev/test TSV transcripts.",
    )
    return parser.parse_args()


def prepare_split(split: str, transcript_dir: Path, output_dir: Path):
    tsv_path = ROOT / transcript_dir / f"{split}.tsv"
    recordings = []
    supervisions = []

    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            utt_id = row["utt_id"].strip()
            speaker = row["speaker"].strip()
            audio_path = (ROOT / row["audio_path"]).resolve()
            text = " ".join(row["text"].strip().split())

            recording = Recording.from_file(audio_path, recording_id=utt_id)
            recordings.append(recording)

            supervision = SupervisionSegment(
                id=utt_id,
                recording_id=utt_id,
                start=0.0,
                duration=recording.duration,
                text=text,
                speaker=speaker,
                language="vi",
            )
            supervisions.append(supervision)

    recs = RecordingSet.from_recordings(recordings)
    sups = SupervisionSet.from_segments(supervisions)

    recs.to_file(output_dir / f"{split}_recordings.jsonl.gz")
    sups.to_file(output_dir / f"{split}_supervisions.jsonl.gz")

    print(f"{split}: recordings={len(recs)}, supervisions={len(sups)}")

if __name__ == "__main__":
    args = get_args()
    args.transcript_dir = args.transcript_dir if args.transcript_dir.is_absolute() else ROOT / args.transcript_dir
    args.output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "dev", "test"]:
        prepare_split(split, args.transcript_dir, args.output_dir)
