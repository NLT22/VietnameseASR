# prepare_manifests.py
from pathlib import Path
import csv
from lhotse import Recording, RecordingSet, SupervisionSegment, SupervisionSet

ROOT = Path("./")
OUT = ROOT / "manifests"
OUT.mkdir(parents=True, exist_ok=True)

def prepare_split(split: str):
    tsv_path = ROOT / "transcripts" / f"{split}.tsv"
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

    recs.to_file(OUT / f"{split}_recordings.jsonl.gz")
    sups.to_file(OUT / f"{split}_supervisions.jsonl.gz")

    print(f"{split}: recordings={len(recs)}, supervisions={len(sups)}")

if __name__ == "__main__":
    for split in ["train", "dev", "test"]:
        prepare_split(split)
