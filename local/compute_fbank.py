from pathlib import Path

from lhotse import RecordingSet, SupervisionSet, CutSet, Fbank, FbankConfig
from lhotse import set_audio_duration_mismatch_tolerance
from lhotse.features import LilcomChunkyWriter

ROOT = Path("./")
MANIFESTS = ROOT / "manifests_fixed"
FBANK_DIR = ROOT / "fbank"

# Có thể bỏ dòng này nếu không thật sự cần siết chặt tolerance
set_audio_duration_mismatch_tolerance(0.5)


def make_fbank(split: str):
    recs = RecordingSet.from_file(MANIFESTS / f"{split}_recordings.jsonl.gz")
    sups = SupervisionSet.from_file(MANIFESTS / f"{split}_supervisions.jsonl.gz")

    cuts = CutSet.from_manifests(recordings=recs, supervisions=sups)
    cuts = cuts.filter(lambda c: 0.3 <= c.duration <= 20.0)

    extractor = Fbank(FbankConfig(num_mel_bins=80))
    storage_path = FBANK_DIR / split
    storage_path.mkdir(parents=True, exist_ok=True)

    cuts = cuts.compute_and_store_features(
        extractor=extractor,
        storage_path=storage_path,
        storage_type=LilcomChunkyWriter,
        num_jobs=8,
    )

    cuts.to_file(FBANK_DIR / f"{split}_cuts.jsonl.gz")
    print(f"{split}: {len(cuts)} cuts")


if __name__ == "__main__":
    for split in ["train", "dev", "test"]:
        make_fbank(split)
