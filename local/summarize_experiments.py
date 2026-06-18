#!/usr/bin/env python3
import argparse
import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
PARAM_RE = re.compile(r"Number of model parameters:\s*([0-9]+)")


@dataclass
class Experiment:
    name: str
    model: str
    data_variant: str
    mode: str
    epochs: str
    exp_dir: Path
    checkpoint: str


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize experiment params, training time, and WER."
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="TSV file written by local/run_experiment_suite.sh",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Markdown summary path.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Optional CSV summary path. Defaults to OUTPUT with .csv suffix.",
    )
    return parser.parse_args()


def read_experiments(path: Path) -> list[Experiment]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [
            Experiment(
                name=row["name"],
                model=row["model"],
                data_variant=row["data_variant"],
                mode=row["mode"],
                epochs=row["epochs"],
                exp_dir=Path(row["exp_dir"]),
                checkpoint=row.get("checkpoint", ""),
            )
            for row in reader
        ]


def parse_timestamp(line: str) -> datetime | None:
    match = TIMESTAMP_RE.search(line)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "NA"
    seconds = int(round(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def iter_train_logs(exp_dir: Path) -> Iterable[Path]:
    log_dir = exp_dir / "log"
    if not log_dir.is_dir():
        return []
    return sorted(log_dir.glob("log-train-*"), key=lambda p: p.stat().st_mtime)


def summarize_training(exp_dir: Path) -> tuple[str, str]:
    params: str | None = None
    start: datetime | None = None
    done: datetime | None = None

    for log_path in iter_train_logs(exp_dir):
        with log_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if params is None:
                    match = PARAM_RE.search(line)
                    if match:
                        params = f"{int(match.group(1)):,}"
                ts = parse_timestamp(line)
                if ts is None:
                    continue
                if "Training started" in line:
                    start = ts if start is None else min(start, ts)
                if "Done!" in line:
                    done = ts if done is None else max(done, ts)

    duration = None
    if start is not None and done is not None and done >= start:
        duration = (done - start).total_seconds()

    return params or "NA", format_duration(duration)


def parse_wer_file(path: Path) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("settings"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                rows.append((parts[0], float(parts[-1])))
            except ValueError:
                continue
    return rows


def summarize_wers(exp_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(exp_dir.glob("*/wer-summary-test-*.txt")):
        method = path.parent.name
        for setting, wer in parse_wer_file(path):
            rows.append(
                {
                    "decode_method": method,
                    "setting": setting,
                    "wer": f"{wer:.2f}",
                    "wer_file": str(path),
                }
            )
    return rows


def write_outputs(experiments: list[Experiment], output: Path, csv_output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_output.parent.mkdir(parents=True, exist_ok=True)

    table_rows: list[dict[str, str]] = []
    for exp in experiments:
        params, duration = summarize_training(exp.exp_dir)
        wer_rows = summarize_wers(exp.exp_dir)
        if not wer_rows:
            wer_rows = [
                {
                    "decode_method": "NA",
                    "setting": "NA",
                    "wer": "NA",
                    "wer_file": "NA",
                }
            ]

        for wer_row in wer_rows:
            table_rows.append(
                {
                    "experiment": exp.name,
                    "model": exp.model,
                    "data_variant": exp.data_variant,
                    "mode": exp.mode,
                    "epochs": exp.epochs,
                    "params": params,
                    "train_time": duration,
                    "decode_method": wer_row["decode_method"],
                    "wer": wer_row["wer"],
                    "exp_dir": str(exp.exp_dir),
                    "wer_file": wer_row["wer_file"],
                    "checkpoint": exp.checkpoint,
                }
            )

    with csv_output.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(table_rows[0].keys()) if table_rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table_rows)

    with output.open("w", encoding="utf-8") as f:
        print("# Experiment Suite Results", file=f)
        print("", file=f)
        print(f"- Metadata: `{output.parent / 'experiments.tsv'}`", file=f)
        print(f"- CSV: `{csv_output}`", file=f)
        print("", file=f)
        print(
            "| Experiment | Model | Mode | Data | Epochs | Params | Train time | Decode | WER |",
            file=f,
        )
        print("|---|---|---|---|---:|---:|---:|---|---:|", file=f)
        for row in table_rows:
            print(
                "| {experiment} | {model} | {mode} | {data_variant} | {epochs} | "
                "{params} | {train_time} | {decode_method} | {wer} |".format(**row),
                file=f,
            )


def main():
    args = parse_args()
    csv_output = args.csv_output or args.output.with_suffix(".csv")
    experiments = read_experiments(args.metadata)
    write_outputs(experiments, args.output, csv_output)
    print(f"Wrote {args.output}")
    print(f"Wrote {csv_output}")


if __name__ == "__main__":
    main()
