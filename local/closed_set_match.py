#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import math
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path


RECOG_RE = re.compile(r"^(.*):\s+(ref|hyp)=(\[.*\])$")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map ASR hypotheses to the nearest sentence in a closed catalog."
    )
    parser.add_argument("--recogs", type=Path, required=True)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("closed_set/transcripts/train.tsv"),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def normalize_words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def read_catalog(path: Path) -> list[tuple[str, str, list[str]]]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return [(r["utt_id"], r["text"], normalize_words(r["text"])) for r in rows]


def read_recogs(path: Path) -> dict[str, dict[str, list[str]]]:
    results: dict[str, dict[str, list[str]]] = {}
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            match = RECOG_RE.match(line.strip())
            if not match:
                continue
            cut_id, kind, value = match.groups()
            words = ast.literal_eval(value)
            results.setdefault(cut_id, {})[kind] = [str(w).lower() for w in words]
    return results


def make_idf(catalog: list[tuple[str, str, list[str]]]) -> dict[str, float]:
    document_frequency = Counter()
    for _, _, words in catalog:
        document_frequency.update(set(words))
    n = len(catalog)
    return {
        word: math.log((n + 1) / (frequency + 1)) + 1.0
        for word, frequency in document_frequency.items()
    }


def score_candidate(hyp: list[str], candidate: list[str], idf: dict[str, float]) -> float:
    if not hyp:
        return 0.0

    matcher = SequenceMatcher(a=hyp, b=candidate, autojunk=False)
    longest = matcher.find_longest_match().size / len(hyp)
    sequence_ratio = matcher.ratio()

    candidate_counts = Counter(candidate)
    matched_weight = 0.0
    total_weight = 0.0
    for word, count in Counter(hyp).items():
        weight = idf.get(word, max(idf.values(), default=1.0))
        total_weight += count * weight
        matched_weight += min(count, candidate_counts[word]) * weight
    weighted_precision = matched_weight / total_weight if total_weight else 0.0

    return 0.55 * longest + 0.30 * weighted_precision + 0.15 * sequence_ratio


def main() -> None:
    args = get_args()
    catalog = read_catalog(args.catalog)
    idf = make_idf(catalog)
    recogs = read_recogs(args.recogs)
    output = args.output or args.recogs.with_name(f"closed-set-{args.recogs.name}")

    correct = 0
    evaluated = 0
    with output.open("w", encoding="utf-8") as f:
        for cut_id, result in sorted(recogs.items()):
            hyp = result.get("hyp", [])
            ranked = sorted(
                (
                    (score_candidate(hyp, words, idf), utt_id, text)
                    for utt_id, text, words in catalog
                ),
                reverse=True,
            )
            best_score, best_id, best_text = ranked[0]
            expected_id = cut_id.rsplit("-", 1)[0]
            is_correct = expected_id == best_id
            correct += int(is_correct)
            evaluated += 1
            print(
                f"{cut_id}\tmatch={best_id}\tscore={best_score:.4f}\t"
                f"correct={int(is_correct)}\t{best_text}",
                file=f,
            )
            for score, utt_id, text in ranked[1 : args.top_k]:
                print(f"\talt={utt_id}\tscore={score:.4f}\t{text}", file=f)

    accuracy = correct / evaluated if evaluated else 0.0
    print(f"Closed-set sentence accuracy: {correct}/{evaluated} ({accuracy:.2%})")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
