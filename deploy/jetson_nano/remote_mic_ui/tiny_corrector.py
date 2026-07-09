#!/usr/bin/env python3
import csv
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


SPACE_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)


def clean_text(text):
    text = unicodedata.normalize("NFC", text or "").lower()
    text = PUNCT_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def strip_accents(text):
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("đ", "d")


def char_ngrams(text, sizes=(2, 3, 4)):
    padded = " %s " % text
    feats = Counter()
    for size in sizes:
        if len(padded) < size:
            continue
        for i in range(len(padded) - size + 1):
            gram = padded[i : i + size]
            if gram.strip():
                feats["c%d:%s" % (size, gram)] += 1.0
    return feats


def features(text):
    text = clean_text(text)
    feats = Counter()
    for word in text.split():
        feats["w:%s" % word] += 2.0
    feats.update(char_ngrams(text))

    plain = strip_accents(text)
    if plain != text:
        for key, value in char_ngrams(plain, sizes=(3, 4)).items():
            feats["p:%s" % key] += 0.7 * value
    return feats


def dot(a, b):
    if len(a) > len(b):
        a, b = b, a
    return sum(value * b.get(key, 0.0) for key, value in a.items())


class TinyCorrector:
    def __init__(self, prompts, pairs_path=None):
        self.prompts = list(prompts)
        self.exact = {}
        self.idf = {}
        self.items = []
        self._build()
        if pairs_path:
            self.load_pairs(pairs_path)

    def _build(self):
        docs = []
        df = defaultdict(int)
        for prompt in self.prompts:
            feat = features(prompt.get("text", ""))
            docs.append(feat)
            for key in feat:
                df[key] += 1

        n_docs = max(len(docs), 1)
        self.idf = {
            key: math.log((1.0 + n_docs) / (1.0 + count)) + 1.0
            for key, count in df.items()
        }
        self.items = []
        for prompt, feat in zip(self.prompts, docs):
            vector = self._tfidf(feat)
            norm = math.sqrt(dot(vector, vector)) or 1.0
            self.items.append({"prompt": prompt, "vector": vector, "norm": norm})

    def _tfidf(self, feat):
        return Counter({key: value * self.idf.get(key, 1.0) for key, value in feat.items()})

    def load_pairs(self, pairs_path):
        path = Path(pairs_path)
        if not path.is_file():
            return
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                asr = clean_text(row.get("asr_output", ""))
                corrected = " ".join((row.get("corrected_text") or row.get("target_text") or "").split())
                if asr and corrected:
                    self.exact[asr] = corrected

    def correct(self, text, top_k=5):
        cleaned = clean_text(text)
        if not cleaned:
            return {
                "input": text or "",
                "corrected": "",
                "confidence": 0.0,
                "source": "empty",
                "candidates": [],
            }

        if cleaned in self.exact:
            return {
                "input": text,
                "corrected": self.exact[cleaned],
                "confidence": 1.0,
                "source": "pairs.tsv",
                "candidates": [],
            }

        vector = self._tfidf(features(cleaned))
        norm = math.sqrt(dot(vector, vector)) or 1.0
        scored = []
        for item in self.items:
            score = dot(vector, item["vector"]) / (norm * item["norm"])
            prompt = item["prompt"]
            scored.append(
                {
                    "id": prompt.get("id", ""),
                    "speaker": prompt.get("speaker", ""),
                    "text": prompt.get("text", ""),
                    "score": score,
                }
            )
        scored.sort(key=lambda x: x["score"], reverse=True)
        best = scored[0] if scored else {"text": "", "score": 0.0}
        second_score = scored[1]["score"] if len(scored) > 1 else 0.0
        confidence = max(0.0, min(1.0, best["score"] * 0.75 + (best["score"] - second_score) * 0.75))
        return {
            "input": text,
            "corrected": best["text"],
            "confidence": round(confidence, 4),
            "source": "closed_set_ngram",
            "candidates": [
                dict(candidate, score=round(candidate["score"], 4))
                for candidate in scored[:top_k]
            ],
        }
