#!/usr/bin/env python3
import argparse
import datetime as dt
import hashlib
import json
import mimetypes
import os
import random
import shutil
import ssl
import subprocess
import sys
import time
import csv
import posixpath
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from urllib.parse import parse_qs, urlparse

from tiny_corrector import TinyCorrector


ROOT = Path(__file__).resolve().parent
RECIPE_ROOT = ROOT.parents[2]
TMP_ROOT = Path("/tmp/vietasr_remote_mic_ui")
DEFAULT_COLLECTION_DIR = ROOT / "collected"
RECORDING_COLLECTION_ROOT = ROOT / "collection"
AUG_COMPARISON_ROOT = ROOT / "aug_comparison"
EXAMPLE_IDS = ["Dung_000086", "Khoi_000099", "Trung_000086"]
DEFAULT_COLLECTION_PARTICIPANTS = ["Trung", "Khoi", "Dung", "Hieu", "Quan"]
COLLECTION_SCRIPT_COUNT = 30
COLLECTION_MIN_WORDS = 25
COLLECTION_MAX_WORDS = 35
MODEL_CHOICES = {
    # asrMode "beam" = custom onnxruntime classic beam_search.
    # asrMode "streaming"/"nonstream" = sherpa-onnx modified_beam_search.
    #
    # Default. First model trained AFTER the off-by-one label fix -- every other
    # model here learned the WRONG text for all Trung/Dung recordings -- and the
    # first with female voices in training. Held-out speakers, classic beam_search:
    #   divmix_x8   1.38% (M) / 16.96% (F) /  9.17% overall
    #   mix        49.48%     / 92.04%     / 70.76%
    #   deployed   82.70%     / 99.31%     / 91.00%
    # Caveat: trained on 4-5s sentences; utterances >15s are out of distribution
    # (12,800 cuts, only 72 >= 15s) and can decode to an empty string.
    "divmix-x8-beam": {
        "label": "divmix_x8 (corrected labels, 9 voices) + beam_search  [best]",
        "modelDir": "model_divmix_x8_epoch60_avg10",
        "asrMode": "beam",
    },
    "mix-beam": {
        "label": "Mix (real+TTS clones) + beam_search  [pre-fix labels]",
        "modelDir": "model_realdom_mix_epoch60_avg10",
        "asrMode": "beam",
    },
    "deployed-beam": {
        "label": "Deployed (real only) + beam_search  [pre-fix labels]",
        "modelDir": "model_streaming_u20_epoch55_avg10",
        "asrMode": "beam",
    },
    "streaming-u20": {
        "label": "Deployed + sherpa-onnx modified_beam  [pre-fix labels]",
        "modelDir": "model_streaming_u20_epoch55_avg10",
        "asrMode": "streaming",
    },
    "nonstream-scratch50": {
        "label": "Non-streaming scratch50 (old) + sherpa-onnx",
        "modelDir": "model_nonstream_scratch50_best",
        "asrMode": "nonstream",
    },
}
DEFAULT_MODEL_CHOICE = "divmix-x8-beam"
TINY_CORRECTOR = None


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def get_args():
    parser = argparse.ArgumentParser(description="Remote microphone UI for Jetson ASR.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--jetson-host", default="thayhoang@192.168.1.50")
    parser.add_argument(
        "--ssh-key",
        default="/media/pc/cde52536-a42f-46bb-b753-bbd8c184686f/home/trung/.ssh/id_ed25519",
    )
    parser.add_argument(
        "--remote-dir",
        default="/home/thayhoang/vietnamese_asr_eval/jetson_nano",
    )
    parser.add_argument("--model-dir", default="model_streaming_u20_epoch55_avg10")
    parser.add_argument(
        "--asr-mode",
        choices=["streaming", "nonstream", "beam"],
        default="streaming",
        help="Backend transcriber. 'beam' = custom onnxruntime classic "
             "beam_search (~2.4% WER); streaming/nonstream use sherpa-onnx (~9.2%).",
    )
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--provider", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--max-active-paths", type=int, default=20)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument(
        "--local-mode",
        action="store_true",
        help="Run transcribe_wav.py locally instead of using SSH/SCP.",
    )
    parser.add_argument("--certfile", default="")
    parser.add_argument("--keyfile", default="")
    parser.add_argument(
        "--collection-dir",
        default=str(DEFAULT_COLLECTION_DIR),
        help="Directory for collected wavs, metadata.jsonl, and pairs.tsv.",
    )
    return parser.parse_args()


def run_cmd(cmd, timeout=120):
    started = time.time()
    proc = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    elapsed = time.time() - started
    return proc.returncode, proc.stdout, proc.stderr, elapsed


def shell_quote(value):
    return "'" + value.replace("'", "'\"'\"'") + "'"


def time_ns():
    return int(time.time() * 1000000000)


def resolve_model(query, app_args):
    choice = query.get("model", [DEFAULT_MODEL_CHOICE])[0]
    if choice in MODEL_CHOICES:
        config = MODEL_CHOICES[choice]
        return {
            "choice": choice,
            "label": config["label"],
            "model_dir": config["modelDir"],
            "asr_mode": config["asrMode"],
        }

    return {
        "choice": "custom",
        "label": app_args.model_dir,
        "model_dir": app_args.model_dir,
        "asr_mode": app_args.asr_mode,
    }


def load_prompt_rows():
    # transcripts_matched_u20 is the canonical corrected split. The older dirs
    # predate the off-by-one label fix; snapping to them makes the corrector
    # "fix" a transcript into a sentence the model never trained on.
    for transcript in (
        RECIPE_ROOT / "transcripts_matched_u20" / "test.tsv",
        RECIPE_ROOT / "transcripts_matched" / "test.tsv",
        RECIPE_ROOT / "transcripts" / "test.tsv",
        ROOT / "examples" / "examples.tsv",
    ):
        if not transcript.is_file():
            continue
        prompts = []
        seen = set()
        with transcript.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                text = " ".join(row.get("text", "").split())
                if not text or text in seen:
                    continue
                seen.add(text)
                prompt_id = row.get("utt_id") or f"prompt_{len(prompts) + 1:04d}"
                prompts.append(
                    {
                        "id": prompt_id,
                        "speaker": row.get("speaker", ""),
                        "text": text,
                    }
                )
        if prompts:
            return prompts
    return [
        {"id": item["id"], "speaker": item.get("speaker", ""), "text": item["text"]}
        for item in load_examples()
    ]


def get_tiny_corrector(args):
    global TINY_CORRECTOR
    if TINY_CORRECTOR is None:
        root, wav_dir, metadata_path, pairs_path = collection_paths(args)
        TINY_CORRECTOR = TinyCorrector(load_prompt_rows(), pairs_path)
    return TINY_CORRECTOR


def collection_paths(args):
    root = Path(args.collection_dir)
    if not root.is_absolute():
        root = ROOT / root
    wav_dir = root / "wavs"
    return root, wav_dir, root / "metadata.jsonl", root / "pairs.tsv"


def json_body(handler):
    length = int(handler.headers.get("Content-Length", "0"))
    if length <= 0:
        return {}
    if length > 1024 * 1024:
        raise ValueError("JSON request is larger than 1 MB")
    return json.loads(handler.rfile.read(length).decode("utf-8"))


def safe_id(value):
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value.strip()).strip("_")


def atomic_write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def collection_file_paths():
    root = RECORDING_COLLECTION_ROOT
    return {
        "root": root,
        "participants": root / "participants.json",
        "scripts": root / "scripts.tsv",
        "state": root / "state.json",
        "metadata": root / "metadata.jsonl",
        "wavs": root / "wavs",
    }


def default_collection_scripts():
    candidates = []
    transcript = RECIPE_ROOT / "transcripts_matched_u20" / "test.tsv"
    if transcript.is_file():
        seen = set()
        with transcript.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                text = " ".join(row.get("text", "").split())
                if not text or text in seen:
                    continue
                seen.add(text)
                word_count = len(text.split())
                if COLLECTION_MIN_WORDS <= word_count <= COLLECTION_MAX_WORDS:
                    candidates.append(
                        {
                            "id": row.get("utt_id") or f"script_{len(candidates) + 1:03d}",
                            "text": text,
                            "word_count": word_count,
                            "source_utt_id": row.get("utt_id", ""),
                        }
                    )
                if len(candidates) >= COLLECTION_SCRIPT_COUNT:
                    break

    if candidates:
        return candidates

    prompts = load_prompt_rows()[:COLLECTION_SCRIPT_COUNT]
    return [
        {
            "id": f"script_{idx:03d}",
            "text": item["text"],
            "word_count": len(item["text"].split()),
            "source_utt_id": item.get("id", ""),
        }
        for idx, item in enumerate(prompts, start=1)
    ]


def ensure_collection_files():
    paths = collection_file_paths()
    paths["root"].mkdir(parents=True, exist_ok=True)
    paths["wavs"].mkdir(parents=True, exist_ok=True)

    if not paths["participants"].is_file():
        participants = [
            {"id": safe_id(name), "name": name}
            for name in DEFAULT_COLLECTION_PARTICIPANTS
        ]
        atomic_write_json(paths["participants"], {"participants": participants})

    if not paths["scripts"].is_file():
        with paths["scripts"].open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["script_id", "text", "word_count", "source_utt_id"],
                delimiter="\t",
            )
            writer.writeheader()
            for item in default_collection_scripts():
                writer.writerow(
                    {
                        "script_id": item["id"],
                        "text": item["text"],
                        "word_count": item["word_count"],
                        "source_utt_id": item["source_utt_id"],
                    }
                )

    if not paths["state"].is_file():
        atomic_write_json(paths["state"], {"participants": {}, "updated_at": ""})


def load_collection_participants():
    ensure_collection_files()
    paths = collection_file_paths()
    data = json.loads(paths["participants"].read_text(encoding="utf-8"))
    participants = data.get("participants", [])
    return [
        {"id": item.get("id") or safe_id(item.get("name", "")), "name": item.get("name", "")}
        for item in participants
        if item.get("name")
    ]


def load_collection_scripts():
    ensure_collection_files()
    scripts = []
    with collection_file_paths()["scripts"].open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            text = " ".join(row.get("text", "").split())
            if not text:
                continue
            script_id = row.get("script_id") or row.get("id") or f"script_{len(scripts) + 1:03d}"
            try:
                word_count = int(row.get("word_count") or len(text.split()))
            except ValueError:
                word_count = len(text.split())
            scripts.append(
                {
                    "id": script_id,
                    "text": text,
                    "wordCount": word_count,
                    "sourceUttId": row.get("source_utt_id", ""),
                }
            )
    return scripts


def load_collection_state():
    ensure_collection_files()
    path = collection_file_paths()["state"]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        data = {"participants": {}, "updated_at": ""}
    data.setdefault("participants", {})
    return data


def save_collection_state(state):
    state["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    atomic_write_json(collection_file_paths()["state"], state)


def participant_by_id(participant_id):
    for participant in load_collection_participants():
        if participant["id"] == participant_id:
            return participant
    return None


def script_by_id(script_id):
    for script in load_collection_scripts():
        if script["id"] == script_id:
            return script
    return None


def collection_summary(state, scripts):
    total = len(scripts)
    summary = {}
    for participant in load_collection_participants():
        records = state.get("participants", {}).get(participant["id"], {})
        completed = sum(1 for script in scripts if records.get(script["id"], {}).get("status") == "recorded")
        summary[participant["id"]] = {
            "completed": completed,
            "total": total,
            "remaining": max(total - completed, 0),
        }
    return summary


def load_examples():
    transcript_rows = {}
    # examples.tsv ships next to the packaged wavs and is authoritative for them.
    # The recipe transcripts are a fallback and may be stale (they lag dataset
    # rebuilds), so they must not shadow it -- first hit wins via setdefault.
    for transcript in (
        ROOT / "examples" / "examples.tsv",
        RECIPE_ROOT / "transcripts_matched_u20" / "test.tsv",
        RECIPE_ROOT / "transcripts_matched" / "test.tsv",
        RECIPE_ROOT / "transcripts" / "test.tsv",
    ):
        if not transcript.is_file():
            continue
        with transcript.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                transcript_rows.setdefault(row.get("utt_id", ""), row)

    examples = []
    packaged_wavs = sorted((ROOT / "examples").glob("*.wav"))
    if packaged_wavs:
        for wav in packaged_wavs:
            utt_id = wav.stem
            row = transcript_rows.get(utt_id, {})
            speaker = row.get("speaker") or utt_id.split("_", 1)[0]
            text = " ".join((row.get("text") or utt_id).split())
            examples.append(
                {
                    "id": utt_id,
                    "speaker": speaker,
                    "audioPath": "examples/%s" % wav.name,
                    "text": text,
                    "exists": True,
                }
            )
        return examples

    transcript = RECIPE_ROOT / "transcripts" / "test.tsv"
    examples = []
    if not transcript.is_file():
        return examples
    with transcript.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["utt_id"] not in EXAMPLE_IDS:
                continue
            audio_path = RECIPE_ROOT / row["audio_path"]
            examples.append(
                {
                    "id": row["utt_id"],
                    "speaker": row["speaker"],
                    "audioPath": row["audio_path"],
                    "text": row["text"],
                    "exists": audio_path.is_file(),
                }
            )
    by_id = {item["id"]: item for item in examples}
    return [by_id[item] for item in EXAMPLE_IDS if item in by_id]


def example_audio_path(example_id):
    packaged = ROOT / "examples" / f"{example_id}.wav"
    if packaged.is_file():
        return packaged
    for example in load_examples():
        if example["id"] == example_id:
            path = RECIPE_ROOT / example["audioPath"]
            return path if path.is_file() else None
    return None


class AppHandler(BaseHTTPRequestHandler):
    server_version = "VietASRRemoteMic/1.0"

    def log_message(self, fmt, *args):
        print("[%s] %s" % (self.log_date_time_string(), fmt % args))

    @property
    def app_args(self):
        return self.server.app_args

    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path, content_type=None):
        if not path.is_file():
            self.send_error(404)
            return

        file_size = path.stat().st_size
        start = 0
        end = file_size - 1
        status = 200
        range_header = self.headers.get("Range", "")
        if range_header.startswith("bytes="):
            range_spec = range_header[6:].split(",", 1)[0].strip()
            try:
                if range_spec.startswith("-"):
                    suffix_len = int(range_spec[1:])
                    if suffix_len <= 0:
                        raise ValueError
                    start = max(file_size - suffix_len, 0)
                else:
                    parts = range_spec.split("-", 1)
                    start = int(parts[0])
                    if parts[1]:
                        end = int(parts[1])
                if start < 0 or start >= file_size or end < start:
                    raise ValueError
                end = min(end, file_size - 1)
                status = 206
            except ValueError:
                self.send_response(416)
                self.send_header("Content-Range", "bytes */%d" % file_size)
                self.end_headers()
                return

        length = end - start + 1
        self.send_response(status)
        self.send_header(
            "Content-Type",
            content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream",
        )
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == 206:
            self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, file_size))
        self.end_headers()
        if self.command == "HEAD":
            return
        with path.open("rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self.send_file(ROOT / "static" / "index.html", "text/html; charset=utf-8")
        elif parsed.path == "/collect":
            self.send_file(ROOT / "static" / "collect.html", "text/html; charset=utf-8")
        elif parsed.path == "/app.js":
            self.send_file(ROOT / "static" / "app.js", "text/javascript; charset=utf-8")
        elif parsed.path == "/collect.js":
            self.send_file(ROOT / "static" / "collect.js", "text/javascript; charset=utf-8")
        elif parsed.path == "/style.css":
            self.send_file(ROOT / "static" / "style.css", "text/css; charset=utf-8")
        elif parsed.path == "/collect.css":
            self.send_file(ROOT / "static" / "collect.css", "text/css; charset=utf-8")
        elif parsed.path == "/health":
            self.send_json(
                {
                    "ok": True,
                    "jetsonHost": self.app_args.jetson_host,
                    "remoteDir": self.app_args.remote_dir,
                    "modelDir": self.app_args.model_dir,
                    "asrMode": self.app_args.asr_mode,
                    "defaultModel": DEFAULT_MODEL_CHOICE,
                    "models": [
                        {
                            "id": key,
                            "label": value["label"],
                            "modelDir": value["modelDir"],
                            "asrMode": value["asrMode"],
                        }
                        for key, value in MODEL_CHOICES.items()
                    ],
                }
            )
        elif parsed.path == "/examples":
            self.send_json({"ok": True, "examples": load_examples()})
        elif parsed.path == "/prompts":
            prompts = load_prompt_rows()
            query = parse_qs(parsed.query)
            if query.get("random", ["0"])[0] == "1" and prompts:
                self.send_json({"ok": True, "prompt": random.choice(prompts)})
            else:
                self.send_json({"ok": True, "prompts": prompts})
        elif parsed.path == "/collection/config":
            scripts = load_collection_scripts()
            state = load_collection_state()
            self.send_json(
                {
                    "ok": True,
                    "participants": load_collection_participants(),
                    "scripts": scripts,
                    "summary": collection_summary(state, scripts),
                }
            )
        elif parsed.path == "/collection/progress":
            query = parse_qs(parsed.query)
            participant_id = query.get("participant_id", [""])[0]
            if participant_by_id(participant_id) is None:
                self.send_json({"ok": False, "error": "Unknown participant"}, status=404)
                return
            scripts = load_collection_scripts()
            state = load_collection_state()
            records = state.get("participants", {}).get(participant_id, {})
            self.send_json(
                {
                    "ok": True,
                    "participantId": participant_id,
                    "records": records,
                    "summary": collection_summary(state, scripts).get(participant_id, {}),
                }
            )
        elif parsed.path == "/tiny-correct":
            query = parse_qs(parsed.query)
            text = query.get("text", [""])[0]
            self.send_json({"ok": True, "tiny": get_tiny_corrector(self.app_args).correct(text)})
        elif parsed.path == "/example-audio":
            query = parse_qs(parsed.query)
            example_id = query.get("id", [""])[0]
            path = example_audio_path(example_id)
            if path is None:
                self.send_json({"ok": False, "error": "Example audio not found"}, status=404)
                return
            self.send_file(path, "audio/wav")
        elif parsed.path == "/aug-comparison" or parsed.path.startswith("/aug-comparison/"):
            rel = posixpath.normpath(parsed.path[len("/aug-comparison"):].lstrip("/") or "index.html")
            path = (AUG_COMPARISON_ROOT / rel).resolve()
            try:
                path.relative_to(AUG_COMPARISON_ROOT.resolve())
            except ValueError:
                self.send_error(404)
                return
            self.send_file(path)
        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/save-collection":
            self.save_collection()
            return
        if parsed.path == "/collection/record":
            self.save_training_recording(parsed)
            return
        if parsed.path == "/collection/reset-recording":
            self.reset_training_recording()
            return

        if parsed.path not in ("/transcribe", "/transcribe-example"):
            self.send_error(404)
            return

        query = parse_qs(parsed.query)
        use_fp32 = query.get("fp32", ["0"])[0] == "1" or self.app_args.fp32
        model_config = resolve_model(query, self.app_args)

        if parsed.path == "/transcribe-example":
            example_id = query.get("id", [""])[0]
            local_wav = example_audio_path(example_id)
            if local_wav is None:
                self.send_json({"ok": False, "error": "Example audio not found"}, status=404)
                return
            self.transcribe_local_wav(
                local_wav, use_fp32, label=example_id, model_config=model_config
            )
            return

        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            self.send_json({"ok": False, "error": "Empty request body"}, status=400)
            return
        if length > 20 * 1024 * 1024:
            self.send_json({"ok": False, "error": "Audio is larger than 20 MB"}, status=413)
            return

        TMP_ROOT.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        local_wav = TMP_ROOT / f"recording-{stamp}-{time_ns()}.wav"
        local_wav.write_bytes(self.rfile.read(length))
        self.transcribe_local_wav(
            local_wav, use_fp32, label="recording", model_config=model_config
        )

    def save_training_recording(self, parsed):
        query = parse_qs(parsed.query)
        participant_id = query.get("participant_id", [""])[0]
        script_id = query.get("script_id", [""])[0]
        duration_sec = query.get("duration_sec", [""])[0]
        participant = participant_by_id(participant_id)
        script = script_by_id(script_id)
        if participant is None:
            self.send_json({"ok": False, "error": "Unknown participant"}, status=404)
            return
        if script is None:
            self.send_json({"ok": False, "error": "Unknown script"}, status=404)
            return

        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            self.send_json({"ok": False, "error": "Empty request body"}, status=400)
            return
        if length > 25 * 1024 * 1024:
            self.send_json({"ok": False, "error": "Audio is larger than 25 MB"}, status=413)
            return

        paths = collection_file_paths()
        participant_dir = paths["wavs"] / participant_id
        participant_dir.mkdir(parents=True, exist_ok=True)
        created_at = dt.datetime.now(dt.timezone.utc).isoformat()
        stamp = time.strftime("%Y%m%d-%H%M%S")
        digest = hashlib.sha1(
            f"{created_at}|{participant_id}|{script_id}|{length}".encode("utf-8")
        ).hexdigest()[:10]
        wav_path = participant_dir / f"{script_id}__{stamp}__{digest}.wav"
        wav_path.write_bytes(self.rfile.read(length))

        try:
            duration_value = float(duration_sec) if duration_sec else None
        except ValueError:
            duration_value = None
        sample_id = f"{participant_id}_{script_id}_{digest}"
        record = {
            "id": sample_id,
            "created_at": created_at,
            "participant_id": participant_id,
            "participant_name": participant["name"],
            "script_id": script_id,
            "text": script["text"],
            "word_count": script["wordCount"],
            "duration_sec": duration_value,
            "audio_path": os.path.relpath(wav_path, paths["root"]),
            "byte_count": length,
            "user_agent": self.headers.get("User-Agent", ""),
        }

        state = load_collection_state()
        participant_records = state.setdefault("participants", {}).setdefault(participant_id, {})
        previous = participant_records.get(script_id)
        participant_records[script_id] = {
            "status": "recorded",
            "sample_id": sample_id,
            "audio_path": record["audio_path"],
            "duration_sec": duration_value,
            "updated_at": created_at,
            "previous_sample_id": previous.get("sample_id") if previous else "",
        }
        save_collection_state(state)

        with paths["metadata"].open("a", encoding="utf-8") as f:
            f.write(json.dumps({"event": "recorded", **record}, ensure_ascii=False) + "\n")

        scripts = load_collection_scripts()
        self.send_json(
            {
                "ok": True,
                "record": record,
                "summary": collection_summary(state, scripts).get(participant_id, {}),
            }
        )

    def reset_training_recording(self):
        try:
            data = json_body(self)
        except (json.JSONDecodeError, ValueError) as exc:
            self.send_json({"ok": False, "error": str(exc)}, status=400)
            return

        participant_id = data.get("participant_id", "")
        script_id = data.get("script_id", "")
        if participant_by_id(participant_id) is None:
            self.send_json({"ok": False, "error": "Unknown participant"}, status=404)
            return
        if script_by_id(script_id) is None:
            self.send_json({"ok": False, "error": "Unknown script"}, status=404)
            return

        state = load_collection_state()
        records = state.setdefault("participants", {}).setdefault(participant_id, {})
        previous = records.pop(script_id, None)
        save_collection_state(state)
        event = {
            "event": "reset",
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "participant_id": participant_id,
            "script_id": script_id,
            "previous": previous,
            "user_agent": self.headers.get("User-Agent", ""),
        }
        with collection_file_paths()["metadata"].open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
        scripts = load_collection_scripts()
        self.send_json(
            {
                "ok": True,
                "summary": collection_summary(state, scripts).get(participant_id, {}),
            }
        )

    def transcribe_local_wav(self, local_wav, use_fp32, label, model_config):
        if self.app_args.local_mode:
            self.transcribe_on_local_host(local_wav, use_fp32, model_config)
            return

        remote_upload_dir = f"{self.app_args.remote_dir}/ui_uploads"
        remote_wav = f"{remote_upload_dir}/{label}-{time_ns()}.wav"

        ssh_base = [
            "ssh",
            "-i",
            self.app_args.ssh_key,
            "-o",
            "BatchMode=yes",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "ConnectTimeout=8",
            self.app_args.jetson_host,
        ]
        scp_base = [
            "scp",
            "-i",
            self.app_args.ssh_key,
            "-o",
            "BatchMode=yes",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "ConnectTimeout=8",
        ]

        rc, out, err, _ = run_cmd(ssh_base + [f"mkdir -p {shell_quote(remote_upload_dir)}"])
        if rc != 0:
            self.send_json({"ok": False, "error": "Could not create remote upload dir", "stderr": err})
            return

        rc, out, err, _ = run_cmd(scp_base + [str(local_wav), f"{self.app_args.jetson_host}:{remote_wav}"])
        if rc != 0:
            self.send_json({"ok": False, "error": "Could not copy audio to Jetson", "stderr": err})
            return

        fp32_flag = " --fp32" if use_fp32 else ""
        transcribe_script = (
            "transcribe_wav.py"
            if model_config["asr_mode"] == "nonstream"
            else "transcribe_streaming_wav.py"
        )
        remote_cmd = (
            f"cd {shell_quote(self.app_args.remote_dir)} && "
            ". .venv/bin/activate && "
            f"python3 {transcribe_script} "
            f"--model-dir {shell_quote(model_config['model_dir'])} "
            f"--threads {self.app_args.threads} "
            f"--provider {shell_quote(self.app_args.provider)} "
            f"--max-active-paths {self.app_args.max_active_paths}"
            f"{fp32_flag} "
            f"{shell_quote(remote_wav)}"
        )
        rc, out, err, elapsed = run_cmd(ssh_base + [remote_cmd], timeout=180)
        transcript = out.strip().splitlines()[-1].strip() if out.strip() else ""
        if rc != 0:
            self.send_json(
                {
                    "ok": False,
                    "error": "Jetson transcription failed",
                    "stdout": out,
                    "stderr": err,
                }
            )
            return

        tiny = get_tiny_corrector(self.app_args).correct(transcript)
        self.send_json(
            {
                "ok": True,
                "transcript": transcript,
                "tinyCorrected": tiny["corrected"],
                "tinyConfidence": tiny["confidence"],
                "tinySource": tiny["source"],
                "tinyCandidates": tiny["candidates"],
                "stdout": out,
                "stderr": err,
                "elapsedSec": round(elapsed, 3),
                "remoteWav": remote_wav,
                "localWav": str(local_wav),
                "precision": "fp32" if use_fp32 else "int8",
                "model": model_config,
            }
        )

    def save_collection(self):
        try:
            data = json_body(self)
        except (json.JSONDecodeError, ValueError) as exc:
            self.send_json({"ok": False, "error": str(exc)}, status=400)
            return

        target_text = " ".join(data.get("target_text", "").split())
        asr_output = " ".join(data.get("asr_output", "").split())
        corrected_text = " ".join(data.get("corrected_text", "").split())
        source_wav = Path(data.get("local_wav", ""))
        if not target_text:
            self.send_json({"ok": False, "error": "Missing target_text"}, status=400)
            return
        if not asr_output:
            self.send_json({"ok": False, "error": "Missing asr_output"}, status=400)
            return
        if not source_wav.is_file():
            self.send_json({"ok": False, "error": "Missing recorded wav"}, status=400)
            return

        root, wav_dir, metadata_path, pairs_path = collection_paths(self.app_args)
        wav_dir.mkdir(parents=True, exist_ok=True)
        root.mkdir(parents=True, exist_ok=True)

        created_at = dt.datetime.now(dt.timezone.utc).isoformat()
        digest = hashlib.sha1(
            f"{created_at}|{target_text}|{asr_output}|{source_wav}".encode("utf-8")
        ).hexdigest()[:12]
        sample_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{digest}"
        wav_path = wav_dir / f"{sample_id}.wav"
        shutil.copy2(source_wav, wav_path)

        record = {
            "id": sample_id,
            "created_at": created_at,
            "prompt_id": data.get("prompt_id", ""),
            "speaker": data.get("speaker", ""),
            "target_text": target_text,
            "asr_output": asr_output,
            "corrected_text": corrected_text or target_text,
            "model": data.get("model", {}),
            "precision": data.get("precision", ""),
            "elapsed_sec": data.get("elapsed_sec"),
            "duration_sec": data.get("duration_sec"),
            "audio_path": os.path.relpath(wav_path, root),
            "source_local_wav": str(source_wav),
            "user_agent": self.headers.get("User-Agent", ""),
        }

        with metadata_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        new_pairs_file = not pairs_path.exists()
        with pairs_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            if new_pairs_file:
                writer.writerow(["asr_output", "corrected_text", "target_text", "sample_id"])
            writer.writerow([asr_output, record["corrected_text"], target_text, sample_id])

        self.send_json(
            {
                "ok": True,
                "id": sample_id,
                "metadata": str(metadata_path),
                "pairs": str(pairs_path),
                "audio": str(wav_path),
            }
        )

    def transcribe_on_local_host(self, local_wav, use_fp32, model_config):
        fp32_flag = ["--fp32"] if use_fp32 else []
        # "beam" uses the custom onnxruntime classic beam_search decoder (~2.4% WER);
        # sherpa-onnx's modified_beam_search floors out around 9.2%.
        asr_mode = model_config["asr_mode"]
        transcribe_script = {
            "nonstream": "transcribe_wav.py",
            "beam": "transcribe_beam_wav.py",
        }.get(asr_mode, "transcribe_streaming_wav.py")
        python_bin = Path(self.app_args.remote_dir) / ".venv" / "bin" / "python3"
        if not python_bin.is_file():
            python_bin = Path(sys.executable)
        cmd = [
            str(python_bin),
            transcribe_script,
            "--model-dir",
            model_config["model_dir"],
            "--threads",
            str(self.app_args.threads),
            "--provider",
            self.app_args.provider,
            "--max-active-paths",
            str(self.app_args.max_active_paths),
            *fp32_flag,
            str(local_wav),
        ]
        started = time.time()
        proc = subprocess.run(
            cmd,
            cwd=self.app_args.remote_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=180,
        )
        elapsed = time.time() - started
        transcript = proc.stdout.strip().splitlines()[-1].strip() if proc.stdout.strip() else ""
        if proc.returncode != 0:
            self.send_json(
                {
                    "ok": False,
                    "error": "Local Jetson transcription failed",
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                }
            )
            return
        tiny = get_tiny_corrector(self.app_args).correct(transcript)
        self.send_json(
            {
                "ok": True,
                "transcript": transcript,
                "tinyCorrected": tiny["corrected"],
                "tinyConfidence": tiny["confidence"],
                "tinySource": tiny["source"],
                "tinyCandidates": tiny["candidates"],
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "elapsedSec": round(elapsed, 3),
                "localWav": str(local_wav),
                "precision": "fp32" if use_fp32 else "int8",
                "model": model_config,
            }
        )


def main():
    args = get_args()
    if not args.local_mode and not Path(args.ssh_key).is_file():
        raise SystemExit(f"Missing SSH key: {args.ssh_key}")
    if not args.local_mode and (shutil.which("ssh") is None or shutil.which("scp") is None):
        raise SystemExit("Missing ssh/scp commands")

    server = ThreadingHTTPServer((args.host, args.port), AppHandler)
    server.app_args = args
    scheme = "https" if args.certfile and args.keyfile else "http"
    if args.certfile and args.keyfile:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(args.certfile, args.keyfile)
        server.socket = context.wrap_socket(server.socket, server_side=True)
    print(f"Remote mic UI: {scheme}://{args.host}:{args.port}")
    print(f"Mode: {'local Jetson' if args.local_mode else 'remote SSH'}")
    print(f"Jetson: {args.jetson_host}")
    print(f"Remote dir: {args.remote_dir}")
    print(f"Model: {args.model_dir}")
    print(f"ASR mode: {args.asr_mode}")
    server.serve_forever()


if __name__ == "__main__":
    main()
