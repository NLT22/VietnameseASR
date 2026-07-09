# Remote Mic UI

Small local browser UI for recording audio on this computer and running
VietnameseASR inference on the Jetson Nano over SSH.

The Jetson does not need a microphone. The browser records a 16 kHz mono WAV,
the local backend copies it to the Jetson, runs `transcribe_wav.py`, and returns
the transcript.

## Run

From this computer:

```bash
cd /media/pc/c88ba509-53f0-4c97-9e44-e33483754b08/icefall/egs/VietnameseASR/deploy/jetson_nano/remote_mic_ui
python3 server.py --host 0.0.0.0
```

Open:

```text
http://127.0.0.1:8765
```

The ASR demo is at `/`. The recording-only data collection page is at:

```text
http://127.0.0.1:8765/collect
```

From another device on the same Wi-Fi/LAN, use this computer's LAN IP:

```text
http://<this-computer-ip>:8765
```

Default connection settings:

- Jetson: `thayhoang@192.168.1.50`
- SSH key: `/media/pc/cde52536-a42f-46bb-b753-bbd8c184686f/home/trung/.ssh/id_ed25519`
- Remote deploy dir: `/home/thayhoang/vietnamese_asr_eval/jetson_nano`
- Model dir: `model_nonstream_scratch50_best`

Override them if needed:

```bash
python3 server.py \
  --jetson-host thayhoang@192.168.1.50 \
  --ssh-key /path/to/id_ed25519 \
  --remote-dir /home/thayhoang/vietnamese_asr_eval/jetson_nano \
  --model-dir model_nonstream_scratch50_best
```

## Notes

- Browser microphone access works on `localhost`.
- On phones, direct browser microphone access usually requires trusted HTTPS.
  If you open the HTTP Jetson page, use `Phone Record / Choose Audio`, then
  `Send To Jetson`.
- The dataset example buttons work over plain HTTP.
- The UI sends audio only to the local backend; the backend copies it to Jetson
  with `scp`.
- Default inference uses CPU int8 ONNX on Jetson, matching the current reliable
  deployment path.

## Run Directly On Jetson

This is useful when another device cannot connect to this computer.

```bash
cd /home/thayhoang/vietnamese_asr_eval/jetson_nano/remote_mic_ui
openssl req -x509 -newkey rsa:2048 -nodes \
  -keyout ui.key -out ui.crt -days 365 \
  -subj "/CN=192.168.1.50"
../.venv/bin/python3 server.py \
  --host 0.0.0.0 \
  --port 8765 \
  --local-mode \
  --remote-dir /home/thayhoang/vietnamese_asr_eval/jetson_nano \
  --certfile ui.crt \
  --keyfile ui.key
```

Then open:

```text
https://192.168.1.50:8765
```

The browser will warn about the self-signed certificate. Accept it for this
local Jetson page.

## Recording Collection

`/collect` is for collecting clean retraining audio. It does not run ASR while
people record. The current participant list is:

```text
Trung, Khoi, Dung, Hieu, Quan
```

Collection files are stored here:

```text
remote_mic_ui/collection/
```

Important files:

- `participants.json`: editable participant list.
- `scripts.tsv`: editable recording script list.
- `state.json`: latest per-person progress.
- `metadata.jsonl`: append-only log for saved recordings and resets.
- `wavs/<participant_id>/`: saved 16 kHz mono WAV recordings.

After collecting audio, create ASR-output pairs later with:

```bash
cd /home/thayhoang/vietnamese_asr_eval/jetson_nano/remote_mic_ui
../.venv/bin/python3 batch_transcribe_collection.py --dry-run
../.venv/bin/python3 batch_transcribe_collection.py \
  --remote-dir /home/thayhoang/vietnamese_asr_eval/jetson_nano \
  --model-dir model_streaming_u20_epoch55_avg10 \
  --asr-mode streaming \
  --provider cuda
```

Outputs:

```text
collection/metadata_asr.jsonl
collection/asr_pairs.tsv
```
