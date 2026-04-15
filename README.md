# VietnameseASR / vi_asr_corpus

README này chỉ giữ phần cần thiết để biết các file chạy như nào và thứ tự dùng trong project hiện tại.

## 1. Cấu trúc chính

```bash
~/icefall/egs/vi_asr_corpus/
├── ASR/zipformer/
├── audio/
├── transcripts/
├── manifests/
├── manifests_fixed/
├── fbank/
├── data/
│   └── lang_bpe_<vocab_size>/
├── local/
│   ├── prepare_manifests.py
│   ├── export_text_corpus.py
│   ├── train_bpe_model.py
│   ├── prepare_lang_bpe.py
│   ├── compute_fbank.py
│   ├── validate_manifest.py
│   ├── display_manifest_statistics.py
│   ├── tokenize_test.py
│   └── compute_fbank_musan.py
├── prepare_vi_asr_corpus.py
├── augment_train_with_musan.py
├── run.sh
└── README.md
```

---

## 2. `prepare_vi_asr_corpus.py`

File này dùng để tạo `audio/` và `transcripts/`.

### Chế độ 1: single-speaker
```bash
cd ~/icefall/egs/vi_asr_corpus

python prepare_vi_asr_corpus.py \
  --audio-dir /duong/dan/toi/audio \
  --prompts /duong/dan/toi/prompts.txt \
  --speaker trung \
  --overwrite
```

### Chế độ 2: auto multi-speaker
```bash
cd ~/icefall/egs/vi_asr_corpus

python prepare_vi_asr_corpus.py \
  --auto \
  --overwrite
```

### Shuffle trước khi split
Nếu dữ liệu của bạn đang được ghi theo block kiểu:

```text
câu 1
câu 1
câu 1
...
câu 2
...
câu 3
```

thì nên bật shuffle pair `(audio, text)` trước khi chia train/dev/test:

```bash
python prepare_vi_asr_corpus.py \
  --auto \
  --shuffle-before-split \
  --seed 42 \
  --overwrite
```

### Normalize text
Mặc định đang bật normalize. Nếu muốn tắt:

```bash
python prepare_vi_asr_corpus.py --auto --no-normalize-text --overwrite
```

---

## 3. `augment_train_with_musan.py`

File này dùng để **tạo thêm audio train mới trên disk** từ MUSAN.

Nó sẽ:
- đọc `transcripts/train.tsv`
- với mỗi utterance train gốc, sinh thêm `n` bản noisy mới
- thêm các dòng mới vào `transcripts/train.tsv`
- lưu audio mới vào `audio/train_aug_musan/`

Ví dụ:

```bash
cd ~/icefall/egs/vi_asr_corpus

python augment_train_with_musan.py \
  --corpus-root . \
  --musan-dir /home/trung/icefall/egs/librispeech/ASR/download/musan \
  --copies-per-utt 3 \
  --snr-min 10 \
  --snr-max 20
```

Nếu muốn làm lại từ đầu:

```bash
python augment_train_with_musan.py \
  --corpus-root . \
  --musan-dir /home/trung/icefall/egs/librispeech/ASR/download/musan \
  --copies-per-utt 3 \
  --snr-min 10 \
  --snr-max 20 \
  --overwrite
```

---

## 4. Các file trong `local/`

### `prepare_manifests.py`
Tạo:
- `manifests/train_recordings.jsonl.gz`
- `manifests/train_supervisions.jsonl.gz`
- tương tự cho dev/test

Chạy:
```bash
python local/prepare_manifests.py
```

### `export_text_corpus.py`
Gộp text train để chuẩn bị train SentencePiece/BPE.

Chạy:
```bash
python local/export_text_corpus.py
```

### `train_bpe_model.py`
Train SentencePiece BPE.

File này đã sửa để nhận `--vocab-size`, không cần đổi tay trong code nữa.

Ví dụ:
```bash
python local/train_bpe_model.py --vocab-size 100
```

Kết quả sẽ nằm ở:
```bash
data/lang_bpe_100/
```

### `prepare_lang_bpe.py`
Chuẩn bị tối thiểu `tokens.txt`, `words.txt` từ `bpe.model`.

Ví dụ:
```bash
python local/prepare_lang_bpe.py --lang-dir data/lang_bpe_100
```

### `compute_fbank.py`
Tạo `train_cuts.jsonl.gz`, `dev_cuts.jsonl.gz`, `test_cuts.jsonl.gz` và feature trong `fbank/`.

Ví dụ:
```bash
python local/compute_fbank.py \
  --bpe-model data/lang_bpe_100/bpe.model \
  --manifest-dir manifests_fixed \
  --output-dir fbank
```

Nếu muốn speed perturb:
```bash
python local/compute_fbank.py \
  --bpe-model data/lang_bpe_100/bpe.model \
  --manifest-dir manifests_fixed \
  --output-dir fbank \
  --perturb-speed
```

### `validate_manifest.py`
Kiểm tra cut manifests có hợp lệ cho ASR không.

Ví dụ:
```bash
python local/validate_manifest.py --all --manifest-dir fbank
```

### `display_manifest_statistics.py`
In thống kê cut để xem số lượng câu, total duration, mean duration,...

Ví dụ:
```bash
python local/display_manifest_statistics.py --all --manifest-dir fbank
```

### `tokenize_test.py`
Test tokenize với đúng `vocab_size`.

Ví dụ:
```bash
python local/tokenize_test.py --vocab-size 100
```

Hoặc:
```bash
python local/tokenize_test.py --vocab-size 100 --text "hôm nay tôi học nhận dạng tiếng nói"
```

### MUSAN manifests
Manifest MUSAN dùng format giống LibriSpeech recipe và nằm trong:

```bash
data/manifests/musan_recordings_music.jsonl.gz
data/manifests/musan_recordings_noise.jsonl.gz
data/manifests/musan_recordings_speech.jsonl.gz
```

Nếu cần tạo lại từ thư mục MUSAN raw:

```bash
lhotse prepare musan /home/trung/icefall/egs/librispeech/ASR/download/musan data/manifests
```

### `compute_fbank_musan.py`
Tạo `fbank/musan_cuts.jsonl.gz` để dùng MUSAN online khi train.

Ví dụ:
```bash
python local/compute_fbank_musan.py \
  --manifest-dir data/manifests \
  --output-dir fbank
```

---

## 5. `run.sh`

`run.sh` hiện đã sửa để:
- nhận `--vocab_size`
- tự suy ra:
  - `data/lang_bpe_<vocab_size>`
  - `ASR/zipformer/exp_bpe<vocab_size>`
- hỗ trợ cả:
  - offline MUSAN augmentation
  - online MUSAN augmentation

### Các stage hiện tại

- **Stage 0**: audit dataset
- **Stage 1**: offline MUSAN augmentation cho train split
- **Stage 2**: prepare manifests
- **Stage 3**: `lhotse fix`
- **Stage 4**: export text corpus
- **Stage 5**: train BPE
- **Stage 6**: prepare minimal BPE lang dir
- **Stage 7**: compute fbank / cuts
- **Stage 8**: compute MUSAN fbank/cuts cho online CutMix
- **Stage 9**: validate cut manifests
- **Stage 10**: display manifest statistics
- **Stage 11**: tokenize smoke test
- **Stage 12**: train
- **Stage 13**: decode

### Các tham số chính

```bash
--vocab_size
--num_epochs
--world_size
--max_duration
--base_lr
--use_fp16
--enable_musan
--enable_spec_aug
--bucketing_sampler
--num_buckets
--perturb_speed
--musan_dir
--offline_musan_aug
--copies_per_utt
--snr_min
--snr_max
--decode_method
--use_averaged_model
--avg
```

---

## 6. Ví dụ chạy

### 6.1. Chạy data prep cơ bản với vocab size 100
```bash
cd ~/icefall/egs/vi_asr_corpus

bash run.sh \
  --vocab_size 100 \
  --stage 0 --stop_stage 11
```

### 6.2. Chỉ train BPE với vocab size 100
```bash
cd ~/icefall/egs/vi_asr_corpus

bash run.sh \
  --vocab_size 100 \
  --stage 5 --stop_stage 6
```

### 6.3. Train model
```bash
cd ~/icefall/egs/vi_asr_corpus

bash run.sh \
  --vocab_size 100 \
  --stage 12 --stop_stage 12
```

### 6.4. Decode
```bash
cd ~/icefall/egs/vi_asr_corpus

bash run.sh \
  --vocab_size 100 \
  --stage 13 --stop_stage 13
```

### 6.5. Offline MUSAN augmentation + train
```bash
cd ~/icefall/egs/vi_asr_corpus

bash run.sh \
  --vocab_size 100 \
  --offline_musan_aug 1 \
  --musan_dir /home/trung/icefall/egs/librispeech/ASR/download/musan \
  --copies_per_utt 2 \
  --snr_min 10 \
  --snr_max 20 \
  --enable_musan 0 \
  --stage 1 --stop_stage 12
```

### 6.6. Online MUSAN augmentation + train
```bash
cd ~/icefall/egs/vi_asr_corpus

bash run.sh \
  --vocab_size 100 \
  --enable_musan 1 \
  --stage 8 --stop_stage 12
```

### 6.7. Kết hợp cả offline + online MUSAN
```bash
cd ~/icefall/egs/vi_asr_corpus

bash run.sh \
  --vocab_size 100 \
  --offline_musan_aug 1 \
  --musan_dir /home/trung/icefall/egs/librispeech/ASR/download/musan \
  --copies_per_utt 2 \
  --snr_min 10 \
  --snr_max 20 \
  --enable_musan 1 \
  --stage 1 --stop_stage 12
```

---
