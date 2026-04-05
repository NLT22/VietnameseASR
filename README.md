# VietnameseASR / vi_asr_corpus

Tài liệu này mô tả cách chuẩn bị corpus, chạy pipeline bằng `run.sh`, dùng MUSAN theo cả kiểu offline và online, xem kết quả train/decode, và mở TensorBoard.

## 1. Cấu trúc project

Ví dụ cấu trúc thư mục:

```bash
~/icefall/egs/vi_asr_corpus/
├── ASR/
│   └── zipformer/
│       ├── train.py
│       ├── decode.py
│       ├── streaming_decode.py
│       ├── asr_datamodule.py
│       └── ...
├── audio/
│   ├── train/<speaker>/*.wav
│   ├── dev/<speaker>/*.wav
│   ├── test/<speaker>/*.wav
│   └── train_aug_musan/<speaker>/*.wav
├── transcripts/
│   ├── train.tsv
│   ├── dev.tsv
│   └── test.tsv
├── manifests/
├── manifests_fixed/
├── fbank/
│   └── musan_cuts.jsonl.gz
├── data/
│   └── lang_bpe_100/
│       ├── bpe.model
│       ├── bpe.vocab
│       ├── tokens.txt
│       └── words.txt
├── local/
│   ├── prepare_manifests.py
│   ├── compute_fbank.py
│   ├── export_text_corpus.py
│   ├── train_bpe_model.py
│   ├── prepare_lang_bpe.py
│   ├── validate_manifest.py
│   ├── display_manifest_statistics.py
│   ├── tokenize_test.py
│   ├── prepare_musan_manifest.py
│   └── compute_fbank_musan.py
├── augment_train_with_musan.py
├── prepare_vi_asr_corpus.py
├── run.sh
├── .gitignore
└── README.md
```

## 2. Chuẩn bị corpus bằng `prepare_vi_asr_corpus.py`

Script hiện hỗ trợ 2 chế độ:

- **single-speaker mode**: truyền `--audio-dir --prompts --speaker`
- **auto mode**: truyền `--auto`, script tự quét `dataset/`

### 2.1. Single-speaker mode

```bash
cd ~/icefall/egs/vi_asr_corpus

python prepare_vi_asr_corpus.py \
  --audio-dir /duong/dan/toi/thu_muc_audio \
  --prompts /duong/dan/toi/prompts.txt \
  --speaker trung \
  --overwrite
```

### 2.2. Auto mode

Giả sử cấu trúc:

```bash
~/icefall/egs/vi_asr_corpus/
├── prepare_vi_asr_corpus.py
├── dataset/
│   ├── trung/
│   │   ├── abc(1).wav
│   │   ├── abc(2).wav
│   │   └── script.txt
│   ├── lan/
│   │   ├── rec(1).m4a
│   │   ├── rec(2).m4a
│   │   └── prompts.txt
│   └── minh/
│       ├── sample1.wav
│       ├── sample2.wav
│       └── transcript.txt
```

chạy:

```bash
cd ~/icefall/egs/vi_asr_corpus
python prepare_vi_asr_corpus.py --auto --overwrite
```

### 2.3. Text normalization

Hiện mặc định **normalize text luôn**. Việc normalize gồm:

- Unicode NFC
- lowercase
- bỏ dấu câu
- gom nhiều khoảng trắng thành 1
- bỏ dòng trống nếu có trong file prompt/script

Nếu muốn tắt normalize:

```bash
python prepare_vi_asr_corpus.py --auto --no-normalize-text --overwrite
```

### 2.4. `--overwrite` có tác dụng gì

`--overwrite` hiện **an toàn**: nó chỉ xóa các phần do script quản lý trong `output-root`:

- `audio/`
- `transcripts/`
- `README_PREPARED.txt`

Nó **không xóa** các thư mục/file khác như:

- `ASR/`
- `fbank/`
- `manifests/`
- `manifests_fixed/`
- `data/`
- `local/`
- `run.sh`
- `train.py`, `decode.py`, ...

## 3. Pipeline `run.sh`

`run.sh` chạy toàn bộ pipeline theo stage.

### 3.1. Các stage hiện tại

- **Stage 0**: audit dataset
- **Stage 1**: offline MUSAN augmentation cho train split
- **Stage 2**: prepare manifests
- **Stage 3**: `lhotse fix`
- **Stage 4**: export text corpus
- **Stage 5**: train BPE
- **Stage 6**: prepare minimal BPE lang dir
- **Stage 7**: compute fbank / cuts
- **Stage 8**: prepare MUSAN manifests/features cho online CutMix
- **Stage 9**: validate cut manifests
- **Stage 10**: display manifest statistics
- **Stage 11**: tokenize smoke test
- **Stage 12**: train
- **Stage 13**: decode
- **Stage 14**: in lệnh mở TensorBoard
- **Stage 15**: in lệnh xem file kết quả

### 3.2. Chạy toàn bộ

```bash
cd ~/icefall/egs/vi_asr_corpus
bash run.sh
```

### 3.3. Chạy theo stage

Chỉ chuẩn bị dữ liệu đến tokenize smoke test:

```bash
cd ~/icefall/egs/vi_asr_corpus
bash run.sh --stage 0 --stop_stage 11
```

Chỉ train:

```bash
cd ~/icefall/egs/vi_asr_corpus
bash run.sh --stage 12 --stop_stage 12
```

Chỉ decode:

```bash
cd ~/icefall/egs/vi_asr_corpus
bash run.sh --stage 13 --stop_stage 13
```

## 4. Các tham số của `run.sh`

`run.sh` hiện hỗ trợ các tham số chính sau:

```bash
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

## 5. MUSAN: offline và online khác nhau thế nào

## 5.1. Offline MUSAN augmentation

Đây là kiểu:
- lấy mỗi file trong `train.tsv`
- sinh thêm `n` file noisy mới trên disk
- append thêm các dòng mới vào `transcripts/train.tsv`

Bật bằng:

```bash
--offline_musan_aug 1
--musan_dir /path/to/musan
--copies_per_utt 2
--snr_min 10
--snr_max 20
```

Ví dụ:

```bash
cd ~/icefall/egs/vi_asr_corpus

bash run.sh \
  --stage 1 --stop_stage 1 \
  --offline_musan_aug 1 \
  --musan_dir /home/trung/icefall/egs/librispeech/ASR/download/musan \
  --copies_per_utt 3 \
  --snr_min 10 \
  --snr_max 20
```

### Kết quả
- tạo audio mới trong `audio/train_aug_musan/`
- thêm `utt_id` mới vào `transcripts/train.tsv`
- transcript giữ nguyên như file gốc

## 5.2. Online MUSAN augmentation

Đây là kiểu:
- không tạo file audio noisy mới trên disk
- chỉ tạo `musan_cuts.jsonl.gz`
- đến lúc train, `asr_datamodule.py` dùng `CutMix` để trộn noise online trong batch

Bật bằng:

```bash
--enable_musan 1
--musan_dir /path/to/musan
```

Ví dụ:

```bash
cd ~/icefall/egs/vi_asr_corpus

bash run.sh \
  --stage 8 --stop_stage 12 \
  --enable_musan 1 \
  --musan_dir /home/trung/icefall/egs/librispeech/ASR/download/musan
```

## 5.3. Nên dùng kiểu nào

Với dữ liệu rất ít, thứ tự thử hợp lý là:

1. **baseline**: không augmentation
2. **offline MUSAN**: để tăng số lượng dữ liệu train thực sự
3. **online MUSAN**: để tăng đa dạng hơn khi train
4. **SpecAugment**: bật sau cùng

Với dataset nhỏ, không nên bật quá nhiều augmentation cùng lúc ngay từ đầu.

## 6. Tăng cường dữ liệu hiện có

### 6.1. Offline MUSAN
Bật:

```bash
--offline_musan_aug 1
```

### 6.2. Online MUSAN CutMix
Bật:

```bash
--enable_musan 1
```

Điều kiện: phải có `musan_cuts.jsonl.gz`, và `run.sh` sẽ tự chuẩn bị nếu bạn truyền `--musan_dir`.

### 6.3. SpecAugment
Bật:

```bash
--enable_spec_aug 1
```

Với dataset rất nhỏ, nên để `0` khi smoke test.

### 6.4. Speed perturb
Bật:

```bash
--perturb_speed 1
```

ở bước tạo cuts/fbank.

### 6.5. DynamicBucketingSampler
Bật:

```bash
--bucketing_sampler 1 --num_buckets 4
```

Khuyến nghị:

- dataset nhỏ: `--bucketing_sampler 0`
- dataset lớn hơn: bật `1`

## 7. Ví dụ pipeline thường dùng

### 7.1. Baseline không augmentation

```bash
cd ~/icefall/egs/vi_asr_corpus

bash run.sh \
  --stage 0 --stop_stage 13 \
  --enable_musan 0 \
  --enable_spec_aug 0 \
  --bucketing_sampler 0
```

### 7.2. Chỉ offline MUSAN augmentation

```bash
cd ~/icefall/egs/vi_asr_corpus

bash run.sh \
  --stage 1 --stop_stage 13 \
  --offline_musan_aug 1 \
  --musan_dir /home/trung/icefall/egs/librispeech/ASR/download/musan \
  --copies_per_utt 2 \
  --snr_min 10 \
  --snr_max 20 \
  --enable_musan 0 \
  --enable_spec_aug 0
```

### 7.3. Chỉ online MUSAN augmentation

```bash
cd ~/icefall/egs/vi_asr_corpus

bash run.sh \
  --stage 0 --stop_stage 13 \
  --enable_musan 1 \
  --musan_dir /home/trung/icefall/egs/librispeech/ASR/download/musan \
  --enable_spec_aug 0
```

### 7.4. Kết hợp cả offline và online

```bash
cd ~/icefall/egs/vi_asr_corpus

bash run.sh \
  --stage 1 --stop_stage 13 \
  --offline_musan_aug 1 \
  --musan_dir /home/trung/icefall/egs/librispeech/ASR/download/musan \
  --copies_per_utt 2 \
  --snr_min 10 \
  --snr_max 20 \
  --enable_musan 1 \
  --enable_spec_aug 0
```

## 8. Xem kết quả train và decode

### 8.1. Xem checkpoint

```bash
cd ~/icefall/egs/vi_asr_corpus
ls ASR/zipformer
ls ASR/zipformer/exp_100bpe_0.02
```

### 8.2. Xem file decode

```bash
cd ~/icefall/egs/vi_asr_corpus
ls ASR/zipformer/exp_100bpe_0.02/greedy_search
```

### 8.3. Xem transcript nhận dạng

```bash
cd ~/icefall/egs/vi_asr_corpus
cat ASR/zipformer/exp_100bpe_0.02/greedy_search/recogs-test-epoch-30_avg-1_context-2_max-sym-per-frame-1.txt
```

Tên file thực tế có thể khác theo `epoch`, `avg` và `decode_method`. Để xem tất cả file:

```bash
cd ~/icefall/egs/vi_asr_corpus
find ASR/zipformer/exp_100bpe_0.02 -maxdepth 2 -type f | sort
```

### 8.4. Xem WER summary

```bash
cd ~/icefall/egs/vi_asr_corpus
cat ASR/zipformer/exp_100bpe_0.02/greedy_search/wer-summary-test-epoch-30_avg-1_context-2_max-sym-per-frame-1.txt
```

## 9. Mở TensorBoard

```bash
cd ~/icefall/egs/vi_asr_corpus
tensorboard --logdir ASR/zipformer/exp_100bpe_0.02/tensorboard --port 6006
```

Sau đó mở:

```text
http://localhost:6006
```

## 10. Nếu dữ liệu thay đổi

Khi:
- thêm speaker
- sửa transcript
- chạy offline MUSAN augmentation lại

thì nên rebuild lại từ:

- manifests
- manifests_fixed
- transcript_words.txt
- bpe.model
- prepare_lang_bpe
- fbank
- validate_manifest

## 11. Git và `.gitignore`

Project này có rất nhiều thư mục dữ liệu lớn, thường **không nên đẩy lên GitHub**.

### 11.1. Nên ignore
- `dataset/`
- `audio/`
- `musan/`
- `fbank/`
- `manifests/`
- `manifests_fixed/`
- `transcripts/`
- `data/`
- `lang/`
- checkpoint và log trong `exp/`

### 11.2. Ý tưởng
Git chỉ nên giữ:
- code
- script
- README
- cấu hình
- file nhỏ, có thể tái tạo

Còn những thứ có thể generate lại hoặc quá lớn thì nên ignore.
