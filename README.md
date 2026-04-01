# VietnameseASR / vi_asr_corpus

Tài liệu này mô tả cách chuẩn bị corpus, chạy pipeline bằng `run.sh`, xem kết quả train/decode, mở TensorBoard, và cách tổ chức file để tránh đẩy dữ liệu lớn lên Git.

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
│   └── test/<speaker>/*.wav
├── transcripts/
│   ├── train.tsv
│   ├── dev.tsv
│   └── test.tsv
├── manifests/
├── manifests_fixed/
├── fbank/
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
│   └── tokenize_test.py
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

Vì vậy bạn có thể dùng `--overwrite` để build lại corpus mà không làm mất code và artifact khác.

## 3. Pipeline `run.sh`

`run.sh` chạy toàn bộ pipeline theo stage.

### 3.1. Các stage hiện tại

- **Stage 0**: audit dataset
- **Stage 1**: prepare manifests
- **Stage 2**: `lhotse fix`
- **Stage 3**: export text corpus
- **Stage 4**: train BPE
- **Stage 5**: prepare minimal BPE lang dir
- **Stage 6**: compute fbank / cuts
- **Stage 7**: validate cut manifests
- **Stage 8**: display manifest statistics
- **Stage 9**: tokenize smoke test
- **Stage 10**: train
- **Stage 11**: decode
- **Stage 12**: in lệnh mở TensorBoard
- **Stage 13**: in lệnh xem file kết quả

### 3.2. Chạy toàn bộ

```bash
cd ~/icefall/egs/vi_asr_corpus
bash run.sh
```

### 3.3. Chạy theo stage

Chỉ chuẩn bị dữ liệu đến tokenize smoke test:

```bash
cd ~/icefall/egs/vi_asr_corpus
bash run.sh --stage 0 --stop_stage 9
```

Chỉ train:

```bash
cd ~/icefall/egs/vi_asr_corpus
bash run.sh --stage 10 --stop_stage 10
```

Chỉ decode:

```bash
cd ~/icefall/egs/vi_asr_corpus
bash run.sh --stage 11 --stop_stage 11
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
--decode_method
--use_averaged_model
--avg
```

### Ví dụ train smoke test ổn định

```bash
cd ~/icefall/egs/vi_asr_corpus

bash run.sh \
  --stage 10 --stop_stage 10 \
  --num_epochs 1 \
  --world_size 1 \
  --max_duration 30 \
  --base_lr 0.02 \
  --use_fp16 1 \
  --enable_musan 0 \
  --enable_spec_aug 0 \
  --bucketing_sampler 0
```

### Ví dụ thử `DynamicBucketingSampler`

Nếu dataset đã đủ lớn và bạn đã test thấy bucketing chạy được:

```bash
cd ~/icefall/egs/vi_asr_corpus

bash run.sh \
  --stage 10 --stop_stage 10 \
  --num_epochs 1 \
  --bucketing_sampler 1 \
  --num_buckets 4 \
  --enable_spec_aug 0 \
  --enable_musan 0
```

### Ví dụ bật speed perturb khi tạo fbank

```bash
cd ~/icefall/egs/vi_asr_corpus

bash run.sh --stage 6 --stop_stage 6 --perturb_speed 1
```

## 5. Tăng cường dữ liệu hiện có

Trong pipeline hiện tại có sẵn:

### 5.1. MUSAN noise mixing

Bật:

```bash
--enable_musan 1
```

Điều kiện: phải có `musan_cuts.jsonl.gz` trong `manifest-dir` tương ứng.

### 5.2. SpecAugment

Bật:

```bash
--enable_spec_aug 1
```

Với dataset rất nhỏ, nên để `0` khi smoke test. Khi dữ liệu nhiều hơn có thể bật lại.

### 5.3. Speed perturb

Bật ở bước tạo cuts/fbank:

```bash
--perturb_speed 1
```

### 5.4. DynamicBucketingSampler

Bật:

```bash
--bucketing_sampler 1 --num_buckets 4
```

Khuyến nghị:

- dataset nhỏ: `--bucketing_sampler 0`
- dataset lớn hơn: bật `1`

## 6. Xem kết quả train và decode

### 6.1. Xem checkpoint

```bash
cd ~/icefall/egs/vi_asr_corpus
ls ASR/zipformer
ls ASR/zipformer/exp_100bpe_0.02
```

### 6.2. Xem file decode

```bash
cd ~/icefall/egs/vi_asr_corpus
ls ASR/zipformer/exp_100bpe_0.02/greedy_search
```

### 6.3. Xem transcript nhận dạng

```bash
cd ~/icefall/egs/vi_asr_corpus
cat ASR/zipformer/exp_100bpe_0.02/greedy_search/recogs-test-epoch-30_avg-1_context-2_max-sym-per-frame-1.txt
```

Tên file thực tế có thể khác theo `epoch`, `avg` và `decode_method`. Để xem tất cả file:

```bash
cd ~/icefall/egs/vi_asr_corpus
find ASR/zipformer/exp_100bpe_0.02 -maxdepth 2 -type f | sort
```

### 6.4. Xem WER summary

```bash
cd ~/icefall/egs/vi_asr_corpus
cat ASR/zipformer/exp_100bpe_0.02/greedy_search/wer-summary-test-epoch-30_avg-1_context-2_max-sym-per-frame-1.txt
```

### 6.5. Tìm nhanh các file kết quả

```bash
cd ~/icefall/egs/vi_asr_corpus
find ASR/zipformer/exp_100bpe_0.02 -maxdepth 2 -type f | sort
```

## 7. Mở TensorBoard

Chạy đúng các lệnh sau:

```bash
cd ~/icefall/egs/vi_asr_corpus
tensorboard --logdir ASR/zipformer/exp/tensorboard --port 6006
```

Nếu bạn đang dùng thư mục experiment khác, ví dụ `exp_100bpe_0.02`, hãy đổi đúng logdir:

```bash
cd ~/icefall/egs/vi_asr_corpus
tensorboard --logdir ASR/zipformer/exp_100bpe_0.02/tensorboard --port 6006
```

Sau đó mở trình duyệt tại:

```text
http://localhost:6006
```

## 8. Một số lưu ý thực tế

### 8.1. Dataset nhỏ
Với dataset rất nhỏ, cấu hình an toàn là:

```bash
--enable_musan 0
--enable_spec_aug 0
--bucketing_sampler 0
```

### 8.2. Khi dữ liệu tăng lên
Bạn có thể thử lại:

```bash
--bucketing_sampler 1
--num_buckets 4
--enable_spec_aug 1
--perturb_speed 1
```

rồi sau đó mới cân nhắc `MUSAN`.

### 8.3. Nếu dữ liệu thay đổi
Khi thêm speaker hoặc sửa transcript, nên rebuild lại từ:

- manifests
- manifests_fixed
- transcript_words.txt
- bpe.model
- prepare_lang_bpe
- fbank
- validate_manifest

## 9. Git và `.gitignore`

Project này có rất nhiều thư mục dữ liệu lớn, thường **không nên đẩy lên GitHub**.

### 9.1. Nên ignore
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

### 9.2. Ý tưởng
Git chỉ nên giữ:
- code
- script
- README
- cấu hình
- file nhỏ, có thể tái tạo

Còn những thứ có thể generate lại hoặc quá lớn thì nên ignore.

## 10. Ví dụ nhanh

### Chuẩn bị corpus auto

```bash
cd ~/icefall/egs/vi_asr_corpus
python prepare_vi_asr_corpus.py --auto --overwrite
```

### Chạy pipeline data prep

```bash
cd ~/icefall/egs/vi_asr_corpus
bash run.sh --stage 0 --stop_stage 9
```

### Train

```bash
cd ~/icefall/egs/vi_asr_corpus
bash run.sh --stage 10 --stop_stage 10 --num_epochs 1 --bucketing_sampler 1 --num_buckets 4
```

### Decode

```bash
cd ~/icefall/egs/vi_asr_corpus
bash run.sh --stage 11 --stop_stage 11 --num_epochs 30 --decode_method greedy_search --avg 1 --use_averaged_model 0
```

### Mở TensorBoard

```bash
cd ~/icefall/egs/vi_asr_corpus
tensorboard --logdir ASR/zipformer/exp_100bpe_0.02/tensorboard --port 6006
```
