# VietnameseASR / vi_asr_corpus

Tài liệu này hướng dẫn sử dụng hai thành phần chính trong project:

1. `prepare_vi_asr_corpus.py`: chuẩn bị dữ liệu thô thành cấu trúc corpus dùng cho ASR.
2. `run.sh`: chạy pipeline chuẩn bị dữ liệu và huấn luyện mô hình theo từng stage.

## 1. Cấu trúc project

Ví dụ cấu trúc thư mục sau khi chuẩn bị dữ liệu:

```bash
vi_asr_corpus/
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
│   └── lang_bpe_500/
├── local/
│   ├── prepare_manifests.py
│   ├── compute_fbank.py
│   ├── export_text_corpus.py
│   ├── train_bpe_model.py
│   └── tokenize_test.py
└── run.sh
```

## 2. Hướng dẫn dùng `prepare_vi_asr_corpus.py`

Script này dùng để:
- đọc một thư mục chứa các file ghi âm
- đọc file `prompts.txt` với mỗi dòng là transcript tương ứng với một file audio
- sắp xếp các file audio theo tên tăng dần
- ghép từng file audio với từng dòng transcript theo thứ tự
- chuyển toàn bộ audio về chuẩn `mono, 16kHz, PCM_16 WAV`
- chia dữ liệu thành `train / dev / test`
- tạo các file `train.tsv`, `dev.tsv`, `test.tsv`

### 2.1. Đầu vào yêu cầu

Bạn cần chuẩn bị:

#### a) Thư mục audio

Ví dụ:

```bash
/home/trung/Downloads/Record/
├── 0001.m4a
├── 0002.m4a
├── 0003.m4a
└── ...
```

Các file phải được đặt tên sao cho khi sắp xếp tăng dần theo tên thì đúng thứ tự transcript.

#### b) File prompts

Ví dụ `prompts.txt`:

```text
xin chào mọi người
hôm nay trời đẹp quá
tôi đang học nhận dạng tiếng nói
```

Yêu cầu:
- mỗi dòng tương ứng đúng 1 file audio
- số dòng phải bằng số file audio
- dòng 1 khớp với file audio thứ 1 sau khi sắp xếp tên
- dòng 2 khớp với file audio thứ 2, v.v.

### 2.2. Cách chạy cơ bản

```bash
python prepare_vi_asr_corpus.py \
  --audio-dir /duong/dan/toi/thu_muc_audio \
  --prompts /duong/dan/toi/prompts.txt \
  --speaker trung \
  --output-root /duong/dan/toi/vi_asr_corpus
```

Ví dụ thực tế:

```bash
python prepare_vi_asr_corpus.py \
  --audio-dir /home/trung/Downloads/Record-20260328T033526Z-1-001/Record \
  --prompts /home/trung/Downloads/Record-20260328T033526Z-1-001/prompts.txt \
  --speaker trung \
  --output-root /home/trung/icefall/egs/vi_asr_corpus/prepared
```

### 2.3. Các tham số chính

#### `--audio-dir`
Đường dẫn tới thư mục chứa file ghi âm đầu vào.

#### `--prompts`
Đường dẫn tới file transcript, mỗi dòng ứng với một file ghi âm.

#### `--speaker`
Tên hoặc mã speaker, ví dụ:

```bash
--speaker trung
--speaker spk001
```

Tên này sẽ được dùng để tạo:
- thư mục speaker trong `audio/train`, `audio/dev`, `audio/test`
- `utt_id` dạng `trung_000001`, `trung_000002`, ...

#### `--output-root`
Thư mục đầu ra chứa corpus đã chuẩn bị.

#### `--train-ratio`, `--dev-ratio`, `--test-ratio`
Tỉ lệ chia tập dữ liệu, ví dụ:

```bash
--train-ratio 0.8 --dev-ratio 0.1 --test-ratio 0.1
```

#### `--normalize-text`
Chuẩn hóa text:
- lowercase
- chuẩn Unicode NFC
- bỏ dấu câu
- chuẩn hóa khoảng trắng

#### `--overwrite`
Xóa và tạo lại thư mục output-root. Chỉ dùng khi bạn chắc chắn muốn ghi đè.

### 2.4. Kết quả đầu ra

Sau khi chạy thành công, bạn sẽ có:

```bash
output-root/
├── audio/
│   ├── train/<speaker>/*.wav
│   ├── dev/<speaker>/*.wav
│   └── test/<speaker>/*.wav
├── transcripts/
│   ├── train.tsv
│   ├── dev.tsv
│   └── test.tsv
└── README_PREPARED.txt
```

Các file `.tsv` có dạng:

```tsv
utt_id	speaker	audio_path	text
trung_000001	trung	audio/train/trung/trung_000001.wav	xin chào mọi người
```

### 2.5. Lưu ý quan trọng

1. Không nên đặt `output-root` trùng đúng thư mục hiện tại nếu script có chế độ overwrite.
2. Số file audio phải bằng số dòng trong `prompts.txt`.
3. Nên dùng transcript đã được kiểm tra kỹ trước khi tạo corpus.
4. Với nhiều speaker, nên chuẩn bị dữ liệu từng speaker riêng hoặc dùng bản script append nếu bạn có.

## 3. Hướng dẫn dùng `run.sh`

`run.sh` là script chạy toàn bộ pipeline theo stage, giúp bạn không phải gõ tay từng lệnh.

Một pipeline chuẩn thường gồm:
1. kiểm tra dataset
2. tạo manifests
3. sửa manifests bằng `lhotse fix`
4. trích transcript train
5. train BPE
6. tính fbank
7. test tokenization
8. train model
9. decode / evaluate

### 3.1. Cách chạy tổng quát

Từ thư mục project:

```bash
cd ~/icefall/egs/vi_asr_corpus
bash run.sh
```

Hoặc chạy theo stage:

```bash
bash run.sh --stage 0 --stop-stage 5
```

Ý nghĩa:
- `--stage`: bắt đầu từ bước nào
- `--stop-stage`: dừng ở bước nào

### 3.2. Ví dụ từng giai đoạn

#### Chỉ chuẩn bị dữ liệu đến fbank

```bash
bash run.sh --stage 0 --stop-stage 5
```

#### Chỉ train model

```bash
bash run.sh --stage 7 --stop-stage 7
```

#### Chỉ decode sau khi đã có checkpoint

```bash
bash run.sh --stage 8 --stop-stage 8
```

### 3.3. Ví dụ logic thường dùng trong `run.sh`

Thông thường script sẽ có dạng:

```bash
stage=0
stop_stage=100
```

Các stage thường là:

#### Stage 0: Audit dataset
Chạy kiểm tra dữ liệu đầu vào:
- file audio có tồn tại không
- transcript có rỗng không
- sample rate có đúng không
- duration có bất thường không

#### Stage 1: Prepare manifests
Sinh:
- `train_recordings.jsonl.gz`
- `train_supervisions.jsonl.gz`
- tương tự cho dev/test

#### Stage 2: Fix manifests
Dùng:

```bash
lhotse fix ...
```

để làm sạch manifest.

#### Stage 3: Export text corpus
Gộp transcript train vào một file text dùng để huấn luyện BPE.

#### Stage 4: Train BPE
Huấn luyện SentencePiece BPE, sinh:
- `bpe.model`
- `bpe.vocab`

#### Stage 5: Compute fbank
Sinh:
- `train_cuts.jsonl.gz`
- `dev_cuts.jsonl.gz`
- `test_cuts.jsonl.gz`
- các file đặc trưng trong `fbank/`

#### Stage 6: Tokenize test
Kiểm tra token hóa với `bpe.model`.

#### Stage 7: Train
Chạy huấn luyện Zipformer.

#### Stage 8: Decode
Chạy giải mã trên tập test hoặc dev.

### 3.4. Cách chỉnh `run.sh`

Bạn thường cần chỉnh các biến này trong script:

```bash
corpus_root=$PWD
exp_dir=$PWD/ASR/zipformer/exp
bpe_dir=$PWD/data/lang_bpe_500

num_epochs=20
world_size=1
max_duration=30
```

Ý nghĩa:
- `corpus_root`: thư mục gốc project
- `exp_dir`: nơi lưu checkpoint, log, tensorboard
- `bpe_dir`: nơi chứa `bpe.model`
- `num_epochs`: số epoch train
- `world_size`: số GPU / số process DDP
- `max_duration`: tổng thời lượng tối đa mỗi batch

### 3.5. Ví dụ lệnh train thường dùng

Với dataset nhỏ, nên chạy an toàn như sau:

```bash
python ASR/zipformer/train.py \
  --world-size 1 \
  --num-epochs 2 \
  --start-epoch 1 \
  --use-fp16 0 \
  --exp-dir ASR/zipformer/exp \
  --manifest-dir ./fbank \
  --bpe-model ./data/lang_bpe_500/bpe.model \
  --max-duration 30 \
  --enable-musan 0 \
  --bucketing-sampler 0 \
  --enable-spec-aug 0
```

Giải thích thêm:
- `--bucketing-sampler 0`: nên tắt với dataset rất nhỏ
- `--enable-spec-aug 0`: tránh augmentation quá mạnh khi mới smoke test
- `--enable-musan 0`: không cần noise augmentation ở giai đoạn đầu

### 3.6. Sau khi train xong

Checkpoint sẽ thường nằm ở:

```bash
ASR/zipformer/exp/
```

Ví dụ:
- `epoch-1.pt`
- `epoch-2.pt`
- `best-train-loss.pt`
- `best-valid-loss.pt`

Lúc này bạn có thể chạy:
- `decode.py`
- `streaming_decode.py`
- `export.py`
- `export-onnx.py`

## 4. Quy trình làm việc khuyến nghị

### Trường hợp mới bắt đầu
1. chuẩn bị file audio + prompts
2. chạy `prepare_vi_asr_corpus.py`
3. chạy `run.sh` từ stage 0 đến stage 5
4. kiểm tra `fbank/`, `bpe.model`
5. train smoke test vài epoch
6. decode thử

### Trường hợp thêm speaker mới
1. thêm dữ liệu mới
2. cập nhật lại corpus
3. rebuild từ:
   - manifests
   - manifests_fixed
   - transcript_words.txt
   - bpe.model
   - fbank
4. train lại hoặc fine-tune tiếp

## 5. Một số lỗi thường gặp

### Lỗi số bucket lớn hơn số cut

Ví dụ:

```text
AssertionError: The number of buckets (30) must be smaller than or equal to the number of cuts (8)
```

Cách xử lý:
- dùng `--bucketing-sampler 0`
- hoặc giảm `--num-buckets`

### Lỗi transcript không khớp audio
Nguyên nhân:
- số dòng prompts khác số file audio
- thứ tự file audio không khớp transcript

### Lỗi output-root bị ghi đè
Nguyên nhân:
- dùng `--overwrite` không cẩn thận

## 6. Gợi ý tổ chức dữ liệu

Nên:
- dùng file audio ngắn, mỗi file một câu
- transcript sạch, nhất quán
- speaker ID rõ ràng
- dev/test khác train nếu có đủ speaker

## 7. Tóm tắt

- `prepare_vi_asr_corpus.py` dùng để chuyển dữ liệu thô thành corpus chuẩn cho ASR.
- `run.sh` dùng để chạy pipeline từng stage từ chuẩn bị dữ liệu đến train/decode.
- Với dataset nhỏ, nên tắt:
  - bucketing sampler
  - spec augment
  - musan
- Khi dữ liệu thay đổi, nên rebuild lại các artifact phụ thuộc phía sau.

## 8. Ví dụ nhanh

### Chuẩn bị corpus

```bash
python prepare_vi_asr_corpus.py \
  --audio-dir ./raw_audio \
  --prompts ./prompts.txt \
  --speaker trung \
  --output-root ./prepared \
  --normalize-text
```

### Chạy pipeline

```bash
bash run.sh --stage 0 --stop-stage 5
```

### Train smoke test

```bash
python ASR/zipformer/train.py \
  --world-size 1 \
  --num-epochs 2 \
  --start-epoch 1 \
  --use-fp16 0 \
  --exp-dir ASR/zipformer/exp \
  --manifest-dir ./fbank \
  --bpe-model ./data/lang_bpe_500/bpe.model \
  --max-duration 30 \
  --enable-musan 0 \
  --bucketing-sampler 0 \
  --enable-spec-aug 0
```
