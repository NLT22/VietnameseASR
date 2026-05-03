# Theory Code Practical — Pipeline ASR (vi_asr_corpus)

Tài liệu này giải thích pipeline từ góc độ **code thực tế** trong project.  
Mọi tên file, class, hàm, tham số đều lấy từ code hiện tại.

---

## Mục lục

1. [Map pipeline → file code](#1-map-pipeline--file-code)
2. [Dữ liệu và manifest](#2-dữ-liệu-và-manifest)
3. [Feature extraction (Fbank)](#3-feature-extraction-fbank)
4. [Tokenizer / BPE](#4-tokenizer--bpe)
5. [Kiến trúc model trong code](#5-kiến-trúc-model-trong-code)
6. [Hàm loss trong code](#6-hàm-loss-trong-code)
7. [Training loop](#7-training-loop)
8. [Checkpoint và average nhiều epoch](#8-checkpoint-và-average-nhiều-epoch)
9. [Decode trong code](#9-decode-trong-code)
10. [Output và tính WER](#10-output-và-tính-wer)
11. [Debug thực tế](#11-debug-thực-tế)
12. [Ví dụ minh họa xuyên suốt](#12-ví-dụ-minh-họa-xuyên-suốt)
13. [Checklist chạy pipeline](#13-checklist-chạy-pipeline)

---

## 1. Map pipeline → file code

| Bước | File chính | Vai trò |
|---|---|---|
| Chuẩn bị audio/transcript | `prepare_vi_asr_corpus.py` | Tạo `audio/`, `transcripts/` từ raw data |
| Tạo manifest | `local/prepare_manifests.py` | TSV → lhotse `.jsonl.gz` recordings + supervisions |
| `lhotse fix` | `run.sh` stage 4 | Sửa offset, duration không hợp lệ |
| Export text corpus | `local/export_text_corpus.py` | Gom toàn bộ text train → file .txt |
| Train BPE | `local/train_bpe_model.py` | Train SentencePiece BPE model |
| Tạo lang dir | `local/prepare_lang_bpe.py` | Tạo `tokens.txt`, `words.txt` |
| Compute fbank | `local/compute_fbank.py` | Tạo `*_cuts.jsonl.gz` + feature trong `fbank/` |
| Compute fbank MUSAN | `local/compute_fbank_musan.py` | Feature cho MUSAN noise (online CutMix) |
| Validate manifest | `local/validate_manifest.py` | Kiểm tra cuts hợp lệ |
| Train | `ASR/zipformer/train.py` | Huấn luyện từ đầu |
| Finetune | `ASR/zipformer/finetune.py` | Load encoder pretrained, train lại |
| Decode | `ASR/zipformer/decode.py` | Greedy / beam / modified_beam |
| Streaming decode | `ASR/zipformer/streaming_decode.py` | Decode với chunk causal |
| CTC decode | `ASR/zipformer/ctc_decode.py` | Decode chỉ dùng CTC head |
| CTC align | `ASR/zipformer/ctc_align.py` | Forced alignment dùng CTC |
| Export JIT | `ASR/zipformer/export.py` | Export TorchScript `.pt` |
| Export ONNX | `ASR/zipformer/export-onnx.py` | Export ONNX transducer |
| Export ONNX CTC | `ASR/zipformer/export-onnx-ctc.py` | Export ONNX CTC head |
| Inference JIT | `ASR/zipformer/jit_pretrained.py` | Inference từ `.pt`, không cần lhotse |
| Inference JIT streaming | `ASR/zipformer/jit_pretrained_streaming.py` | Inference streaming từ `.pt` |
| Average checkpoint | `ASR/zipformer/generate_averaged_model.py` | Average N epoch → 1 file `.pt` |

### Cấu trúc thư mục output

```
vi_asr_corpus/
├── data/
│   ├── manifests/fixed/          # recordings*.jsonl.gz, supervisions*.jsonl.gz
│   └── lang_bpe_100/             # bpe.model, tokens.txt, words.txt
├── fbank/                        # train_cuts.jsonl.gz, dev_cuts.jsonl.gz, test_cuts.jsonl.gz
│   └── data/                     # feature storage (lhotse Lilcom/Arrow)
└── ASR/zipformer/
    └── exp_bpe100_small/         # epoch-1.pt, epoch-2.pt, ..., pretrained.pt
        └── log/                  # tensorboard, training log
```

---

## 2. Dữ liệu và manifest

### Cấu trúc dataset mong đợi

```
audio/
  train/  dev/  test/   ← file .wav hoặc .flac (16kHz mono khuyến nghị)
transcripts/
  train.tsv  dev.tsv  test.tsv
```

Format TSV (do `prepare_vi_asr_corpus.py` tạo ra):
```
recording_id  audio_path  transcript  duration_seconds
```

### Manifest lhotse

File `local/prepare_manifests.py` đọc TSV → tạo:
- `recordings_train.jsonl.gz` — danh sách audio (path, sample_rate, duration)
- `supervisions_train.jsonl.gz` — danh sách văn bản (recording_id, text, start, duration)

Sau `lhotse fix` (stage 4) → lưu vào `data/manifests/fixed/`.

### Lỗi thường gặp

| Lỗi | Nguyên nhân | Xử lý |
|---|---|---|
| `Recording not found` | Sai đường dẫn audio | Kiểm tra `audio_path` trong TSV |
| `duration mismatch` | Duration trong TSV ≠ thực tế | Chạy `lhotse fix` lại |
| `empty text` | Transcript rỗng | Lọc trước khi tạo manifest |
| Encoding lỗi | Unicode tiếng Việt | Đảm bảo TSV UTF-8 |

---

## 3. Feature extraction (Fbank)

### File: `local/compute_fbank.py`

- Đọc manifest từ `data/manifests/fixed/`
- Tính Fbank 80-dim (mặc định), frame 25ms, shift 10ms, 16kHz
- Lưu cuts vào `fbank/train_cuts.jsonl.gz`, feature vào `fbank/data/`

### Vì sao dùng Fbank thay waveform?

Waveform thô (hàng chục nghìn điểm/giây) quá dài cho Transformer. Fbank nén thông tin âm thanh thành spectrogram log-mel: giữ thông tin tần số quan trọng, loại bỏ nhiễu ít ảnh hưởng đến giọng nói.

### Tham số quan trọng

| Tham số | Giá trị mặc định | Ý nghĩa |
|---|---|---|
| `num_mel_bins` | 80 | Số bank tần số mel |
| `frame_length` | 25ms | Độ dài cửa sổ |
| `frame_shift` | 10ms | Bước dịch |
| `sample_rate` | 16000 | Hz |

Sau subsampling × 4 trong model: 1 giây audio ≈ 25 frame encoder.

---

## 4. Tokenizer / BPE

### File: `local/train_bpe_model.py`

Dùng SentencePiece BPE. Lệnh từ `run.sh`:
```bash
python3 local/train_bpe_model.py --vocab-size 100 \
  --input data/lang_bpe_100/text_corpus.txt \
  --model-prefix data/lang_bpe_100/bpe
```

### File: `local/prepare_lang_bpe.py`

Tạo:
- `tokens.txt` — mapping token → integer id
- `words.txt` — vocabulary từ transcript

### Token đặc biệt

| Token | ID thường gặp | Vai trò |
|---|---|---|
| `<blk>` | 0 | CTC blank, không phát âm |
| `<sos/eos>` | cuối | Bắt đầu/kết thúc câu (attention decoder) |
| `<unk>` | - | Từ chưa biết |

### Ảnh hưởng vocab_size

- `vocab_size=100`: token nhỏ (âm tiết/ký tự), ít OOV, train nhanh với dataset nhỏ
- `vocab_size=500+`: subword dài hơn, cần nhiều data hơn
- Project này mặc định dùng `100` (phù hợp tiếng Việt âm tiết rõ)

---

## 5. Kiến trúc model trong code

### File: `ASR/zipformer/model.py`

**Class chính:** `AsrModel(nn.Module)`

```python
AsrModel(
  encoder_embed,   # Conv2dSubsampling (subsampling.py)
  encoder,         # Zipformer2 (zipformer.py)
  decoder,         # RNN Predictor (decoder.py)
  joiner,          # Linear Joiner (joiner.py)
  attention_decoder,  # optional (attention_decoder.py)
  encoder_dim=256,
  decoder_dim=512,
  vocab_size=100,
  use_transducer=True,
  use_ctc=False,
)
```

### Luồng tensor

```
audio waveform
  → Fbank 80-dim: (N, T, 80)
  → Conv2dSubsampling × 4: (N, T/4, encoder_dim)
  → Zipformer2 encoder: (N, T', encoder_dim)   ← T' ≈ T/4

[Transducer path]
  encoder_out → joiner(encoder_out, decoder_out) → log_softmax → vocab_size
  decoder_out từ: Predictor(y_prev_token) [RNN 2-layer, context_size=2]

[CTC path] (nếu use_ctc=True)
  encoder_out → linear → log_softmax → vocab_size
```

### Zipformer (encoder)

File `zipformer.py` — mô hình Transformer biến thể với:
- Nhiều scale (6 stack với downsampling 1,2,4,8,4,2)
- Relative position encoding
- Causal mode: `--causal 1` + `--chunk-size` → streaming

### Model presets (run.sh)

| `--model_size` | Encoder dims | Params |
|---|---|---|
| `base` | `192,256,384,512,384,256` | ~65M |
| `small` | `192,256,256,256,256,256` | ~23M |

---

## 6. Hàm loss trong code

### File: `ASR/zipformer/train.py`, hàm `compute_loss()`

```python
def compute_loss(params, model, sp, batch, is_training, spec_augment):
    feature = batch["inputs"]        # (N, T, 80)
    texts = batch["supervisions"]["text"]
    y = sp.encode(texts, out_type=int)  # tokenize transcript
    loss, loss_info = model(x, x_lens, y, ...)
    return loss, loss_info
```

### Các loss có trong `model.forward()`

| Loss | Flag | Ý nghĩa |
|---|---|---|
| **Pruned Transducer loss** | `use_transducer=True` (mặc định) | Loss chính, hiệu quả hơn full transducer |
| **Simple Transducer loss** | cùng flag, scale `simple_loss_scale` | Warmup, giảm dần theo `warm_step` |
| **CTC loss** | `--use-ctc 1` | Auxiliary, hội tụ nhanh hơn |
| **CR-CTC loss** | `--use-cr-ctc 1` | Consistency regularization CTC (tốt cho tiếng Việt có dấu) |
| **Attention Decoder loss** | `--use-attention-decoder 1` | Encoder–decoder cross-attention |

### Pruned vs Simple transducer

- **Simple**: tính trên toàn bộ (T × U) lattice — đúng nhưng tốn RAM
- **Pruned**: chỉ tính trên dải hẹp quanh alignment tốt nhất — nhanh hơn ~4×
- Warmup: `simple_loss_scale` bắt đầu = 0.5, sau `warm_step` batches giảm về 0

### Loss vs WER

Loss là xác suất log trên training set. WER đo lỗi **từ** trên test set.  
Loss thấp chưa chắc WER thấp vì:
- Overfit: loss train thấp, test cao
- Tokenization mismatch: model đúng token nhưng ghép sai từ
- Decode method: cùng model, greedy có WER cao hơn beam search

---

## 7. Training loop

### Optimizer & scheduler

```python
# optim.py
optimizer = ScaledAdam(model.parameters(), lr=params.base_lr, ...)
scheduler = Eden(optimizer, lr_batches=7500, lr_epochs=3.5)
```

- **ScaledAdam**: Adam với per-parameter scale (ổn định hơn với model lớn)
- **Eden**: LR schedule theo số batch + số epoch, không cần warmup cố định

### Mixed precision

```python
# train.py
with torch_autocast(enabled=params.use_autocast, dtype=params.dtype):
    loss, loss_info = compute_loss(...)
scaler.scale(loss).backward()
scaler.step(optimizer)
```

`--use-fp16 1` → dtype=float16, dùng GradScaler tự động. Tăng tốc ~1.5–2× trên GPU.

### Các tham số CLI quan trọng

| Tham số | Default | Ý nghĩa |
|---|---|---|
| `--num-epochs` | 30 | Số epoch huấn luyện |
| `--start-epoch` | 1 | Resume từ epoch nào |
| `--base-lr` | 0.045 | LR nền (librispeech mặc định); project này dùng 0.01 |
| `--world-size` | 1 | Số GPU (DDP) |
| `--use-fp16` | False | Mixed precision float16 |
| `--exp-dir` | zipformer/exp | Thư mục lưu checkpoint |
| `--bpe-model` | data/lang_bpe_500/bpe.model | Tokenizer |
| `--manifest-dir` | fbank | Thư mục chứa `*_cuts.jsonl.gz` |
| `--max-duration` | 50 (giây/batch) | Tổng duration mỗi batch |
| `--enable-musan` | False | Online MUSAN noise augmentation |
| `--enable-spec-aug` | True | SpecAugment (mask time + freq) |
| `--bucketing-sampler` | True | Group audio theo duration → batch đều |
| `--causal` | False | Chế độ streaming (causal Zipformer) |
| `--chunk-size` | `16,32,64,-1` | Chunk size khi train streaming (frame) |
| `--save-every-n` | 4000 | Lưu checkpoint mỗi N batch |
| `--keep-last-k` | 30 | Giữ tối đa K checkpoint batch |
| `--average-period` | 200 | Cập nhật running average model mỗi N batch |
| `--prune-range` | 5 | Dải pruning cho pruned transducer loss |
| `--lm-scale` | 0.25 | Scale cho prediction network (LM part) |
| `--am-scale` | 0.0 | Scale cho acoustic model part |

### Checkpoint được lưu khi nào

- Cuối mỗi epoch: `exp-dir/epoch-N.pt`
- Mỗi `save_every_n` batches: `exp-dir/checkpoint-B.pt`
- `model_avg` (running average): `exp-dir/avg_model.pt`

---

## 8. Checkpoint và average nhiều epoch

### Nội dung file `.pt`

```python
{
  "model": model.state_dict(),
  "optimizer": optimizer.state_dict(),
  "scheduler": scheduler.state_dict(),
  "scaler": scaler.state_dict(),
  "params": {...},           # best_train_loss, epoch, batch_idx, ...
}
```

### Tham số decode liên quan

| Tham số | Ý nghĩa |
|---|---|
| `--epoch N` | Load `epoch-N.pt` |
| `--avg K` | Average K epoch từ N trở về trước (N, N-1, ..., N-K+1) |
| `--use-averaged-model` | Dùng `model_avg` (running average trong training) thay vì `epoch-N.pt` |

### Khi nào nên dùng avg

| Kịch bản | Khuyến nghị |
|---|---|
| Debug nhanh | `--avg 1` (chỉ epoch cuối) |
| Báo cáo kết quả cuối | `--avg 5` hoặc `--avg 10` |
| Dataset nhỏ, dao động lớn | `--avg 10–20` |
| Model chưa hội tụ (loss vẫn giảm) | Tránh avg epoch quá cũ |

### Script average thủ công

```bash
python3 ASR/zipformer/generate_averaged_model.py \
  --epoch 50 --avg 10 \
  --exp-dir ASR/zipformer/exp_bpe100_small \
  --tokens data/lang_bpe_100/tokens.txt
```

---

## 9. Decode trong code

### File: `ASR/zipformer/decode.py`

#### Bảng so sánh decode method

| Method | Hàm (beam_search.py) | Tốc độ | Độ chính xác | Khi nào dùng |
|---|---|---|---|---|
| `greedy_search` | `greedy_search_batch()` | Nhanh nhất | Thấp nhất | Debug, thử nhanh |
| `modified_beam_search` | `modified_beam_search()` | Trung bình | Tốt | Báo cáo thông thường |
| `beam_search` | `beam_search()` | Chậm | Tốt | Khi cần reference |
| `fast_beam_search` | `fast_beam_search_one_best()` | Nhanh | Tốt | Khi có k2 graph |

**Khuyến nghị project nhỏ**: dùng `modified_beam_search` để báo cáo.

#### Tham số decode (CLI)

```bash
python3 ASR/zipformer/decode.py \
  --epoch 50 --avg 5 \
  --exp-dir ASR/zipformer/exp_bpe100_small \
  --manifest-dir fbank \
  --bpe-model data/lang_bpe_100/bpe.model \
  --max-duration 100 \
  --decoding-method modified_beam_search \
  --beam-size 4
```

### File: `ASR/zipformer/streaming_decode.py`

Decode với `--causal 1`. Thêm tham số:
- `--chunk-size` — số frame mỗi chunk (32 ~ 320ms)
- `--left-context-frames` — context nhìn về trái (256 ~ 2.56s)
- `--decoding-method greedy_search` (streaming không hỗ trợ full beam)

### File: `ASR/zipformer/ctc_decode.py`

Chỉ dùng CTC head (encoder + linear + softmax), không cần predictor/joiner.  
Cần `--use-ctc 1` khi train để có CTC head trong model.

---

## 10. Output và tính WER

### Output decode

File kết quả lưu trong `exp-dir/`:
```
recogs-test-modified_beam_search.txt     ← (ref, hyp) từng câu
wer-summary-test-modified_beam_search.txt ← WER tổng
```

Format `recogs-*.txt`:
```
utt_id  ref: hôm nay trời đẹp  hyp: hôm nay trời đẹp
```

### Tính WER

```python
# icefall/utils.py: write_error_stats()
wer = write_error_stats(f, test_set_name, results)
```

WER = (S + D + I) / N  
- S: substitution (sai từ), D: deletion (mất từ), I: insertion (thêm từ)
- N: tổng số từ trong reference

### Lưu ý tiếng Việt

| Vấn đề | Ví dụ lỗi |
|---|---|
| Sai dấu thanh | "trời" → "troi", "tròi" |
| Ghép/tách âm tiết | "hôm nay" → "hômnay" |
| Lặp token | "đẹp đẹp" thay vì "đẹp" |
| BPE vocab nhỏ | "nghiêng" split thành nhiều piece, ghép lại sai |

Với `vocab_size=100`, mỗi âm tiết thường là 1 token → ít lỗi ghép.

---

## 11. Debug thực tế

| Lỗi | Nguyên nhân | Hướng xử lý |
|---|---|---|
| `FileNotFoundError: *_cuts.jsonl.gz` | Chưa chạy compute_fbank | Chạy stage 8 trong run.sh |
| `FileNotFoundError: bpe.model` | Chưa train BPE | Chạy stage 6-7 |
| `RuntimeError: exp-dir not found` | `--exp-dir` sai | Kiểm tra đường dẫn |
| CUDA OOM | `--max-duration` quá lớn | Giảm xuống 30–40 |
| `Loss is nan` | LR quá cao hoặc batch bad | Giảm `--base-lr`, kiểm tra data |
| WER = 100% | Model chưa hội tụ hoặc tokenizer sai | Train thêm epoch, kiểm tra BPE |
| Decode không ra chữ | Thiếu `--bpe-model` | Truyền đúng đường dẫn |
| Model overfit | Val loss tăng, train loss giảm | Tăng SpecAugment, dùng MUSAN, giảm epoch |
| `checkpoint not found` | `--epoch` lớn hơn số epoch đã train | Giảm `--epoch` |
| WER cao với `greedy_search` | Đúng bình thường | Thử `modified_beam_search` |

---

## 12. Ví dụ minh họa xuyên suốt

**Input:** file audio `test_001.wav` với transcript "hôm nay trời đẹp"

### Bước 1 — Prepare manifest

```
recordings_test.jsonl.gz:
  { "id": "test_001", "path": "audio/test/test_001.wav", "duration": 1.8, "sampling_rate": 16000 }

supervisions_test.jsonl.gz:
  { "id": "test_001-0", "recording_id": "test_001", "text": "hôm nay trời đẹp", "start": 0, "duration": 1.8 }
```

### Bước 2 — Compute fbank

```
fbank/test_cuts.jsonl.gz:
  cut chứa feature 80-dim, T ≈ 180 frames (1.8s × 100fps)
```

### Bước 3 — Tokenize (BPE vocab=100)

```
"hôm nay trời đẹp"  →  [12, 34, 67, 89]  (4 token, mỗi âm tiết 1 token)
```

### Bước 4 — Forward pass

```
feature (1, 180, 80)
→ Conv2dSubsampling: (1, 45, 256)
→ Zipformer encoder: (1, 45, 256)
→ Predictor([<sos>]): hidden
→ Joiner(encoder_out, decoder_out) × 45 × 5 (prune_range) → log_prob
```

### Bước 5 — Loss

Pruned transducer loss so khớp alignment `[12, 34, 67, 89]` với output joiner.

### Bước 6 — Decode (modified_beam_search)

```
encoder_out (1, 45, 256)
→ beam_size=4, giữ 4 hypothesis mỗi bước
→ best hyp token ids: [12, 34, 67, 89]
→ sp.decode([12, 34, 67, 89]) = "hôm nay trời đẹp"
```

### Bước 7 — WER

```
ref: hôm nay trời đẹp  (4 từ)
hyp: hôm nay trời đẹp  (4 từ)
S=0 D=0 I=0 → WER = 0%
```

---

## 13. Checklist chạy pipeline

### Chuẩn bị

- [ ] `audio/train/`, `audio/dev/`, `audio/test/` có đầy đủ file
- [ ] `transcripts/train.tsv`, `dev.tsv`, `test.tsv` đúng format, UTF-8
- [ ] Không có transcript rỗng
- [ ] Sample rate 16kHz (hoặc resample trước)

### Data prep (stage 1–12)

- [ ] `bash run.sh --stage 1 --stop_stage 1` — audit dataset
- [ ] `bash run.sh --stage 3 --stop_stage 4` — manifest + fix
- [ ] `bash run.sh --stage 5 --stop_stage 7` — BPE vocab=100
- [ ] `bash run.sh --stage 8 --stop_stage 9` — fbank + musan fbank
- [ ] `bash run.sh --stage 10 --stop_stage 12` — validate + tokenize test

### Train

- [ ] `bash run.sh --model_size small --num_epochs 50 --stage 13 --stop_stage 13`
- [ ] Monitor loss: `tensorboard --logdir ASR/zipformer/exp_bpe100_small/log`
- [ ] Kiểm tra loss không NaN sau epoch 1-2

### Decode

- [ ] `bash run.sh --model_size small --num_epochs 50 --stage 14 --stop_stage 14`
- [ ] Đọc kết quả: `cat ASR/zipformer/exp_bpe100_small/wer-summary-test-*.txt`
- [ ] So sánh greedy vs modified_beam_search

### Báo cáo

- [ ] Ghi lại WER cho từng decode method
- [ ] So sánh avg=1 vs avg=5 vs avg=10
- [ ] Ghi config: model_size, vocab_size, num_epochs, decode_method
