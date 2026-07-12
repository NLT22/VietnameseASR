# VietnameseASR

## Current matched ASR target

This folder shares the same project structure as `egs/vi_asr_corpus`, but uses
the VietnameseASR dataset. The active demo target is a real ASR model, not a
template matcher:

- BPE vocab size 100
- small Zipformer transducer
- non-streaming ONNX first
- streaming ONNX/CUDA/TensorRT as the optimization path
- matched train/dev/test splits for intentional "hoc vet" ASR evaluation

Prepare matched manifests/features without starting GPU training:

```bash
bash run_matched_asr.sh --stage 0 --stop-stage 8
```

When the GPU is free, train/evaluate non-streaming:

```bash
bash run_matched_asr.sh --stage 13 --stop-stage 14 --streaming 0 --avg 5
```

Train/evaluate streaming for the TensorRT investigation path:

```bash
bash run_matched_asr.sh --stage 13 --stop-stage 14 --streaming 1 --avg 1
```

After choosing the best WER checkpoint, export a Jetson ONNX package:

```bash
bash local/export_for_jetson.sh --exp-dir ASR/zipformer/exp_bpe100_small_raw_matched --epoch 50 --avg 5 --streaming 0
```

See `docs/PROJECT_NOTES.md` for old result interpretation and Jetson/TensorRT notes.

## Cấu trúc

```
VietnameseASR/
├── ASR/zipformer/          # train.py, decode.py, export.py, finetune.py, ...
├── local/                  # scripts chuẩn bị data
├── audio/  audio_nr/
├── transcripts/  transcripts_nr/
├── fbank/  fbank_nr/
├── data/
│   ├── manifests/fixed/
│   ├── manifests_nr/fixed/
│   └── lang_bpe_<vocab>/
├── docs/                   # theory + presentation notes
├── run.sh                  # single pipeline entry point (bash run.sh --help)
└── run_x10.sh              # thin backward-compat wrapper around run.sh
```

---

## Luồng chạy nhanh

```bash
cd /path/to/icefall/egs/vi_asr_corpus

# 1. Tạo audio/ + transcripts/ từ raw data
python local/prepare_vi_asr_corpus.py --auto --shuffle-before-split --overwrite

# 2. Chuẩn bị data → fbank (stage 1–12)
bash run.sh --vocab_size 100 --stage 1 --stop_stage 12

# 3. Train
bash run.sh --vocab_size 100 --model_size small --num_epochs 50 --stage 13 --stop_stage 13

# 4. Decode (chạy greedy + modified_beam + beam_search)
bash run.sh --vocab_size 100 --model_size small --num_epochs 50 --stage 14 --stop_stage 14
```

---

## run.sh — Stages

| Stage | Việc làm |
|---|---|
| -1 | Chuẩn bị `audio/` + `transcripts/` từ `dataset/` với split mặc định 80/10/10 |
| 0 | Tạo NoiseReduce variant `nr` (cần `--enable_nr 1`) |
| 1 | Audit dataset |
| 2 | Offline MUSAN augmentation |
| 3 | Prepare manifests |
| 4 | `lhotse fix` |
| 5 | Export text corpus |
| 6 | Train BPE |
| 7 | Prepare BPE lang dir |
| 8 | Compute fbank / cuts |
| 9 | Compute MUSAN fbank (online CutMix) |
| 10 | Validate cut manifests |
| 11 | Display manifest statistics |
| 12 | Tokenize smoke test |
| 13 | Train |
| 14 | Decode |

## run.sh — Tham số chính

```bash
--vocab_size 100
--train_ratio 0.8
--dev_ratio 0.1
--test_ratio 0.1
--model_size medium|small        # small = Zipformer S ~23M; medium = Zipformer M
--num_epochs 40
--world_size 1
--max_duration 500
--base_lr 0.045
--use_fp16 0|1

# Augmentation
--enable_spec_aug 0|1
--enable_musan 0|1               # online MUSAN
--offline_musan_aug 0|1          # offline MUSAN (stage 2)
--musan_dir /path/to/musan
--copies_per_utt 3
--snr_min 10  --snr_max 20
--perturb_speed 0|1

# Loss (train.py hỗ trợ, mặc định chỉ dùng transducer)
--use_ctc 0|1
--use_cr_ctc 0|1                 # CR-CTC, đáng thử với tiếng Việt có tone
--ctc_loss_scale 0.2
--cr_loss_scale 0.2
--attention_decoder_loss_scale 0.0

# Decode
--decode_method all|greedy_search|modified_beam_search|beam_search
--use_averaged_model 0|1
--avg 1

# Finetune
--do_finetune 0|1
--finetune_ckpt /path/to/epoch-N.pt
--init_modules encoder           # khuyến nghị: chỉ load encoder từ pretrained

# Data variant
--data_variant raw|nr
--enable_nr 0|1

# Experiment dir suffix
--exp_suffix _myexp
--exp_dir runs                   # đổi thư mục cha lưu experiment
--exp_dir_policy auto|reuse|fail
```

### Model size presets

`--model_size medium --data_variant raw` → `exp_bpe100_medium_raw/` (Zipformer M)

`--model_size small --data_variant raw` → `exp_bpe100_small_raw/`
```
--num-encoder-layers 2,2,2,2,2,2   --feedforward-dim 512,768,768,768,768,768
--num-heads 4,4,4,8,4,4            --encoder-dim 192,256,256,256,256,256
--encoder-unmasked-dim 192,192,192,192,192,192
--decoder-dim 512  --joiner-dim 512
```

> Giải thích chi tiết từng tham số + bảng scale **S / M / L** (small = S, base = M):
> xem `docs/TEACHING_NOTES_VI_chi_tiet.md` mục "Kích thước model".

Hậu tố exp dir tự động: `_small`, `_raw`, `_nr`, `_streaming`, `_finetune` (kết hợp được). Các kết quả cũ như `exp_bpe100_small/` vẫn còn nguyên, nhưng lần chạy mới mặc định sẽ ghi sang folder có suffix variant để tránh nhầm raw/nr.

Nếu chỉ muốn đổi nơi lưu kết quả nhưng vẫn giữ tên experiment chứa đủ thông tin model/data/finetune, dùng `--exp_dir`. Tham số này là thư mục cha, nhận cả relative path và absolute path:

```bash
bash run.sh \
  --model_size small \
  --data_variant raw \
  --exp_dir runs \
  --stage 13 --stop_stage 14
```

Lệnh trên sẽ lưu vào `runs/exp_bpe100_small_raw/`, không mất các hậu tố `_small`, `_raw`, `_nr`, `_finetune`. Mặc định `--exp_dir_policy auto`: nếu chạy stage train và experiment dir đã có file, `run.sh` tự chuyển sang folder mới dạng `exp_..._YYYYmmdd_HHMMSS` để tránh ghi đè checkpoint/log cũ. Dùng `reuse` để ghi tiếp vào folder cũ, hoặc `fail` để dừng nếu folder đã có dữ liệu.

---

## Finetune từ pretrained LibriSpeech

```bash
# Tải checkpoint
git lfs install
git clone https://huggingface.co/Zengwei/icefall-asr-librispeech-zipformer-small-2023-05-16 \
  pretrained/zipformer_small_ls960

# Finetune encoder
bash run.sh \
  --model_size small --num_epochs 30 --base_lr 0.001 \
  --do_finetune 1 \
  --finetune_ckpt pretrained/zipformer_small_ls960/exp/epoch-50.pt \
  --init_modules encoder \
  --stage 13 --stop_stage 14
```

| model_size | Pretrained |
|---|---|
| `base` | `Zengwei/icefall-asr-librispeech-zipformer-2023-05-15` (~65M params) |
| `small` | `Zengwei/icefall-asr-librispeech-zipformer-small-2023-05-16` (~23M params) |

---

## Scripts trong `local/`

| File | Việc làm |
|---|---|
| `prepare_vi_asr_corpus.py` | Raw `dataset/` → `audio/` + `transcripts/` (stage -1) |
| `audit_dataset.py` | Kiểm tra transcript/audio khớp nhau (stage 1) |
| `augment_train_with_musan.py` | Offline MUSAN augmentation cho train split (stage 2) |
| `prepare_manifests.py` | TSV → lhotse recordings/supervisions jsonl.gz |
| `export_text_corpus.py` | Gom text train để train BPE |
| `train_bpe_model.py --vocab-size N` | Train SentencePiece BPE |
| `prepare_lang_bpe.py --lang-dir data/lang_bpe_N` | Tạo tokens.txt, words.txt |
| `compute_fbank.py` | Tạo `*_cuts.jsonl.gz` + features trong `fbank/` |
| `compute_fbank_musan.py` | Tạo `musan_cuts.jsonl.gz` cho online CutMix |
| `validate_manifest.py --all` | Kiểm tra cut manifests |
| `display_manifest_statistics.py --all` | In thống kê (duration, số câu,...) |
| `tokenize_test.py --vocab-size N` | Smoke test tokenizer |
| `noise_reduce_audio.py` | Tạo `audio_nr/` + `transcripts_nr/` bằng noisereduce |

## Scripts trong `ASR/zipformer/`

| File | Việc làm |
|---|---|
| `train.py` | Train from scratch |
| `finetune.py` | Finetune từ pretrained checkpoint |
| `decode.py` | Decode (greedy / beam / modified_beam / fast_beam + LG) |
| `export-onnx.py` | Export transducer → ONNX chi tiết hơn |
| `export-onnx-streaming.py` | Export streaming transducer → ONNX |
| `generate_averaged_model.py` | Average N checkpoint offline → file `.pt` duy nhất |
| `onnx_pretrained.py` | Inference từ ONNX transducer |
| `onnx_pretrained-streaming.py` | Inference từ ONNX streaming |
| `onnx_decode.py` | Decode tập test dùng ONNX |
| `pretrained.py` | Inference từ `.pt` checkpoint thường |
| `streaming_decode.py` | Decode streaming |

---

## Các ví dụ hay dùng

```bash
# NoiseReduce variant
bash run.sh --enable_nr 1 --data_variant nr --model_size small \
  --num_epochs 50 --stage 0 --stop_stage 14

# Offline MUSAN aug → train
bash run.sh --offline_musan_aug 1 --musan_dir "$MUSAN_DIR" \
  --copies_per_utt 3 --enable_musan 0 --stage 2 --stop_stage 13

# Average checkpoint thủ công
python3 ASR/zipformer/generate_averaged_model.py \
  --epoch 50 --avg 10 \
  --exp-dir ASR/zipformer/exp_bpe100_small_raw \
  --tokens data/lang_bpe_100/tokens.txt

```

## NoiseReduce dependency (tùy chọn)

```bash
pip install noisereduce
```

Nếu thiếu, `local/noise_reduce_audio.py` sẽ báo lỗi rõ ràng.

## MUSAN manifests

```bash
lhotse prepare musan "$MUSAN_DIR" data/manifests
```
