# vi_asr_corpus

## Cấu trúc

```
vi_asr_corpus/
├── ASR/zipformer/          # train.py, decode.py, export.py, finetune.py, ...
├── local/                  # scripts chuẩn bị data
├── audio/  audio_nr/
├── transcripts/  transcripts_nr/
├── fbank/  fbank_nr/
├── data/
│   ├── manifests/fixed/
│   ├── manifests_nr/fixed/
│   └── lang_bpe_<vocab>/
├── prepare_vi_asr_corpus.py
├── augment_train_with_musan.py
└── run.sh
```

---

## Luồng chạy nhanh

```bash
cd /path/to/icefall/egs/vi_asr_corpus

# 1. Tạo audio/ + transcripts/ từ raw data
python prepare_vi_asr_corpus.py --auto --shuffle-before-split --overwrite

# 2. Chuẩn bị data → fbank (stage 1–12)
bash run.sh --vocab_size 100 --stage 1 --stop_stage 12

# 3. Train raw small model
bash run.sh --data_variant raw --vocab_size 100 --model_size small \
  --exp_suffix _scratch30 --num_epochs 30 --stage 13 --stop_stage 13

# 4. Decode (chạy greedy + modified_beam + beam_search)
bash run.sh --data_variant raw --vocab_size 100 --model_size small \
  --exp_suffix _scratch30 --num_epochs 30 --decode_method all \
  --stage 14 --stop_stage 14
```

---

## run.sh — Stages

| Stage | Việc làm |
|---|---|
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
--model_size base|small          # small = official Zipformer-small ~23.3M params
--num_epochs 50
--world_size 1
--max_duration 50
--base_lr 0.01
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
--exp_dir_policy auto|reuse|fail
```

### Model size presets

`--model_size base --data_variant raw` → `exp_bpe100_raw/`

`--model_size base --data_variant nr` → `exp_bpe100_nr/`

`--model_size small --data_variant raw` → `exp_bpe100_small_raw/`

`--model_size small --data_variant nr` → `exp_bpe100_small_nr/`
```
--num-encoder-layers 2,2,2,2,2,2   
--feedforward-dim 512,768,768,768,768,768
--num-heads 4,4,4,8,4,4            
--encoder-dim 192,256,256,256,256,256
--encoder-unmasked-dim 192,192,192,192,192,192
--decoder-dim 512  
--joiner-dim 512
```

Hậu tố exp dir tự động: `_raw` hoặc `_nr`, thêm `_streaming` nếu `--causal 1`, và thêm `--exp_suffix` nếu có. Ví dụ:

| Lệnh chính | Exp dir mặc định |
|---|---|
| `--model_size small --data_variant raw --exp_suffix _scratch30` | `ASR/zipformer/exp_bpe100_small_raw_scratch30` |
| `--model_size small --data_variant nr --exp_suffix _scratch30` | `ASR/zipformer/exp_bpe100_small_nr_scratch30` |
| `--model_size base --data_variant raw --exp_suffix _scratch50` | `ASR/zipformer/exp_bpe100_raw_scratch50` |

Mặc định `--exp_dir_policy auto`: nếu chạy stage train và `exp_dir` đã có file, `run.sh` tự chuyển sang folder mới dạng `exp_..._YYYYmmdd_HHMMSS` để tránh ghi đè checkpoint/log cũ. Nếu muốn hành vi cũ thì dùng `--exp_dir_policy reuse`; nếu muốn dừng ngay khi folder đã tồn tại thì dùng `--exp_dir_policy fail`.

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
| `prepare_manifests.py` | TSV → lhotse recordings/supervisions jsonl.gz |
| `export_text_corpus.py` | Gom text train để train BPE |
| `train_bpe_model.py --vocab-size N` | Train SentencePiece BPE |
| `prepare_lang_bpe.py --lang-dir data/lang_bpe_N` | Tạo tokens.txt, words.txt |
| `compute_fbank.py` | Tạo `*_cuts.jsonl.gz` + features trong `fbank/` |
| `compute_fbank_musan.py` | Tạo `musan_cuts.jsonl.gz` cho online CutMix |
| `filter_cuts.py` | Lọc cuts quá ngắn/dài hoặc T < S |
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
| `export.py` | Export checkpoint → ONNX hoặc TorchScript (`--jit 1`) |
| `export-onnx.py` | Export transducer → ONNX chi tiết hơn |
| `export-onnx-streaming.py` | Export streaming transducer → ONNX |
| `generate_averaged_model.py` | Average N checkpoint offline → file `.pt` duy nhất |
| `jit_pretrained.py` | Inference từ TorchScript `.pt`, không cần lhotse |
| `onnx_pretrained.py` | Inference từ ONNX transducer |
| `onnx_pretrained-streaming.py` | Inference từ ONNX streaming |
| `onnx_check.py` | Kiểm tra ONNX output khớp PyTorch |
| `onnx_decode.py` | Decode tập test dùng ONNX |
| `pretrained.py` | Inference từ `.pt` checkpoint thường |
| `streaming_decode.py` | Decode streaming |

---

## Các ví dụ hay dùng

```bash
# NoiseReduce variant
bash run.sh --enable_nr 1 --stage 0 --stop_stage 0

bash run.sh --data_variant nr --model_size small \
  --exp_suffix _scratch30 --num_epochs 30 --stage 1 --stop_stage 14

# Offline MUSAN aug → train
bash run.sh --offline_musan_aug 1 --musan_dir "$MUSAN_DIR" \
  --copies_per_utt 3 --enable_musan 0 --stage 2 --stop_stage 13

# Train với CTC phụ trợ
bash run.sh --model_size small --use_ctc 1 --ctc_loss_scale 0.2 \
  --stage 13 --stop_stage 13

# Train với CR-CTC (consistency regularization)
bash run.sh --model_size small --use_ctc 1 --use_cr_ctc 1 \
  --stage 13 --stop_stage 13

# Average checkpoint thủ công
python3 ASR/zipformer/generate_averaged_model.py \
  --epoch 50 --avg 10 \
  --exp-dir ASR/zipformer/exp_bpe100_small \
  --tokens data/lang_bpe_100/tokens.txt

# Lọc cuts trước train
python3 local/filter_cuts.py \
  --bpe-model data/lang_bpe_100/bpe.model \
  --in-cuts fbank/train_cuts.jsonl.gz \
  --out-cuts fbank/train_cuts_filtered.jsonl.gz

# Inference nhanh từ JIT model (không cần lhotse)
python3 ASR/zipformer/jit_pretrained.py \
  --nn-model-filename ASR/zipformer/exp_bpe100_small/jit_script.pt \
  --tokens data/lang_bpe_100/tokens.txt \
  /path/to/audio.wav

# Microphone demo dùng JIT best hiện tại nếu file tồn tại
python3 mic_streaming_asr.py --mode full --decode-method beam --beam 4
```

## Dọn checkpoint để tiết kiệm dung lượng

Mặc định `run_all_experiments.sh` và `local/run_experiment_suite.sh` sẽ dọn checkpoint sau mỗi experiment nếu `CLEAN_CHECKPOINTS=1`.

Rule giữ lại:
- `epoch-30.pt`
- `epoch-50.pt`
- `best-*.pt`
- `pretrained.pt`, `averaged*.pt`, `jit_script.pt` nếu có

Các `epoch-*.pt` khác và `checkpoint-*.pt` sẽ bị xóa.

Chạy dry-run trước:

```bash
bash local/cleanup_checkpoints.sh --dry-run ASR/zipformer/exp_bpe100_scratch50
```

Xóa thật:

```bash
bash local/cleanup_checkpoints.sh ASR/zipformer/exp_bpe100_scratch50
```

Tắt auto-clean khi chạy experiment:

```bash
CLEAN_CHECKPOINTS=0 bash ../run_all_experiments.sh
```

---

## NoiseReduce dependency (tùy chọn)

```bash
pip install noisereduce
```

Nếu thiếu, `local/noise_reduce_audio.py` sẽ báo lỗi rõ ràng.

## MUSAN manifests

```bash
lhotse prepare musan "$MUSAN_DIR" data/manifests
```
