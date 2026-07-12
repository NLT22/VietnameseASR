# VietnameseASR

Real Vietnamese ASR with icefall (streaming Zipformer2 + Pruned Transducer).
Single recipe, one entry point (`run.sh`). Intentional "học vẹt" (memorization)
target: matched train/dev/test splits, evaluated honestly with a held-out
speaker metric.

- BPE vocab 100
- **medium (M)** streaming/causal Zipformer (small = S kept as backup)
- int8 ONNX for Jetson Nano / live-UI deployment
- 1000 real utterances (5 male speakers × 200) + Gwen-TTS voice clones

Deployed model: `ASR/zipformer/exp_bpe100_medium_streaming_main_lr0045`
(epoch 30, avg 10). Numbers live in `docs/RESULTS.md`; how it works in
`docs/TEACHING_NOTES.md` (EN) / `docs/TEACHING_NOTES_VI.md` (VI).

---

## Cấu trúc

```
VietnameseASR/
├── ASR/zipformer/                 # train.py, decode.py, export-onnx*.py, finetune.py, ...
├── local/                         # scripts chuẩn bị data + local/tts/ (Gwen-TTS clone gen)
├── dataset/                       # raw recordings (per-speaker wav + transcript)
├── transcripts_matched_u20/       # real 1000-utt matched splits (5 spk, all clips <20s)
├── transcripts_main/  fbank_main/ # deployed clone-mixed set (--data_tag main)
├── data/manifests_main/  data/lang_bpe_100/
├── lang/                          # BPE training corpus (transcript_words_<tag>.txt)
├── deploy/jetson_nano/            # int8 ONNX package + on-device beam_search decoder
├── live_ui/                       # local streaming mic UI (VAD + speaker-id)
├── docs/                          # RESULTS + teaching notes
├── run.sh                         # single pipeline entry point (bash run.sh --help)
└── run_x10.sh                     # thin backward-compat wrapper around run.sh
```

---

## Luồng chạy nhanh

`run.sh` là entry point duy nhất, chạy toàn bộ pipeline theo số stage.

```bash
cd /path/to/icefall/egs/VietnameseASR

# Full pipeline (deployed medium model): clone gen -> train -> decode -> export.
# Cần Gwen-TTS checkout cho stage -2 (--build_clones 1).
GWEN_TTS_DIR=/path/to/gwen-tts bash run.sh --data_tag main --build_clones 1 \
  --model_size medium --causal 1 --base_lr 0.045 --num_epochs 40 --avg 10 \
  --use_averaged_model 1 --enable_musan 1 --enable_spec_aug 1 \
  --exp_suffix "_lr0045" --stage -2 --stop_stage 16

# Train từ clone set đã dựng sẵn (bỏ qua clone gen), export ở cuối:
bash run.sh --data_tag main --model_size medium --causal 1 --base_lr 0.045 \
  --num_epochs 40 --avg 10 --use_averaged_model 1 --enable_musan 1 \
  --enable_spec_aug 1 --exp_suffix "_lr0045" --stage 3 --stop_stage 16

# Real-only (không clone), model small, from scratch:
bash run.sh --model_size small --causal 1 --num_epochs 60 --stage -1 --stop_stage 14
```

---

## run.sh — Stages

| Stage | Việc làm |
|---|---|
| -2 | Sinh TTS voice clones + gom training set (tùy chọn, `--build_clones 1`, cần Gwen-TTS) |
| -1 | Dựng matched train/dev/test splits từ `dataset/` |
| 0 | Tạo NoiseReduce variant `nr` (cần `--enable_nr 1`) |
| 1 | Audit dataset |
| 2 | Offline MUSAN augmentation (`--offline_musan_aug 1`) |
| 3 | Prepare manifests |
| 4 | `lhotse fix` |
| 5 | Export text corpus |
| 6 | Train BPE |
| 7 | Prepare BPE lang dir |
| 8 | Compute fbank / cuts |
| 9 | Compute MUSAN fbank (online CutMix, `--enable_musan 1`) |
| 10 | Validate cut manifests |
| 11 | Display manifest statistics |
| 12 | Tokenize smoke test |
| 13 | Train |
| 14 | Decode (greedy + modified_beam + beam_search) |
| 15 | Streaming decode (cần `--causal 1`) |
| 16 | Export int8 ONNX cho deployment (`--do_export 1`, mặc định bật) |

## run.sh — Tham số chính

```bash
--vocab_size 100
--model_size medium|small        # medium = Zipformer M; small = S (~23M)
--num_epochs 40
--world_size 1
--max_duration 500
--base_lr 0.045
--use_fp16 0|1

# Streaming (causal)
--causal 0|1                     # train + decode streaming; bắt buộc cho stage 15
--decode_chunk_size 32  --decode_left_context_frames 256

# Augmentation
--enable_spec_aug 0|1
--enable_musan 0|1               # online MUSAN (stage 9)
--offline_musan_aug 0|1          # offline MUSAN (stage 2)
--musan_dir /path/to/musan  --copies_per_utt 10  --snr_min 10  --snr_max 20
--perturb_speed 0|1

# Loss (mặc định chỉ dùng transducer)
--use_ctc 0|1  --use_cr_ctc 0|1  # CR-CTC đáng thử với tiếng Việt có tone
--ctc_loss_scale 0.2  --cr_loss_scale 0.2  --attention_decoder_loss_scale 0.0

# Decode
--decode_method all|greedy_search|modified_beam_search|beam_search
--use_averaged_model 0|1  --avg 1

# Clone generation (stage -2)
--build_clones 0|1  --gwen_python /path/.venv-gwen/bin/python  --real_mult 8

# Export (stage 16)
--do_export 0|1

# Finetune
--do_finetune 0|1  --finetune_ckpt /path/epoch-N.pt  --init_modules encoder

# Data version / variant
--data_tag NAME                  # chạy transcripts_NAME/ (đã dựng sẵn); tắt stage -1
--data_variant raw|nr  --enable_nr 0|1

# Experiment dir
--exp_suffix _myexp  --exp_dir runs  --exp_dir_policy auto|reuse|fail
```

### Model size presets

`--model_size medium` → `exp_bpe100_medium_...` (Zipformer M, dims mặc định trong `run.sh`).

`--model_size small` → `exp_bpe100_small_...`:
```
--num-encoder-layers 2,2,2,2,2,2   --feedforward-dim 512,768,768,768,768,768
--num-heads 4,4,4,8,4,4            --encoder-dim 192,256,256,256,256,256
--encoder-unmasked-dim 192,192,192,192,192,192
--decoder-dim 512  --joiner-dim 512
```

`run.sh` luôn nhúng `--model_size` vào tên exp dir (`exp_bpeNNN_medium_...` /
`_small_...`), nên `export_for_jetson.sh` tự dò được encoder dims từ đó — không
cần symlink. Hậu tố exp dir tự động: `_medium`/`_small`, `_raw`/`_nr`,
`_streaming`, `_<data_tag>`, `_finetune` (kết hợp được).

`--exp_dir` đổi thư mục cha lưu experiment (nhận relative/absolute path) mà vẫn
giữ đủ hậu tố. `--exp_dir_policy auto` (mặc định): nếu train mà exp dir đã có
file, tự chuyển sang folder `exp_..._YYYYmmdd_HHMMSS` để không ghi đè;
`reuse` ghi tiếp, `fail` dừng.

> Giải thích chi tiết từng tham số + bảng scale **S / M / L**:
> xem `docs/TEACHING_NOTES_VI.md` mục "Kích thước model".

---

## Finetune từ pretrained LibriSpeech

```bash
git lfs install
git clone https://huggingface.co/Zengwei/icefall-asr-librispeech-zipformer-small-2023-05-16 \
  pretrained/zipformer_small_ls960

bash run.sh \
  --model_size small --num_epochs 30 --base_lr 0.001 \
  --do_finetune 1 \
  --finetune_ckpt pretrained/zipformer_small_ls960/exp/epoch-50.pt \
  --init_modules encoder \
  --stage 13 --stop_stage 14
```

| model_size | Pretrained |
|---|---|
| `medium` | `Zengwei/icefall-asr-librispeech-zipformer-2023-05-15` (~65M params) |
| `small`  | `Zengwei/icefall-asr-librispeech-zipformer-small-2023-05-16` (~23M params) |

---

## Scripts trong `local/`

| File | Việc làm |
|---|---|
| `prepare_matched_splits.py` | `dataset/` → matched train/dev/test (stage -1) |
| `prepare_vi_asr_corpus.py` | `dataset/` → auto 80/10/10 split (stage -1, `--matched_splits 0`) |
| `audit_dataset.py` | Kiểm tra transcript/audio khớp nhau (stage 1) |
| `augment_train_with_musan.py` | Offline MUSAN augmentation cho train split (stage 2) |
| `prepare_manifests.py` | TSV → lhotse recordings/supervisions jsonl.gz |
| `export_text_corpus.py` | Gom text train để train BPE |
| `train_bpe_model.py --vocab-size N` | Train SentencePiece BPE |
| `prepare_lang_bpe.py --lang-dir data/lang_bpe_N` | Tạo tokens.txt, words.txt |
| `compute_fbank.py` | Tạo `*_cuts.jsonl.gz` + features trong `fbank_*/` |
| `compute_fbank_musan.py` | Tạo `musan_cuts.jsonl.gz` cho online CutMix |
| `validate_manifest.py --all` | Kiểm tra cut manifests |
| `display_manifest_statistics.py --all` | In thống kê (duration, số câu,...) |
| `tokenize_test.py --vocab-size N` | Smoke test tokenizer |
| `noise_reduce_audio.py` | Tạo `audio_nr/` + `transcripts_nr/` bằng noisereduce |
| `hieu_pipeline.py` | Gom real + clones thành divmix training set (stage -2) |
| `export_for_jetson.sh` | Export + đóng gói int8 ONNX cho Jetson (stage 16) |
| `tts/build_crossspeaker.py`, `tts/build_diverse_clones.py` | Sinh voice clones (Gwen-TTS) |

## Scripts trong `ASR/zipformer/`

| File | Việc làm |
|---|---|
| `train.py` / `finetune.py` | Train from scratch / finetune từ pretrained |
| `decode.py` | Decode (greedy / beam / modified_beam / fast_beam + LG) |
| `export-onnx.py` / `export-onnx-streaming.py` | Export transducer → ONNX |
| `generate_averaged_model.py` | Average N checkpoint → 1 file `.pt` |
| `onnx_pretrained*.py` / `onnx_decode.py` | Inference / decode từ ONNX |
| `pretrained.py` / `streaming_decode.py` | Inference `.pt` / streaming decode |

---

## Các ví dụ hay dùng

```bash
# NoiseReduce variant
bash run.sh --enable_nr 1 --data_variant nr --model_size small \
  --num_epochs 50 --stage 0 --stop_stage 14

# Offline MUSAN aug → train
bash run.sh --offline_musan_aug 1 --musan_dir "$MUSAN_DIR" \
  --copies_per_utt 10 --enable_musan 0 --stage 2 --stop_stage 13

# Average checkpoint thủ công
python3 ASR/zipformer/generate_averaged_model.py \
  --epoch 40 --avg 10 \
  --exp-dir ASR/zipformer/exp_bpe100_medium_streaming_main_lr0045 \
  --tokens data/lang_bpe_100/tokens.txt
```

## Theo dõi training bằng TensorBoard

`train.py` tự ghi log vào `<exp-dir>/tensorboard/`, không cần cấu hình gì thêm.

```bash
pip install tensorboard

# Xem 1 experiment cụ thể (model đang deploy):
tensorboard --logdir ASR/zipformer/exp_bpe100_medium_streaming_main_lr0045/tensorboard --port 6006

# Xem tất cả experiment cùng lúc (so sánh medium vs small):
tensorboard --logdir ASR/zipformer --port 6006
```

Mở `http://localhost:6006` (thêm `--bind_all` nếu xem từ máy khác qua LAN).

Nếu gặp lỗi `ModuleNotFoundError: No module named 'pkg_resources'`: `setuptools`
≥81 đã bỏ `pkg_resources` mặc định. Fix bằng cách hạ version trong venv đang dùng:

```bash
pip install "setuptools<81"
```

## Dependencies phụ (tùy chọn)

```bash
pip install noisereduce                    # cho local/noise_reduce_audio.py
lhotse prepare musan "$MUSAN_DIR" data/manifests   # MUSAN manifests
```

Gwen-TTS (clone gen, stage -2): clone `github.com/ggroup-ai-lab/gwen-tts`,
set `GWEN_TTS_DIR` hoặc `--gwen_python`. Xem `local/tts/README.md`.
