# CLAUDE.md — vi_asr_corpus ASR Project

## Project Overview

Vietnamese ASR toy project using icefall (Zipformer2 + Pruned Transducer).
- **Dataset**: `vi_asr_corpus` — 3 long sentences, 5 speakers, ~408 train / 25 test utterances, ~1 hour audio
- **Model**: Zipformer small (`encoder_dim=192,256,256,256,256,256`, `feedforward_dim=512,768,768,768,768,768`)
- **Vocab**: BPE 100
- **Goal**: Full pipeline demo (train → eval → real-time mic ASR)

Reference dataset for comparison: `../VietnameseASR` (real diverse data, same model architecture).

---

## 3 Training Sentences

```
1. tôi ghi âm dữ liệu thử nghiệm vào lúc tám giờ ba mươi phút trong phòng yên tĩnh để kiểm tra hệ thống nhận dạng tiếng nói hôm nay
2. hôm nay tôi kiểm tra hệ thống nhận dạng tiếng nói bằng dữ liệu thử nghiệm và ghi âm trong phòng yên tĩnh vào lúc tám giờ ba mươi phút
3. trong phòng yên tĩnh vào lúc tám giờ ba mươi phút hôm nay tôi ghi âm dữ liệu thử nghiệm để kiểm tra hệ thống nhận dạng tiếng nói
```

5 speakers (Dung, HIEU, Khoi, Quan, Trung) — all appear in both train and test (closed-speaker).

---

## Key Files

| File | Mô tả |
|---|---|
| `run.sh` | Pipeline chính: data prep → train → decode |
| `mic_streaming_asr.py` | Real-time mic ASR demo (streaming + full mode) |
| `local/compute_fbank.py` | Feature extraction dùng **lhotse Fbank** (phải match inference) |
| `ASR/zipformer/train.py` | Training script |
| `ASR/zipformer/decode.py` | Offline decode + WER |
| `ASR/zipformer/export.py` | JIT TorchScript export cho inference |
| `../RESULTS.md` | WER results của tất cả experiments |
| `../run_all_experiments.sh` | Master script chạy nhiều experiments |

---

## Model Paths (current best: small/scratch30, avg=7, WER=4.62% on test)

```
Non-streaming JIT : ASR/zipformer/exp_bpe100_small_scratch30/jit_script.pt
Streaming JIT     : ASR/zipformer/exp_bpe100_small_streaming_scratch30_streaming/jit_script_chunk_32_left_256.pt
Tokens            : data/lang_bpe_100/tokens.txt
```

Export lại model (ví dụ avg=7):
```bash
source ~/test-icefall/bin/activate
cd /home/trung/icefall/egs/vi_asr_corpus
python3 ASR/zipformer/export.py \
  --tokens data/lang_bpe_100/tokens.txt \
  --use-averaged-model 1 --epoch 30 --avg 7 \
  --exp-dir ASR/zipformer/exp_bpe100_small_scratch30 \
  --encoder-dim 192,256,256,256,256,256 \
  --encoder-unmasked-dim 192,192,192,192,192,192 \
  --num-encoder-layers 2,2,2,2,2,2 \
  --feedforward-dim 512,768,768,768,768,768 \
  --num-heads 4,4,4,8,4,4 \
  --decoder-dim 512 --joiner-dim 512 \
  --jit 1
```

---

## Real-time Mic ASR

```bash
source ~/test-icefall/bin/activate
cd /home/trung/icefall/egs/vi_asr_corpus

python3 mic_streaming_asr.py --mode full              # non-streaming, beam=4
python3 mic_streaming_asr.py --mode streaming         # streaming, partial results
python3 mic_streaming_asr.py --mode full --beam 8     # larger beam
python3 mic_streaming_asr.py --mode full --save-audio /tmp/utt_audio  # lưu audio để debug
```

**QUAN TRỌNG**: Script phải chạy từ thư mục `vi_asr_corpus/` vì paths là relative.

### Feature extraction trong inference

Dùng **lhotse Fbank** (KHÔNG dùng `torchaudio.compliance.kaldi.fbank`) để match training:
```python
from lhotse import Fbank, FbankConfig
extractor = Fbank(FbankConfig(num_mel_bins=80))
feats = extractor.extract(waveform_numpy, sampling_rate=16000)
```

Sai lầm cũ: `torchaudio.compliance.kaldi.fbank(..., snip_edges=True, high_freq=-400)` → feature mismatch hoàn toàn.

---

## Training

```bash
# Train small model, non-streaming, 30 epochs, từ scratch
bash run.sh --model_size small --num_epochs 30 --base_lr 0.01 \
  --exp_suffix _scratch30 --stage 13 --stop_stage 13

# Decode (greedy + beam + modified_beam)
bash run.sh --model_size small --num_epochs 30 \
  --exp_suffix _scratch30 --avg 7 --stage 14 --stop_stage 14
```

**Giữ tất cả epoch files** để test avg sau: `CLEAN_CHECKPOINTS=0` (default trong `run_all_experiments.sh`).

### Small model args (QUAN TRỌNG — decoder/joiner là 512, KHÔNG phải 256)
```
--encoder-dim 192,256,256,256,256,256
--encoder-unmasked-dim 192,192,192,192,192,192
--num-encoder-layers 2,2,2,2,2,2
--feedforward-dim 512,768,768,768,768,768
--num-heads 4,4,4,8,4,4
--decoder-dim 512
--joiner-dim 512
```

---

## Known Issues & Diagnoses

### 1. Model hallucinate / tự complete câu
**Nguyên nhân**: Dataset chỉ có 3 câu dài (~130 tokens mỗi câu, các câu chia sẻ nhiều sub-phrase). Model học ngữ cảnh mạnh → nói nửa câu là tự complete phần còn lại.
**Giải pháp**: Thêm dữ liệu đa dạng hơn, hoặc chấp nhận behavior này vì đây là dataset toy.

### 2. Mic không nhận ra dù nói đúng câu
**Nguyên nhân chính**: User là unseen speaker (model chỉ train trên 5 speaker: Dung, HIEU, Khoi, Quan, Trung). Không có speaker normalization.
**Giải pháp**: Ghi thêm giọng của user vào dataset và retrain.

### 3. Feature mismatch training vs inference (ĐÃ FIX)
Training dùng lhotse Fbank (`snip_edges=False`, `high_freq=0`).
Inference cũ dùng torchaudio (`snip_edges=True`, `high_freq=-400`) → hoàn toàn khác.

### 4. best-train-loss.pt / best-valid-loss.pt không đáng tin
Với dataset nhỏ, val loss plateau sớm (epoch 8–11) không phản ánh WER thực.
Nên decode epoch-30 hoặc avg nhiều epochs, không dùng best-loss checkpoint.

### 5. streaming_decode.py cần `--causal 1`
AssertionError nếu thiếu. Thêm `--causal 1` khi decode streaming model.

---

## WER Summary (vi_asr_corpus, small model)

| Config | Checkpoint | Best WER | Method |
|---|---|---|---|
| non-streaming scratch30 | epoch-30 (avg=7) | **4.62%** | beam_search |
| non-streaming scratch30 | epoch-30 (avg=1) | 16.30% | modified_beam_search |
| streaming scratch30 | epoch-30 (avg=7) | **35.46%** | modified_beam_search |

*WER thấp là do test set có cùng speaker với train (closed-speaker). Không phản ánh khả năng generalize.*

---

## Next Steps (xem phần Analysis bên dưới để biết thứ tự ưu tiên)

1. Ghi thêm giọng user → retrain (fix speaker generalization ngay lập tức)
2. CMVN utterance-level (normalize acoustic conditions)
3. Speed perturbation + SpecAugment (đã có trong run.sh, cần bật)
4. Tăng số speakers và dữ liệu đa dạng hơn
