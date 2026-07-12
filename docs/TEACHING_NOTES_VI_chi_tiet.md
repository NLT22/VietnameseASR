# VietnameseASR — Ghi chú toàn tập (tiếng Việt)

Viết cho **chính bạn, vài tháng sau**, khi chỉ còn nhớ là dự án này từng tồn tại.
Mọi con số và shape trong file này đều được **đọc thẳng từ code** hoặc đo trên
máy — không có gì viết theo trí nhớ. Chỗ nào dự án đang hỏng thì nói là hỏng.
Chỗ nào trước đây tôi kết luận sai thì cũng ghi rõ là sai.

Các thuật ngữ tiếng Anh (encoder, decoder, joiner, transducer, beam search,
checkpoint, fbank, WER, streaming, chunk...) **giữ nguyên tiếng Anh**, không dịch.

- `RESULTS.md` — các con số.
- `TEACHING_NOTES.md` — bản tiếng Anh của file này.
- File này = giải thích **tại sao mọi thứ lại như vậy** + **toàn bộ config**.

> **Cập nhật (2026-07-12):** model đang deploy giờ là **medium (M)**
> (`deploy/jetson_nano/model_medium_epoch30_avg10`), 5 người nói (có Hieu):
> WER thực **1.84%**, held-out **1.80%** — tốt hơn model nhỏ ở mọi người nói, vẫn
> chạy thời gian thực trên Nano (RTF ≈ 0.31). Model `divmix_x8` (4 người) dùng làm
> ví dụ xuyên suốt bên dưới đã được lưu kết quả rồi xoá checkpoint; shape và pipeline
> của nó vẫn đúng để minh hoạ. Xem `RESULTS.md` và `ARCHIVED_EXPERIMENTS.md`.

---

## 0. Tóm tắt một đoạn

5 người thu âm, mỗi người 200 câu tiếng Việt ngắn (**1.000 recording thật**).
Pipeline hiện tại dùng tag `main`: phần train lặp lại recording thật và thêm
voice clone từ Gwen-TTS, tạo `transcripts_main/train.tsv` khoảng **16.000 dòng**;
dev/test vẫn là tập recording thật sạch để đo ổn định. Ta train một **streaming
Zipformer transducer** để nhận dạng tập câu cố định. Train, dev, test cố ý dùng
cùng tập câu ("học vẹt"), nên WER chính là chỉ số **ghi nhớ miền hẹp**, không
phải năng lực ASR tiếng Việt mở. Model deploy hiện tại là Zipformer **medium**,
export ONNX int8, chạy trên Jetson Nano và chạy live trên trình duyệt ở máy này.

Bài học lớn nhất của dự án: **chỉ số WER chính đã mù hoàn toàn trước một bug làm
hỏng 45% nhãn train suốt hai tháng.**

---

## 1. Model: transducer = ba mạng riêng biệt

Transducer (RNN-T) **không phải một model**. Nó là ba mạng, train chung nhưng
export ra ba ONNX graph riêng. Shape lấy từ
`deploy/jetson_nano/model_medium_epoch30_avg10/`:

| graph | inputs | outputs | nhiệm vụ |
| --- | --- | --- | --- |
| **encoder** | `x[N,77,80]` + **74 state tensors** | `encoder_out[N,16,512]` + 74 states mới | nghe audio |
| **decoder** | `y[N,2]` (2 token id gần nhất) | `decoder_out[N,512]` | nhớ vừa nói gì |
| **joiner** | `encoder_out[N,512]`, `decoder_out[N,512]` | `logit[N,100]` | chấm điểm token kế tiếp |

### 1.1 Encoder

Đầu vào là **fbank 80 chiều** (log-mel filterbank), không phải waveform thô.
Config đọc từ metadata của ONNX:

```
num_encoder_layers   2,2,2,2,2,2      # 6 stack x 2 layer = 12 layer
encoder_dims         192,256,256,256,256,256
num_heads            4,4,4,8,4,4
left_context_len     256,128,64,32,64,128
cnn_module_kernels   31,31,15,15,15,31
decode_chunk_len     64
T                    77
```

Encoder là **causal**: không bao giờ nhìn audio tương lai ngoài chunk của nó. Đó
là điều kiện để streaming được, và nó **đánh đổi độ chính xác** so với encoder
hai chiều.

**Nó subsample ~4×.** Đưa vào `(1,77,80)` thì nhận về `(1,16,512)`. Suy ra:

- 1 feature frame = 10 ms (frame shift)
- 1 **encoder frame = 40 ms** audio
- cửa sổ 77 frame = `25 + 76*10` = **785 ms** audio
- cửa sổ trượt `decode_chunk_len=64` frame = **640 ms**, sinh ra 16 encoder frame

Phần chồng lấn 13 frame (`77 - 64`) chính là right-context: model được "nhìn
trước" 130 ms trong chính chunk của nó.

### 1.2 Bảy mươi tư state tensors (chỗ ai cũng rối)

`encoder.onnx` có **75 inputs**: 1 audio + **74 tensor mang trí nhớ** của encoder
qua các chunk. Shape được tính trong `jetson_beam_decode.py:reset()`:

```python
for i in range(len(num_encoder_layers)):        # 6 stack
    key_dim = query_head_dims[i] * num_heads[i]
    nonlin  = 3 * encoder_dims[i] // 4
    val_dim = value_head_dims[i] * num_heads[i]
    pad     = cnn_module_kernels[i] // 2
    for _ in range(num_encoder_layers[i]):      # 2 layer / stack
        cached_key          (left_context_len[i], N, key_dim)
        cached_nonlin_attn  (1, N, left_context_len[i], nonlin)
        cached_val1         (left_context_len[i], N, val_dim)
        cached_val2         (left_context_len[i], N, val_dim)
        cached_conv1        (N, encoder_dims[i], pad)
        cached_conv2        (N, encoder_dims[i], pad)
embed_states  (N, 128, 3, 19)
processed_lens(N,)
```

`6 stack × 2 layer × 6 tensor = 72`, cộng `embed_states` + `processed_lens` =
**74**. Bốn trong sáu tensor là **attention KV cache** (đó chính là ý nghĩa của
`left_context_len`: mỗi stack được attend về quá khứ bao nhiêu frame); hai tensor
còn lại là **ring buffer của convolution**, vì causal conv kernel 31 cần 15 mẫu
trước đó.

Đầu mỗi utterance: cả 74 tensor = 0. Hàm `reset()` làm việc đó.
**Quên reset giữa hai utterance thì model "nghe" cả câu trước.**

### 1.3 Decoder (không phải cái bạn nghĩ)

Nó **không** phải "bộ giải mã". Nó là một **language model bé xíu trên token**.
Input `y[N,2]` — 2 token id gần nhất. Output vector 512 chiều.

`context_size=2` nghĩa là **bigram**: chỉ biết 2 token trước. Vì thế nó
stateless và cache được — cùng một cặp token luôn cho cùng output. Đó là lý do
`beam_search()` giữ một dict `dec_cache`.

### 1.4 Joiner

Cộng bằng chứng acoustic (encoder) với ngữ cảnh ngôn ngữ (decoder), chiếu ra
`logit[N,100]` — mỗi token trong vocabulary một điểm. `blank_id = 0`.

### 1.5 Vì sao transducer phát **nhiều symbol trên một frame**

Ở mỗi encoder frame `t`, joiner chấm 100 token:

- Nếu chọn **blank** → nhảy sang frame `t+1`.
- Nếu chọn **token thật** → phát ra token đó, cập nhật decoder context, và
  **ở nguyên frame `t`** để có thể phát tiếp token nữa.

Vòng lặp đó là bản chất của transducer, và là **sự thật quan trọng nhất của dự
án**, bởi vì:

> `sherpa-onnx` chỉ có `greedy_search` và `modified_beam_search`, và **cả hai chỉ
> phát tối đa 1 symbol/frame**. Model của ta được train để phát nhiều hơn. Trên
> cùng bộ weight: sherpa được 3.31%, classic `beam_search` của ta được **0.72%**
> — khoảng cách gần như toàn bộ là **deletion** mà thuật toán sherpa về mặt cấu
> trúc không thể sinh ra.

Xem `beam_search()` trong `jetson_beam_decode.py`: vòng `while True:` nằm *bên
trong* vòng `for t in range(...)` chính là chỗ phát nhiều symbol. Hàm `greedy()`
không có vòng trong đó — nên nó tệ hơn.

### 1.6 Loss: vì sao gọi là **Pruned** RNN-T

Trong `ASR/zipformer/train.py`:

```
simple_loss, pruned_loss, ... = model(..., prune_range=5, am_scale=0.0, lm_scale=0.25)
loss = simple_loss_scale * simple_loss + pruned_loss     # simple_loss_scale = 0.5
```

Loss RNN-T đúng nghĩa phải cộng qua **mọi đường đi** trong lattice
`(T frame × U token)` — với ta là ~100 × ~60 ô, mỗi ô cần một lần forward joiner
đầy đủ. Quá đắt. Nên:

1. **`simple_loss`**: thay joiner bằng phép cộng `encoder_out + decoder_out`
   (không MLP). Tính lattice chính xác trên đó. Model tồi, nhưng là **kim chỉ
   nam** tốt.
2. Dùng gradient của nó để tìm, ở mỗi frame, khoảng ~5 vị trí token thực sự đáng
   quan tâm (`prune_range=5`).
3. **`pruned_loss`**: chạy joiner thật **chỉ trên dải hẹp đó**.

`lm_scale=0.25`, `am_scale=0.0` quyết định thành phần LM/AM của model đơn giản
đóng góp bao nhiêu vào việc chọn dải. Đây là tối ưu **loss**, không phải tối ưu
model.

---

## 2. Token: vì sao vocab = 100 cho 1.913 từ

`data/lang_bpe_100/` chứa một **SentencePiece BPE** đúng **100 token**.
`tokens.txt` map token → id, và đó chính là chiều của `logit[N,100]`:

```
<blk>      0     # "không phát gì, tiến thời gian" của transducer
<sos/eos>  1
<unk>      2
▁t         3     # "▁" (U+2581) = đầu từ
ng         4
...
ỵ          99
```

Corpus có **1.913 từ khác nhau / 20.790 token từ**. Không thể dùng vocabulary
mức từ — phần lớn từ chỉ xuất hiện vài lần. BPE cắt từ thành các mảnh subword phổ
biến: `▁t` + `ôi` ghép thành "tôi", và từ chưa gặp vẫn phân rã được thành mảnh đã
biết.

Vì sao 100 mà không phải 500? Tiếng Việt **đơn âm tiết, phân tích tính**: từ ngắn
(1–2 âm tiết), kho âm vị nhỏ, chính tả gần như ghi âm. Một trăm mảnh là đủ phủ
cấu trúc âm tiết. Vocabulary lớn hơn = nhiều lớp output hơn, mỗi lớp ít ví dụ hơn
— trong khi ta chỉ có ~1,1 giờ audio thật.

Detokenize đúng bằng `"".join(pieces).replace("▁", " ")` — xem `detok()`.

> **Bẫy.** `cut -f4 | tr ' ' '\n' | sort -u | wc -l` báo **2.279** cho file này.
> Sai. `sort -u` với glibc collation để sót 366 dòng trùng trên dấu tiếng Việt.
> Hãy đếm bằng Python, đừng đếm bằng `sort`. Bản nháp đầu của tài liệu này đã in
> nhầm 2.279 đúng vì lý do đó.

**Hệ quả cần nhớ:** 100 logit của joiner là điểm của **subword**. Một token sai
có thể làm hỏng một từ mà không đổi số từ trong câu — nên WER trông ổn định trong
khi output đã thành vô nghĩa.

---

## 3. Pipeline dữ liệu, theo từng stage

`run.sh` là **cửa vào duy nhất** (`bash run.sh --help`) cho chuẩn bị dữ liệu,
train, decode và export deploy. Stage 16 gọi `local/export_for_jetson.sh`, script
bọc quanh exporter ONNX của icefall và tạo gói model cho Jetson/live UI.

```
dataset/<Speaker>/            wav thô + script.txt   (mỗi dòng = một recording)
   │  stage -1  local/prepare_matched_splits.py
   ▼
transcripts_matched_u20/{train,dev,test}.tsv          utt_id, speaker, audio_path, text
   │  stage -2  local/tts/*.py + local/hieu_pipeline.py -> transcripts_main/
   │  stage 3   local/prepare_manifests.py            -> lhotse Recording/Supervision
   │  stage 4   fix manifests
   │  stage 5-7 xuất text corpus -> train BPE -> lang_bpe_100/
   │  stage 8   local/compute_fbank.py                -> lhotse Fbank 80 chiều
   ▼
fbank_<tag>/{train,dev,test}_cuts.jsonl.gz
   │  stage 9   MUSAN fbank (để trộn noise online)
   │  stage 13  ASR/zipformer/train.py
   ▼
ASR/zipformer/exp_.../epoch-N.pt
   │  stage 16 local/export_for_jetson.sh
   │           (export-onnx-streaming.py + quantize_dynamic)
   ▼
deploy/jetson_nano/model_.../{encoder,decoder,joiner}.int8.onnx + tokens.txt
```

Đường chạy hiện tại:

```
transcripts_matched_u20/     1.000 utterance thật sạch, cùng tập câu
transcripts_main/            train = real x8 + TTS clone; dev/test = real sạch
data/manifests_main/         manifest lhotse cho transcripts_main
fbank_main/                  cut/feature Fbank 80 chiều
data/lang_bpe_100/           SentencePiece BPE + tokens.txt
ASR/zipformer/exp_bpe100_medium_streaming_main_lr0045/
deploy/jetson_nano/model_medium_epoch30_avg10/
```

### 3.0 Map theory → code hiện tại

Phần này là cầu nối giữa lý thuyết và file thật trong project:

| khái niệm | file hiện tại | nhiệm vụ |
| --- | --- | --- |
| Tạo split thật | `local/prepare_matched_splits.py`, `local/prepare_vi_asr_corpus.py` | ghép audio trong `dataset/<speaker>/` với dòng script, tạo TSV |
| Tạo và trộn voice clone | `local/tts/build_crossspeaker.py`, `local/tts/build_diverse_clones.py`, `local/hieu_pipeline.py` | sinh clone và assemble `transcripts_main` |
| Audit dữ liệu | `local/audit_dataset.py`, `local/validate_manifest.py`, `local/display_manifest_statistics.py` | bắt lỗi path, duration, text rỗng, data bất thường |
| Manifest lhotse | `local/prepare_manifests.py` | TSV -> recordings/supervisions JSONL.GZ |
| Text corpus và BPE | `local/export_text_corpus.py`, `local/train_bpe_model.py`, `local/prepare_lang_bpe.py` | train SentencePiece, tạo `tokens.txt`, `words.txt` |
| Fbank | `local/compute_fbank.py`, `local/compute_fbank_musan.py` | tạo cut và feature 80 mel bins |
| Data loader | `ASR/zipformer/asr_datamodule.py` | đọc cut, batch theo duration, augmentation |
| Model | `ASR/zipformer/model.py`, `zipformer.py`, `subsampling.py`, `decoder.py`, `joiner.py` | Zipformer-Transducer |
| Train | `ASR/zipformer/train.py`, `optim.py`, `scaling.py` | Pruned RNN-T loss, ScaledAdam/Eden, checkpoint |
| Average checkpoint | `ASR/zipformer/generate_averaged_model.py` + flag decode/export | lấy trung bình checkpoint hoặc dùng `model_avg` |
| Decode PyTorch | `ASR/zipformer/decode.py`, `streaming_decode.py`, `beam_search.py`, `streaming_beam_search.py` | tính WER từ checkpoint |
| Export ONNX | `local/export_for_jetson.sh`, `ASR/zipformer/export-onnx-streaming.py`, `export-onnx.py` | xuất encoder/decoder/joiner và quantize int8 |
| Decode ONNX | `deploy/jetson_nano/jetson_beam_decode.py`, `transcribe_beam_wav.py`, `jetson_asr.py`, `onnx_beam_search.py` | inference thật, không cần torch/k2 |
| Đo performance | `deploy/jetson_nano/evaluate_performance.py`, `evaluate_streaming_performance.py`, `run_performance_eval.sh` | đo WER/tốc độ trên PC và Jetson |
| Live UI | `live_ui/server.py`, `live_ui/stream_decoder.py`, `live_ui/vad.py`, `live_ui/speaker_id.py` | mic web, streaming decode, VAD, speaker ID |

Các file CTC/JIT cũ đã bị xoá khỏi pipeline. Đường chính hiện tại là
Transducer -> ONNX encoder/decoder/joiner -> custom beam search.

### 3.1 Audio ghép với text như thế nào (bug định nghĩa cả dự án)

`prepare_matched_splits.py` gọi `scan_auto_dataset()`:

```python
audio_files = list_audio_files(speaker_dir)   # đã sort
texts       = read_prompt_lines(script_file)  # mỗi dòng một recording
assert len(audio_files) == len(texts)
zip(audio_files, texts)                       # GHÉP THEO VỊ TRÍ
```

Transcript **không có id**. Việc ghép hoàn toàn theo vị trí, nên **thứ tự sort
của file audio chính là nhãn**.

Windows Voice Recorder đặt tên bản thu đầu là `Recording.wav`, các bản sau là
`Recording (2).wav`, `(3)`… Natural sort đẩy file **không số xuống cuối**. Kết
quả: dòng 1 ghép với bản thu 2, dòng 2 ghép bản 3, …, dòng 200 ghép bản 1.

Toàn bộ utterance của **Trung và Dung** bị gán nhãn của câu **liền trước** —
**330 / 730 recordings**. Quan (`Recording (1..200)`) và Khoi (`001..200.wav`)
không có file không số nên **không bị ảnh hưởng**.

Sửa bằng `recorder_key()` trong `local/prepare_vi_asr_corpus.py`:

```python
def recorder_key(name):
    m = re.search(r"\((\d+)\)", name)
    return (int(m.group(1)) if m else 1, natural_key(name))   # thiếu "(N)" == bản 1
```

**Phép kiểm tra bắt được bug.** Câu dài thì đọc lâu, nên
`corr(audio_duration, text_word_count)` phải dương rõ ràng theo từng speaker. Tập
bị lệch sẽ tụt về 0. Quan không có file không số nên hai cách sort cho kết quả y
hệt → **control miễn phí**.

| speaker | natural sort | bare-first | |
| --- | ---: | ---: | --- |
| Quan (control) | +0.472 | +0.472 | không ảnh hưởng |
| Trung | +0.858 | **+0.954** | bị lệch |
| Dung | −0.013 | **+0.216** | bị lệch |

**Chạy phép kiểm tra này sau mỗi lần nạp dữ liệu. Ba dòng code, đáng lẽ đã tiết
kiệm được hai tháng.**

### 3.2 Feature: lhotse Fbank, và cái bẫy scaling

Train dùng **`Fbank` của lhotse**, nên inference phải tái tạo **y hệt**.
`live_ui/stream_decoder.py:fbank_opts()` và `deploy/jetson_nano/jetson_asr.py`
đều chép lại:

```
samp_freq 16000   dither 0.0        snip_edges False
frame_length 25ms frame_shift 10ms  preemph 0.97
window povey      remove_dc_offset True
num_mel_bins 80   low_freq 20       high_freq -400   (tức 8000-400 = 7600 Hz)
```

**Waveform bắt buộc là float trong `[-1, 1]`.** Kaldi gốc dùng
`[-32768, 32767]`. Nếu đưa số nguyên vào, **mọi mel bin lệch đi
`ln(32768²) ≈ 20,8`**, model nhìn thấy feature nó chưa từng thấy. Đây là lỗi
**âm thầm và toàn phần** — không exception, chỉ ra rác. Ở mọi nơi:
`pcm.astype(np.float32) / 32768.0`.

### 3.3 Augmentation

- **Online MUSAN** (`--enable_musan 1`): trộn noise trong lúc train. Cần
  `fbank_*/musan_cuts.jsonl.gz` từ stage 9.
- **SpecAugment** (`--enable_spec_aug 1`): che dải time/frequency của fbank.
- **Speed perturb**: **tắt**. Nó nhân ba số cut, để đổi lấy sự đa dạng mà 9 giọng
  của ta vốn đã cung cấp.
- **Offline MUSAN copies**: đã bỏ. Tốn đĩa 5–10× và chỉ dạy chống noise, trong
  khi trục ta thực sự thiếu là **đa dạng giọng nói**.

### 3.4 Clone giọng

`local/tts/` dùng **Gwen-TTS** (Qwen3-TTS-0.6B finetune trên 1000 h tiếng Việt).
Cross-speaker clone cho 5 giọng nam; diverse clone thêm 3 nữ + 1 nam.

Hai luật, đều rút ra từ việc đã vi phạm:

1. **`yen_nhi` và `khanh_toan` tuyệt đối không vào train.** Đó là hai giọng eval.
   Clone câu train sang hai giọng này thì benchmark trung thực **âm thầm** biến
   thành một bài kiểm tra học thuộc khác.
2. **Chất lượng audio tham chiếu lan truyền.** Bản gốc của Dung peak 0.059 →
   clone ra giọng gần như không nghe được. Reference lấy từ
   `datasets/vi_asr_corpus/`, phải cắt khoảng lặng + peak-normalize, và
   `ref_text` **phải khác** `gen_text`, nếu không output sẽ lặp từ.

Tỉ lệ quan trọng: ~80% synthetic → decode suy biến toàn blank (27–48% WER).
26–50% synthetic thì train khoẻ.

---

## 4. TOÀN BỘ CONFIG PARAMETER

### 4.1 `run.sh` — default và ý nghĩa

Giá trị dưới đây là **default trong `run.sh`** (đọc từ code). Default hiện tại là
`base_lr=0.045`; những ghi chú cũ nhắc `0.01` đang nói về bug learning-rate của
các lần train trước.

**Điều khiển pipeline**

| tham số | default | ý nghĩa |
| --- | --- | --- |
| `--stage` / `--stop_stage` | `-1` / `100` | chạy từ stage nào đến stage nào |
| `--data_tag` | `""` | chạy một bộ transcript dựng sẵn: `transcripts_NAME/` → `fbank_NAME/`, `data/manifests_NAME/`, exp suffix `_NAME`. Tự đặt `matched_splits=0` |
| `--matched_splits` | `1` | dùng split "học vẹt" (train == dev == test) |
| `--data_variant` | `raw` | `raw` hoặc bản noise-reduce |
| `--exp_suffix` | `_x10_matched` | hậu tố tên thư mục experiment |
| `--exp_dir_policy` | `auto` | `auto`: tự tạo thư mục mới nếu đã có file; `reuse`: ghi đè/resume; `fail`: dừng |
| `--corpus_root` | thư mục recipe | gốc dữ liệu |
| `--build_clones` | `0` | stage -2: sinh Gwen-TTS clone và assemble transcript train |
| `--do_export` | `1` | stage 16: export gói ONNX int8 |

**Dữ liệu / feature**

| tham số | default | ý nghĩa |
| --- | --- | --- |
| `--vocab_size` | `100` | số token BPE → chiều output của joiner |
| `--split_max_duration` | `20` | bỏ recording dài hơn 20 s khi tạo split |
| `--feature_max_duration` | `20` | lọc cut khi tính fbank |
| `--musan_dir` | `./musan` | thư mục MUSAN |
| `--enable_nr` | `0` | biến thể noise-reduce |

**Augmentation**

| tham số | default | dự án dùng | ý nghĩa |
| --- | --- | --- | --- |
| `--enable_musan` | `0` | **1** | trộn noise MUSAN online |
| `--enable_spec_aug` | `0` | **1** | SpecAugment |
| `--perturb_speed` | `0` | `0` | speed perturb (nhân 3 số cut) |
| `--offline_musan_aug` | `1` | **0** | nhân bản file audio có noise ra đĩa |
| `--copies_per_utt` | `10` | – | số bản offline mỗi utterance |
| `--snr_min` / `--snr_max` | `10` / `20` | – | dải SNR khi trộn offline |
| `--real_mult` | `8` | **8** | lặp recording thật khi assemble train có clone |

**Kích thước model**

`--model_size small` ép các giá trị sau (preset `base` = để nguyên default của
`train.py`):

```
num_encoder_layers    2,2,2,2,2,2
feedforward_dim       512,768,768,768,768,768
num_heads             4,4,4,8,4,4
encoder_dim           192,256,256,256,256,256
encoder_unmasked_dim  192,192,192,192,192,192
decoder_dim           512
joiner_dim            512
```

Có thể override từng cái: `--num_encoder_layers`, `--encoder_dim`,
`--feedforward_dim`, `--num_heads`, `--encoder_unmasked_dim`, `--decoder_dim`,
`--joiner_dim`.

*Giải thích từng tham số.* Encoder Zipformer gồm **6 "stack"** (khối) nối tiếp,
mỗi stack chạy ở một độ phân giải thời gian khác nhau (downsampling factor
`1,2,4,8,4,2` — hình chữ U: càng vào giữa càng nén thời gian mạnh để nắm ngữ cảnh
dài, rồi giãn lại). Nên mỗi tham số là **danh sách 6 giá trị, mỗi giá trị cho một
stack**:

| tham số | ý nghĩa |
| --- | --- |
| `num_encoder_layers` `2,2,2,2,2,2` | số layer trong mỗi stack. Cộng = **12 layer** cho small. Sâu hơn = nhớ nhiều hơn. |
| `encoder_dim` `192,256,256,...` | **chiều ẩn** (số kênh) mỗi stack — "độ rộng" model. |
| `feedforward_dim` `512,768,768,...` | chiều lớp MLP bên trong mỗi layer (~3× encoder_dim). |
| `num_heads` `4,4,4,8,4,4` | số **attention head** mỗi stack. Stack 4 (nén mạnh nhất) dùng 8, còn lại 4. |
| `encoder_unmasked_dim` `192,...` | số kênh **luôn giữ**, không bị drop ngẫu nhiên khi train (Zipformer bỏ ngẫu nhiên một phần kênh để chống overfit; đây là "sàn"). |
| `decoder_dim` `512` | chiều embedding của **decoder** (bộ nhớ 2 token gần nhất). |
| `joiner_dim` `512` | chiều chiếu chung của **joiner** trước khi ra `logit[N, vocab]`. |

`num_heads`, `decoder_dim`, `joiner_dim` **giống nhau** ở mọi scale; chỉ số layer
và các chiều encoder/feedforward khác nhau giữa các scale.

*Ba scale S / M / L (theo paper Zipformer).* Tên trong recipe map sang scale của
paper (đã kiểm tra khớp từng số với `train.py`):

| recipe | = scale | num_encoder_layers | encoder_dim | feedforward_dim |
| --- | --- | --- | --- | --- |
| `--model_size small` | **S** | 2,2,2,2,2,2 | 192,256,256,256,256,256 | 512,768,768,768,768,768 |
| `--model_size medium` | **M** | 2,2,3,4,3,2 | 192,256,384,512,384,256 | 512,768,1024,1536,1024,768 |
| *(chưa dùng)* | **L** | 2,2,4,5,4,2 | 192,256,512,768,512,256 | 512,768,1536,2048,1536,768 |

Model càng lớn (S → M → L) = nhiều tham số hơn = **nhớ được nhiều câu hơn**, nhưng
tốn GPU hơn (M ở `max_duration 500` dùng ~15 GB; L cần hạ `max_duration` xuống
~300 để vừa 16 GB). Với bài toán học vẹt: nếu tăng số câu train làm giảm độ chính
xác các câu cũ, đó là **thiếu capacity** → leo thang S → M → L là hướng sửa.

> ⚠️ Nếu train scale khác small, `local/export_for_jetson.sh` phải dùng đúng dims
> của scale đó, nếu không sẽ lỗi size-mismatch khi load checkpoint. Script tự nhận
> diện small/base từ tên exp dir (`exp_bpe*_small_*` / `_medium_*`).

**Training**

| tham số | default `run.sh` | dự án dùng | ý nghĩa |
| --- | --- | --- | --- |
| `--num_epochs` | `30` | **60 khi sweep; deploy epoch 30** | số epoch |
| `--start_epoch` | `1` | – | resume: load `epoch-(N-1).pt` |
| `--world_size` | `1` | `1` | số GPU |
| `--max_duration` | `500` | **700** | tổng **giây audio** mỗi batch (không phải số câu!) |
| `--num_workers` | `2` | **4** | dataloader worker |
| `--base_lr` | `0.045` | **0.045** | learning rate; `0.01` cũ train chậm và WER xấu hơn |
| `--use_fp16` | – | **1** | mixed precision khi train |
| `--bucketing_sampler` | `1` | `1` | gom cut cùng độ dài vào một batch |
| `--num_buckets` | `4` | `4` | số bucket |
| `--causal` | `0` | **1** | **bắt buộc = 1 cho streaming** |
| `--chunk_size` | `16,32,64,-1` | (như default) | list, train ngẫu nhiên nhiều chunk size |
| `--left_context_frames` | `64,128,256,-1` | (như default) | list, tương ứng |

**Loss (truyền thẳng xuống `train.py`)**

| tham số | default | ý nghĩa |
| --- | --- | --- |
| `--prune-range` | `5` | bề rộng dải token giữ lại sau khi prune |
| `--lm-scale` | `0.25` | trọng số phần LM khi chọn dải |
| `--am-scale` | `0.0` | trọng số phần AM khi chọn dải |
| `--simple-loss-scale` | `0.5` | hệ số của `simple_loss` |
| `--ctc-loss-scale` | `0.2` | chỉ dùng khi `--use_ctc 1` |
| `--cr-loss-scale` | `0.2` | chỉ dùng khi `--use_cr_ctc 1` |
| `--attention-decoder-loss-scale` | `0.0` | tắt |
| `--use_ctc` / `--use_cr_ctc` | `0` / `0` | thêm CTC head |

**Chỉ có trong `train.py`** (không expose qua `run.sh`, phải sửa code):

| tham số | default | ý nghĩa |
| --- | --- | --- |
| `--average-period` | **200** | bao nhiêu batch thì cập nhật `model_avg` một lần |
| `--lr-batches` | `7500` | hằng số giảm LR theo batch |
| `--lr-epochs` | `3.5` | hằng số giảm LR theo epoch |
| `--warm-step` | `2000` | warmup, chi phối `simple_loss_scale` giảm dần |
| `--seed` | `42` | seed |

> ⚠️ **`average-period=200` > 159 batch/epoch** (12.800 cut, `max_duration 700`).
> Nghĩa là `model_avg` — thứ mà `--use_averaged_model` đọc — **cập nhật chưa tới
> một lần mỗi epoch**. Hai checkpoint epoch liền nhau thường mang **cùng một**
> averaged snapshot: `avg=1` và `avg=2` export ra **weight giống nhau từng byte**
> (cả fp32 lẫn int8). Checkpoint averaging đang làm **ít hơn** những gì cái flag
> gợi ý. Muốn averaging thật thì hạ `--average-period` xuống ~50.

**Decode / export**

| tham số | default | dự án dùng | ý nghĩa |
| --- | --- | --- | --- |
| `--decode_methods` | `all` | `all` | `greedy_search`, `modified_beam_search`, `beam_search` |
| `--use_averaged_model` | `0` | **1** | dùng `model_avg` |
| `--avg` | `1` | **10** | trung bình N checkpoint cuối |
| `--decode_chunk_size` | `32` | `32` | **một giá trị** (khác `--chunk_size` là list) |
| `--decode_left_context_frames` | `256` | `256` | một giá trị |
| `--streaming_decode_method` | `greedy_search` | – | chỉ dùng ở stage 15 |

**Finetune**

| tham số | default | ý nghĩa |
| --- | --- | --- |
| `--do_finetune` | `0` | đổi sang `finetune.py` |
| `--finetune_ckpt` | `""` | checkpoint khởi tạo (bắt buộc nếu bật) |
| `--init_modules` | `encoder` | module nào được nạp từ checkpoint |

### 4.2 Lệnh cho pipeline `main` hiện tại

```bash
GWEN_TTS_DIR=/path/to/gwen-tts bash run.sh \
  --data_tag main --build_clones 1 \
  --vocab_size 100 --model_size medium --causal 1 \
  --base_lr 0.045 --use_fp16 1 --max_duration 700 --num_workers 4 \
  --num_epochs 60 --use_averaged_model 1 --avg 10 \
  --enable_musan 1 --enable_spec_aug 1 --perturb_speed 0 \
  --offline_musan_aug 0 --exp_suffix "_lr0045" \
  --stage -2 --stop_stage 16
```

Nếu `transcripts_main/` đã có rồi và không muốn sinh clone lại, bắt đầu từ
`--stage 3` với các flag train/export giống trên.

### 4.3 `local/export_for_jetson.sh`

| tham số | ý nghĩa |
| --- | --- |
| `--exp-dir` | thư mục experiment |
| `--epoch` | epoch cuối để lấy |
| `--avg` | trung bình bao nhiêu checkpoint |
| `--streaming` | `1` = `export-onnx-streaming.py`, `0` = `export-onnx.py` |
| `--use-averaged-model` | `1` để dùng `model_avg` |
| `--out-dir` | thư mục gói model đầu ra |

Sau khi export ONNX, script gọi
`quantize_dynamic(..., weight_type=QuantType.QInt8)` và copy `tokens.txt`,
`bpe.model`, `bpe.vocab`.

### 4.4 Decode ONNX (`transcribe_beam_wav.py`, `jetson_asr.py`)

| tham số | default | ý nghĩa |
| --- | --- | --- |
| `--model-dir` | – | thư mục gói model |
| `--method` | `beam_search` | `beam_search` (classic, đa symbol) hoặc `greedy` |
| `--beam` | `4` | bề rộng beam |

`--threads`, `--provider`, `--max-active-paths`, `--fp32` được **nhận và bỏ qua**
(giữ để tương thích CLI cũ của sherpa).

### 4.5 `live_ui/server.py` + `live_ui/vad.py`

| tham số | default | ý nghĩa |
| --- | --- | --- |
| `--model-dir` | `model_medium_epoch30_avg10` | model deploy mặc định |
| `--beam` | `4` | beam width |
| `--host` / `--port` | `0.0.0.0` / `8100` | |
| `--no-vad` | tắt VAD | khi tắt: một hypothesis dài vô tận |
| `--vad-threshold` | `0.5` | ngưỡng vào speech |
| `--min-silence-ms` | `700` | im lặng bao lâu thì kết thúc một câu |

Hằng số trong code (không có flag):

| hằng số | giá trị | ý nghĩa |
| --- | --- | --- |
| `FRAME` | `512` | mẫu audio mỗi lần gọi Silero |
| `CONTEXT` | `64` | **bắt buộc** nối trước → graph nhận 576 mẫu |
| `Endpointer.lo` | `threshold - 0.15` | ngưỡng ra (hysteresis) |
| `min_speech_ms` | `250` | phải nói đủ lâu mới tính là speech |
| `speech_pad_ms` | `200` | đệm thêm trước khi chốt endpoint |
| `PREROLL_FRAMES` | `8` (256 ms) | phát lại audio trước onset vào decoder |
| `TAIL_FRAMES` | `6` (192 ms) | đuôi im lặng còn được đưa vào decoder |

### 4.6 `local/tts/`

| tham số | default | ý nghĩa |
| --- | --- | --- |
| `--source-tsv` | `transcripts_matched_u20/train.tsv` | danh sách câu nguồn |
| `--batch-size` | `8` | batch sinh TTS |
| `--max-duration` | `20.0` | loại clip ≥ 20 s |
| `--target-rms` | `0.09` | chuẩn hoá độ to output |
| `--peak-ceil` | `0.97` | trần peak |
| `GWEN_TTS_DIR` (env) | thư mục anh em | vị trí checkout Gwen-TTS |

`GEN_CFG`: `temperature=0.3, top_k=20, top_p=0.9, repetition_penalty=2.0,
max_new_tokens=4096`.

## 5. Vì sao benchmark của ta nói dối (đọc hai lần)

### 5.1 Tập "học thuộc"

`transcripts_matched_u20/{train,dev,test}.tsv` **giống nhau từng byte** (cùng
md5). Với tag `main`, 1.000 recording thật còn được lặp trong train. Nên "test
WER" thực chất là:

> cho một recording model đã được train trên đó, nó có tái tạo lại đúng cái nhãn
> nó đã học không?

Chỉ số này **mù trước hai thứ**:

1. **Nhãn sai.** Model học thuộc nhãn sai một cách **nhất quán** và vẫn đạt
   ~0,5%. Bug off-by-one không làm chỉ số này nhúc nhích.
2. **Khả năng tổng quát hoá sang giọng mới.** Nó không nói được UI mic có chạy
   với người thật hay không.

Nó không vô dụng — đó vốn là mục tiêu ban đầu — nhưng **không bao giờ được trích
dẫn như "WER của model"** mà thiếu câu cảnh báo đi kèm.

### 5.2 Tập held-out speaker

`eval_heldout_speaker.py`: 25 câu đã biết × 2 giọng **chưa từng có trong train**
(`yen_nhi` ♀, `khanh_toan` ♂). Đây mới là con số dự báo người lạ dùng được hay
không.

### 5.3 Hai chỉ số **mâu thuẫn nhau** — và đó là thông tin

Sweep `--avg` cũ trước khi thêm Hieu (classic `beam_search`, beam 4) vẫn hữu ích
vì nó cho thấy trade-off:

| avg | 800 recording thật (pre-Hieu) | held-out speakers |
| --- | ---: | ---: |
| 1 | 2.30% | 8.48% |
| 10 *(đang deploy)* | 2.18% | 9.17% |
| 30 | **1.54%** | 10.73% |

Averaging nhiều hơn thì **tốt hơn có ý nghĩa thống kê** trên chính những recording
đã train (avg30 vs avg10: −0,64 điểm, 95% CI [−0,79; −0,50], 20.790 từ) và **có
lẽ tệ hơn** trên giọng lạ (+1,54 điểm, 95% CI [−0,52; +3,62], P=0,92, chỉ 578
từ).

Đó chính là **memorize vs generalize**. `avg=30` chỉ khớp chặt hơn với những
recording ta đã có sẵn — **không phải cải thiện ngoài đời thật**. Giữ `avg=10`.

### 5.4 Statistical power — chỗ tôi đã sai

Tập held-out chỉ có **578 từ tham chiếu**. Chênh 4 từ = 0,69 điểm WER. Tôi từng
trích "9,17% so với 8,48%" như thể nó có nghĩa; bootstrap 50 utterance cho 95% CI
= [−0,86; +2,26]. **Tập này không phân giải nổi khác biệt dưới ~2 điểm WER.**
Mọi so sánh tôi từng làm dưới ngưỡng đó là **nhiễu đội lốt kết luận**.

Luật: trước khi tin một khác biệt WER, hãy **bootstrap-resample** các utterance
và lấy CI. Mười dòng code, và nó sẽ xoá một nửa kết luận của bạn.

### 5.5 Benchmark chưa ai làm

Tình huống dùng thật của bạn là **một bản thu mới của một giọng đã biết** — bạn
đọc một câu có trong dataset, nhưng là bản thu khác. Tập học thuộc là *cùng bản
thu*; tập held-out là *giọng khác*. **Không tập nào là tình huống của bạn.**

Hai mươi câu thu mới từ 5 người sẽ giải quyết dứt điểm mọi câu hỏi về `avg`,
decoder và khác biệt theo speaker. Nó **chưa tồn tại**.

---

## 6. Streaming thật sự

### 6.1 Vì sao model stream được

Encoder là causal và mang 74 state đi tiếp. Nó **luôn luôn** chạy được theo từng
chunk; runner cũ chỉ đơn giản là đưa cả utterance vào một lần.
`OnnxTransducer.encode()` đúng nghĩa là:

```python
while start + segment <= len(x):        # segment = 77
    chạy encoder trên x[start : start+77] với state hiện tại
    states = new_states
    start += offset                     # offset = 64
```

### 6.2 Vì sao decode tăng dần là **chính xác tuyệt đối**, không phải xấp xỉ

`beam_search()` **đồng bộ theo frame** — nó lặp qua từng encoder frame, mang theo
tập hypothesis `B` và `dec_cache`. `live_ui/stream_decoder.py` tách vòng lặp đó
thành `IncrementalBeam.step(enc_t)`. Cùng thuật toán, cùng beam, **cùng output**.

Không phải khẳng định suông — có kiểm chứng:

```bash
python3 live_ui/stream_decoder.py --self-check some_16k.wav
# assert: output streaming == output batch beam_search
```

Đúng trên cả 12 clip mẫu (2,9–9,8 s).

**Partial cập nhật mỗi 0,64 s** (`decode_chunk_len=64` × 10 ms). Con số này bị
"nướng" vào lúc export, không chỉnh được từ UI. RTF decode là 0,027–0,23, nên
**biên chunk là độ trễ duy nhất**, không phải tốc độ tính toán.

### 6.3 Silero VAD và ba cái bẫy

`live_ui/vad.py` chạy Silero dạng ONNX qua onnxruntime (không cần torch). Có
speech → mở một decoder session; im lặng 700 ms → chốt câu và mở session mới.
Bắt buộc phải vậy vì model được train trên câu đơn 4–5 s; một hypothesis dài mãi
sẽ trôi ra ngoài phân bố.

1. **Graph cần 576 mẫu**, không phải 512: 64 mẫu context + 512 mẫu mới. Đưa vào
   đúng 512 thì nó **không báo lỗi** — nó trả ≈0 cho mọi frame, tức "không bao
   giờ có speech". Bản đầu của tôi báo 0% speech trên file toàn speech.
   `vad.py --self-check` giờ assert rằng nó **thực sự tìm thấy** speech.
2. **Pre-roll.** Onset bị phát hiện trễ vài frame; phải phát lại 256 ms cuối vào
   decoder, nếu không âm vị đầu bị cắt.
3. **Tuyệt đối không đưa đuôi im lặng vào decoder.** Endpoint chỉ nổ sau 700 ms
   im lặng. Nếu đưa cả khoảng đó vào, model **bịa chữ** — output thật đã quan
   sát: `"...dừng lại và t"`, `"...mùa hạn chỉ"`. Audio bị chặn ngay khi chuỗi im
   lặng bắt đầu, chỉ giữ đuôi 192 ms cho phụ âm cuối.

---

## 7. Deploy

### 7.1 Precision

| | trạng thái |
| --- | --- |
| **int8** | mặc định. Nhanh hơn 35%, nhỏ hơn 3×, **số lỗi y hệt fp32** (đã đo) |
| fp32 | chạy được, không lợi gì về độ chính xác ở đây |
| **fp16** | **không load được** |

fp16 chết ngay lúc tạo session vì graph Zipformer còn `Cast` node lẫn fp16/fp32:

```
Type Error: Type (tensor(float16)) of output arg
(/feed_forward1/out_proj/Cast_output_0) does not match expected type (tensor(float))
```

`--use-fp16` lúc train (mixed-precision autocast) và `--fp16` lúc export (đổi
weight) là **hai chuyện khác nhau**. Đừng nhầm.

### 7.2 Runtime: đừng dùng sherpa-onnx

sherpa-onnx là runtime chuẩn của k2 và nó **sai cho model này** (xem §1.5). Ta
ship `jetson_beam_decode.py`: **chỉ numpy + onnxruntime**, không torch, không k2,
không sherpa. Vừa chính xác hơn **vừa nhanh hơn** cái nó thay thế.

> **Đính chính.** Tôi từng viết ~9,2% của sherpa là "sàn cấu trúc cứng". **Sai.**
> Phần lớn là do bug nhãn — sau khi sửa, chính `modified_beam_search` đạt 3,31%.
> *Khoảng cách giữa hai decoder* là có thật về cấu trúc (3,31% vs 0,72% trên cùng
> weight); nhưng **độ lớn** của nó đã bị bug dữ liệu thổi phồng. Hãy đo **cả hai**
> decoder trên mọi model mới.

### 7.3 Môi trường Jetson Nano (JetPack 4 / L4T R32.7.6 / Python 3.6.9 / CUDA 10)

| thứ | yêu cầu | hỏng thế nào nếu sai |
| --- | --- | --- |
| `onnxruntime` | **1.10.0** | wheel mới cần glibc/CUDA không có |
| `numpy` | **1.19.5** | numpy 1.13.3 của apt thiếu `numpy.core._multiarray_umath` → **segfault** |
| `kaldi-native-fbank` | build từ source | không có wheel aarch64 cho py3.6 |
| **`OPENBLAS_CORETYPE=ARMV8`** | phải export | numpy nhận nhầm CPU → **"Illegal instruction"**, không thông báo gì |

`~/.bashrc` có set `OPENBLAS_CORETYPE`, **nhưng `.bashrc` thoát sớm với shell
không tương tác**, nên mọi lệnh `ssh host 'command'` phải tự export.

- Wheel pip `sherpa_onnx` **chỉ có CPU** và **âm thầm** fallback khi bạn truyền
  `provider="cuda"`. Muốn GPU phải build source với `-DSHERPA_ONNX_ENABLE_GPU=ON`.
- **GPU không đáng** cho UI một câu: CPU ≈7 s/clip; GPU ≈10 s (nóng), ≈30 s
  (lạnh, khởi tạo CUDA mỗi lần gọi). GPU chỉ thắng khi eval hàng loạt.
- **TensorRT không chạy được.** Non-streaming bị từ chối thẳng
  (`Tensorrt support for Online models only`); streaming fp32 không build xong
  engine trong 300 s; streaming int8 abort vì thiếu shape ở một MatMul đã
  quantize.

### 7.4 Vì sao chưa deploy non-streaming

Encoder non-streaming nhận **2 input** (`x[N,T,80]`, `x_lens[N]`) và **không có**
metadata chunk. `jetson_beam_decode.py` đọc `T` / `decode_chunk_len` và dựng 74
state tensor → nó sẽ **crash ngay lúc load**. Port thì nhỏ (một forward thay cho
vòng chunk) nhưng **chưa viết**. Không có nó, non-streaming phải quay về
sherpa-onnx với decoder yếu hơn. Ngoài ra non-streaming **không thể** cho partial
result, và bộ nhớ tăng theo độ dài utterance.

---

## 8. Giới hạn, nói thẳng

- **Câu dài nằm ngoài phân bố.** Câu train chủ yếu là câu ngắn khoảng 4–7 s, nên
  clip rất dài gần như không xuất hiện. Đã từng quan sát clip 19 s decode ra
  **chuỗi rỗng** ở model archived. Ổn với câu lệnh ngắn, **không dùng được** cho
  audio dài.
- **`yen_nhi` (♀) gánh gần hết lỗi held-out còn lại** — 49/53 từ. Một giọng nữ
  trong train đã kéo cô ấy từ 92% → 17%. Thêm giọng nữ là đòn bẩy rõ ràng nhất.
- **Lỗi theo từng speaker vẫn quan trọng.** Ở model medium cuối, Trung vẫn khó
  hơn Quan/Dung/Hieu. Khi chọn model, đọc WER từng người chứ không chỉ đọc trung
  bình.
- **`average_period=200` > 159 batch/epoch** → checkpoint averaging làm ít hơn
  bạn tưởng (xem §4.1).
- **Tập held-out thiếu statistical power** (578 từ, phân giải ~2 điểm WER).
- **Mọi thứ train trước 2026-07-09 đều dùng nhãn sai** và đã bị xoá.

---

## 9. Sổ tay debug

**Model ra rác / chuỗi rỗng.**
1. Feature parity: waveform có nằm trong `[-1,1]` không? 16 kHz mono chưa? fbank
   opts có giống `fbank_opts()` không?
2. Đã `reset()` 74 encoder state giữa hai utterance chưa?
3. Clip có dài hơn ~15 s không? Ngoài phân bố → toàn blank → rỗng.

**WER đẹp nhưng nói thật thì trượt.**
Bạn đang cầm một chỉ số học thuộc và có bug nhãn. Chạy kiểm tra alignment:
`corr(audio_duration, text_word_count)` theo speaker. Rồi decode một wav **trong
train** (phải gần như hoàn hảo) và so với một bản thu mới cùng câu.

**Toàn blank / mất chữ hàng loạt.**
Nhiều khả năng bạn đang dùng `greedy_search` hoặc `modified_beam_search` — chỉ
phát ≤1 symbol/frame. Dùng classic `beam_search`. Đọc **số deletion** trong error
stats, đừng chỉ đọc WER.

**Một khác biệt WER trông có vẻ đáng kể.**
Bootstrap-resample utterance và lấy CI. Trên tập held-out, dưới ~2 điểm là nhiễu.

**`pgrep -f` / `pkill -f` khớp với chính command line của nó.**
Đã cắn tôi **ba lần**, kể cả bên trong đoạn guard tôi viết ra để tránh nó. Một
monitor chạy nền mà grep chính pattern của mình thì hoặc tự giết mình, hoặc lặp
vô tận. Hãy grep **file log**, hoặc dùng bracket class (`serve[r].py`). Ngoài ra
`pkill` trả về 1 (không khớp) sẽ **abort** cả compound command dưới `set -e`.

**Bẫy khác.**
- `/tmp` bị xoá khi reboot. Giữ log train trên đĩa dự án.
- Mất điện làm checkpoint đang ghi thành **0 byte**. Xoá nó và resume bằng
  `--start_epoch N` (nó nạp `epoch-(N-1).pt`). Không mất gì thêm.
- Export là `run.sh` stage 16, bên dưới gọi `local/export_for_jetson.sh`.
- Chọn checkpoint theo validation loss là **sai** ở đây — dev == train.
- `sort -u` đếm sai từ vựng tiếng Việt (xem §2).

---

## 10. Nếu quay lại dự án này

Theo thứ tự giá trị kỳ vọng:

1. **Thu 20 câu mới cho mỗi speaker** (câu đã có trong dataset). Đó mới là điều
   kiện triển khai thật và chưa benchmark nào phủ. Nó giải quyết một lúc cả câu
   hỏi `avg`, decoder và khác biệt theo speaker.
2. **Thêm giọng nữ tham chiếu** vào bộ clone. `yen_nhi` chiếm 92% lỗi held-out
   còn lại.
3. **Mở rộng tập held-out** vượt 578 từ để phân giải được dưới 2 điểm WER.
4. Kiểm tra xem khoảng cách dai dẳng của Trung có phải do de-clipping không:
   chấm audio gốc (còn clipped) so với audio đã sửa trên cùng một model.
5. Hạ `--average-period` xuống ~50 và xem checkpoint averaging có hoạt động đúng
   như tài liệu không.
