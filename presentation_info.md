# Presentation Notes: Vietnamese ASR Pipeline Experiments

## 1. Mục Tiêu Thí Nghiệm

Mục tiêu chính là xây dựng và kiểm chứng một pipeline ASR tiếng Việt chạy được end-to-end bằng icefall/Zipformer:

- Chuẩn bị dữ liệu âm thanh và transcript.
- Tạo manifest, fbank feature, BPE vocabulary.
- Train mô hình Zipformer Transducer.
- Decode và đo WER.
- Xuất TorchScript/JIT để thử nhận dạng từ microphone.

Do dữ liệu ban đầu nhỏ, mục tiêu thực nghiệm không phải chứng minh khả năng tổng quát hóa ASR tiếng Việt ở quy mô production, mà là:

- Chạy được full pipeline ổn định.
- Overfit/fit tốt tập câu mục tiêu.
- So sánh có kiểm soát giữa dataset, model size, checkpoint averaging, noise reduce và raw audio.
- Chọn cấu hình tốt nhất cho demo microphone.

## 2. Thông Số Bộ Dữ Liệu

### 2.1. `vi_asr_corpus`

Đây là bộ dữ liệu nhỏ, chủ yếu dùng để sanity-check pipeline và demo nhận dạng một số câu cố định.

| Split | Số utterance | Số câu unique | Số speaker | Tổng thời lượng |
|---|---:|---:|---:|---:|
| train | 408 | 3 | 5 | 1h 01m 43s |
| dev | 21 | 3 | 5 | 3m 15s |
| test | 25 | 3 | 5 | 3m 51s |
| **Total** | **454** | **3** | **5** | **1h 08m 49s** |

Phân bố số câu theo speaker:

| Speaker | Train | Dev | Test | Total |
|---|---:|---:|---:|---:|
| Dung | 81 | 4 | 5 | 90 |
| HIEU | 81 | 4 | 5 | 90 |
| Khoi | 93 | 5 | 6 | 104 |
| Quan | 72 | 4 | 4 | 80 |
| Trung | 81 | 4 | 5 | 90 |

Phân bố thời lượng theo speaker:

| Speaker | Train | Dev | Test | Total |
|---|---:|---:|---:|---:|
| Dung | 9m 13s | 0m 32s | 0m 34s | 10m 20s |
| HIEU | 12m 13s | 0m 35s | 0m 45s | 13m 33s |
| Khoi | 17m 38s | 0m 47s | 1m 11s | 19m 36s |
| Quan | 10m 06s | 0m 41s | 0m 30s | 11m 16s |
| Trung | 12m 32s | 0m 41s | 0m 51s | 14m 03s |

Đặc điểm:

- Chỉ có 3 nội dung câu chính, được lặp lại bởi nhiều speaker.
- Tổng thời lượng chỉ khoảng 1.15 giờ, nên train nhanh và dễ thử nhiều cấu hình.
- Phù hợp để kiểm tra pipeline và overfit nhanh.
- Không đại diện cho ASR tiếng Việt tổng quát.
- Kết quả WER thấp trên tập này chủ yếu cho thấy model học tốt các câu mục tiêu.

### 2.2. `VietnameseASR`

Đây là bộ dữ liệu thực tế hơn, nhiều câu unique hơn và khó hơn.

| Split | Số utterance | Số câu unique | Số speaker | Tổng thời lượng |
|---|---:|---:|---:|---:|
| train | 657 | 657 | 4 | 2h 40m 35s |
| dev | 36 | 36 | 4 | 8m 24s |
| test | 37 | 37 | 4 | 8m 52s |
| **Total** | **730** | **730** | **4** | **2h 57m 51s** |

Phân bố số câu theo speaker:

| Speaker | Train | Dev | Test | Total |
|---|---:|---:|---:|---:|
| Dung | 117 | 6 | 7 | 130 |
| Khoi | 180 | 10 | 10 | 200 |
| Quan | 180 | 10 | 10 | 200 |
| Trung | 180 | 10 | 10 | 200 |

Phân bố thời lượng theo speaker:

| Speaker | Train | Dev | Test | Total |
|---|---:|---:|---:|---:|
| Dung | 35m 42s | 1m 45s | 2m 07s | 39m 34s |
| Khoi | 28m 44s | 1m 55s | 1m 54s | 32m 34s |
| Quan | 34m 17s | 1m 56s | 2m 01s | 38m 14s |
| Trung | 1h 01m 52s | 2m 47s | 2m 50s | 1h 07m 29s |

Đặc điểm:

- Mỗi utterance gần như là một câu khác nhau.
- Tổng thời lượng khoảng 2.96 giờ, lớn hơn `vi_asr_corpus` nhưng vẫn nhỏ với bài toán ASR tổng quát.
- Khó overfit hơn `vi_asr_corpus`.
- Phù hợp hơn để đánh giá khả năng ASR thực tế, nhưng dữ liệu vẫn còn nhỏ so với yêu cầu train ASR production.

### 2.3. NoiseReduce Variant `nr`

Biến thể `nr` được tạo bằng thư viện `noisereduce`, không ghi đè audio gốc.

| Variant | Audio root | Transcript dir | Feature dir | Ghi chú |
|---|---|---|---|---|
| raw | `audio/` | `transcripts/` | `fbank/` | Dữ liệu gốc |
| nr | `audio_nr/` | `transcripts_nr/` | `fbank_nr/` | Audio đã noise reduce |

Sau khi regenerate, `transcripts_nr` giữ cùng `utt_id`, `speaker`, `text` với raw; chỉ thay `audio_path`. Nhờ vậy raw/nr có thể so sánh công bằng trên cùng nội dung test.

Thời lượng và số utterance của `nr` giống raw:

| Split | Số utterance | Tổng thời lượng |
|---|---:|---:|
| train | 408 | 1h 01m 43s |
| dev | 21 | 3m 15s |
| test | 25 | 3m 51s |
| **Total** | **454** | **1h 08m 49s** |

## 3. Thông Số Setup

### 3.1. Framework và pipeline

| Thành phần | Giá trị |
|---|---|
| Framework | icefall |
| Model family | Zipformer / Zipformer2 |
| Objective | Pruned RNN-T / Transducer |
| Feature | fbank 80 bins |
| Tokenizer | SentencePiece BPE |
| BPE vocab size | 100 |
| Decode beam size | 4 |
| Primary decode method | `beam_search` |
| Training device | 1 GPU |
| Main scratch learning rate | `base_lr=0.01` |
| Finetune learning rate | `base_lr=0.001` |

### 3.2. Model configurations

| Model | Encoder dims | Layers | Decoder dim | Joiner dim | Params |
|---|---|---|---:|---:|---:|
| base | 192,256,384,512,384,256 | 2,2,3,4,3,2 | 512 | 512 | ~65M |
| small | 192,256,256,256,256,256 | 2,2,2,2,2,2 | 512 | 512 | ~23M |

Trong code, số params thực tế từng được log:

- base: khoảng 65.29M params.
- small: khoảng 22.57M params.

### 3.3. Training modes

| Mode | Learning rate | Ý nghĩa |
|---|---:|---|
| scratch50 | 0.01 | Train từ random initialization trong 50 epoch |
| scratch30 | 0.01 | Train từ random initialization trong 30 epoch |
| finetune30 | 0.001 | Load LibriSpeech pretrained encoder, train 30 epoch |
| streaming_scratch50 | 0.01 | Causal/streaming Zipformer, train scratch 50 epoch |
| streaming_finetune30 | 0.001 | Causal/streaming Zipformer, finetune 30 epoch |
| nr_scratch30 | 0.01 | Train trên audio đã noise reduce trong 30 epoch |

Ghi chú về learning rate:

- Ban đầu learning rate thấp hơn cho kết quả kém hơn, sau đó tăng `base_lr` lên `0.01` giúp WER giảm rõ rệt.
- Các thí nghiệm scratch chính trong báo cáo dùng `base_lr=0.01`.
- Các thí nghiệm finetune dùng `base_lr=0.001` để tránh phá pretrained encoder quá nhanh.

## 4. Metric Đánh Giá

Metric chính là **WER**:

```text
WER = (Substitutions + Insertions + Deletions) / Number of reference words
```

Diễn giải:

- WER càng thấp càng tốt.
- Trên `vi_asr_corpus`, WER thấp cho thấy model học tốt các câu cố định, không chứng minh tổng quát hóa.
- Trên `VietnameseASR`, WER phản ánh gần hơn năng lực ASR thực tế.

Ngoài WER, thí nghiệm còn ghi nhận:

- Thời gian huấn luyện.
- Số tham số model.
- Ảnh hưởng của decode method.
- Ảnh hưởng của checkpoint averaging.
- Ảnh hưởng của noise reduce.

## 5. So Sánh Dataset: `VietnameseASR` vs `vi_asr_corpus`

### 5.1. Kết quả tốt nhất trên mỗi dataset

| Dataset | Best config | Best WER | Training time liên quan |
|---|---|---:|---:|
| vi_asr_corpus | small, scratch30 rerun, avg=10, beam_search | **3.40%** | ~12 min |
| VietnameseASR | small, scratch50, beam_search | **74.87%** | ~41 min |
| VietnameseASR x10 | small, scratch30, avg=15, beam_search | **65.91%** | khoảng 30 epoch |

### 5.2. Nhận xét

`VietnameseASR` thực tế hơn vì có nhiều câu unique hơn, nhưng WER vẫn rất cao. Nguyên nhân chính là dữ liệu chưa đủ lớn cho ASR tiếng Việt tổng quát, trong khi số epoch và vocab size vẫn còn khá nhỏ.

Run `x10` trên `VietnameseASR` cho thấy tăng dữ liệu lặp/augmentation giúp WER tốt hơn đáng kể, từ mốc khoảng 74-75% xuống 65.91% khi dùng checkpoint averaging `avg=15` và `beam_search`. Tuy vậy mức này vẫn còn cao cho demo nhận dạng ổn định.

`vi_asr_corpus` nhỏ hơn nhiều, chỉ có 3 câu unique, nên model có thể fit nhanh. Dataset này phù hợp hơn với mục tiêu hiện tại:

- Chứng minh pipeline từ prepare data đến train/decode/export/mic hoạt động.
- Train nhanh, dễ thử nhiều cấu hình.
- Có thể tạo demo nhận dạng 3 câu mục tiêu.

Vì vậy hướng thực nghiệm được chuyển sang `vi_asr_corpus` để tối ưu pipeline và demo, thay vì cố giảm WER trên `VietnameseASR` khi dữ liệu chưa đủ.

## 6. So Sánh Model Size: base vs small

### 6.1. Trên `vi_asr_corpus`

| Model | Mode | Checkpoint | Best WER |
|---|---|---|---:|
| base | scratch50 | epoch-50 | 25.54% |
| small | scratch50 | epoch-30 | 4.62% |
| small | scratch30 rerun | epoch-30 avg=10 | **3.40%** |

Nhận xét:

- `small` tốt hơn `base` rõ rệt trên `vi_asr_corpus`.
- Với dataset chỉ có 3 câu unique, model lớn hơn không đem lại lợi ích.
- `base` có nhiều params hơn, train lâu hơn và khó fit ổn định hơn trên dataset quá nhỏ.
- `small` đủ capacity để học 3 câu, đồng thời hội tụ nhanh hơn.

### 6.2. Trên `VietnameseASR`

| Model | Mode | Best WER |
|---|---|---:|
| base | scratch50 | 76.26% |
| small | scratch50 | **74.87%** |

Nhận xét:

- `small` vẫn nhỉnh hơn `base`.
- Chênh lệch không lớn, nhưng cho thấy dataset hiện tại chưa đủ để tận dụng capacity của `base`.
- Với dữ liệu hạn chế, `small` là lựa chọn thực dụng hơn.

## 7. So Sánh Train From Scratch vs Finetune

Sau khi chọn hướng dùng Zipformer `small`/`base`, một câu hỏi tự nhiên là nên train từ đầu hay tận dụng pretrained LibriSpeech. Vì vậy pipeline có thêm nhánh `finetune30`.

### 7.1. Thiết lập finetune

| Thành phần | Scratch | Finetune |
|---|---|---|
| Khởi tạo | Random initialization | Load pretrained LibriSpeech encoder |
| Init modules | Toàn bộ model train từ đầu | `--init_modules encoder` |
| Epoch | 50 hoặc 30 | 30 |
| Learning rate | 0.01 | 0.001 |
| Pretrained source | Không dùng | LibriSpeech tiếng Anh |
| BPE vocab đang dùng | 100 tiếng Việt | 100 tiếng Việt |

Điểm quan trọng: pretrained LibriSpeech dùng tiếng Anh và vocab khác, trong khi thí nghiệm này dùng BPE vocab 100 cho tiếng Việt. Vì vậy decoder/joiner không được hưởng lợi nhiều từ pretrained checkpoint; phần hữu ích nhất chỉ là encoder acoustic.

### 7.2. Kết quả finetune trên `vi_asr_corpus`

| Model | Mode | LR | Epoch | Best WER |
|---|---|---:|---:|---:|
| base | scratch50 | 0.01 | 50 | **25.54%** |
| base | finetune30 | 0.001 | 30 | 48.91% |
| small | scratch50 | 0.01 | 30/50 | **4.62% / 8.56%** |
| small | scratch30 rerun | 0.01 | 30 | **3.40%** |
| small | finetune30 | 0.001 | 30 | 62.36% |

Trên toy corpus, finetune kém hơn train from scratch rõ rệt. Với dữ liệu chỉ có 3 câu unique, model nhỏ train từ đầu có thể fit nhanh hơn nhiều.

### 7.3. Kết quả finetune trên `VietnameseASR`

| Model | Mode | LR | Epoch | Best WER |
|---|---|---:|---:|---:|
| base | scratch50 | 0.01 | 50 | **76.26%** |
| base | finetune30 | 0.001 | 30 | 97.90% |
| small | scratch50 | 0.01 | 50 | **74.87%** |
| small | finetune30 | 0.001 | 30 | 98.49% |

Trên `VietnameseASR`, finetune gần như thất bại. WER xấp xỉ 98-99%, thấp hơn rất nhiều so với scratch.

### 7.4. Nhận xét

Finetune thất bại trong setup này vì:

- Pretrained model là LibriSpeech tiếng Anh, khác ngôn ngữ và khác phân bố âm vị so với tiếng Việt.
- Vocab tiếng Việt chỉ có 100 BPE, không khớp với vocab pretrained.
- Chỉ load encoder, trong khi decoder/joiner vẫn phải học lại gần như từ đầu.
- 30 epoch chưa đủ để decoder/joiner học ổn định trên dữ liệu nhỏ.
- LR finetune `0.001` an toàn hơn cho pretrained encoder nhưng có thể quá chậm khi decoder/joiner mới cần học nhanh.

Kết luận:

- Với mục tiêu demo hiện tại, không dùng finetune LibriSpeech.
- Train from scratch với `small`, LR `0.01`, rồi dùng checkpoint averaging là hướng tốt nhất.
- Nếu muốn finetune nghiêm túc, cần pretrained tiếng Việt hoặc multilingual, hoặc finetune lâu hơn và xử lý lại vocab/init modules cẩn thận hơn.

## 8. So Sánh Decode Method

Các decode method đã thử:

| Method | Đặc điểm | Kết quả quan sát |
|---|---|---|
| greedy_search | Chọn token tốt nhất tại mỗi bước | Nhanh nhưng WER thường kém nhất |
| modified_beam_search | Beam search biến thể đơn giản hơn | Có lúc tốt hơn greedy, nhưng không ổn định bằng beam_search |
| beam_search | Giữ nhiều hypothesis rồi chọn chuỗi tốt nhất | Ổn định nhất và thường tốt nhất |

Ví dụ trên `small/scratch30 rerun`, epoch-30 avg=1:

| Decode method | WER |
|---|---:|
| greedy_search | 57.88% |
| modified_beam_search | 43.89% |
| beam_search | **26.90%** |

Kết luận:

- `beam_search` là decode method chính cho báo cáo và demo.
- Greedy không phù hợp với mô hình hiện tại vì WER cao.

## 9. Vai Trò Của Checkpoint Averaging (`avg`)

Checkpoint averaging là kỹ thuật lấy trung bình trọng số của nhiều checkpoint cuối, ví dụ `avg=10` nghĩa là trung bình epoch 21 đến epoch 30 khi decode/export epoch 30.

### 9.1. Kết quả trên `small/scratch30 rerun`

| Checkpoint | WER beam_search |
|---|---:|
| epoch-30 avg=1 | 26.90% |
| epoch-30 avg=3 | 19.97% |
| epoch-30 avg=5 | 8.42% |
| epoch-30 avg=7 | 4.76% |
| epoch-30 avg=10 | **3.40%** |

### 9.2. Kết quả trên `small/scratch30` cũ

| Checkpoint | WER beam_search |
|---|---:|
| epoch-30 avg=1 | 18.89% |
| epoch-30 avg=3 | 7.74% |
| epoch-30 avg=5 | 7.61% |
| epoch-30 avg=7 | **6.66%** |
| epoch-30 avg=10 | 8.97% |

### 9.3. Nhận xét

Checkpoint averaging là thay đổi quan trọng nhất sau khi chọn `small`:

- Giảm WER từ 26.90% xuống 3.40% trên rerun.
- Giúp model ổn định hơn trên dataset nhỏ.
- Một checkpoint đơn lẻ có thể rất nhiễu, đặc biệt khi dataset chỉ có 3 câu.

Kết luận:

- Không nên báo cáo chỉ `epoch-30 avg=1`.
- Với `vi_asr_corpus`, nên luôn decode thêm `avg=5`, `avg=7`, `avg=10`.
- Model dùng cho microphone hiện tại được export từ best checkpoint: `small/scratch30 rerun epoch-30 avg=10`.

### 9.4. Kết quả averaged trên `VietnameseASR x10`

| avg | greedy_search | modified_beam_search | beam_search |
|---:|---:|---:|---:|
| 5 | 72.10% | 70.95% | 67.10% |
| 10 | 71.33% | 70.18% | 66.60% |
| 15 | 71.10% | 69.98% | **65.91%** |
| 20 | 70.95% | 69.64% | 66.87% |

Nhận xét:

- `avg=15` là mốc tốt nhất trong nhóm đã thử.
- `beam_search` vẫn là method tốt nhất.
- Averaging quá ít (`avg=5`) chưa đủ ổn định; averaging quá nhiều (`avg=20`) có thể kéo vào các epoch cũ hơn và làm WER tăng lại.

## 10. NoiseReduce: raw vs nr

NoiseReduce được thêm như một biến thể offline:

- Input raw audio từ `audio/`.
- Output audio đã lọc noise vào `audio_nr/`.
- Transcript tương ứng nằm ở `transcripts_nr/`.
- Feature tương ứng nằm ở `fbank_nr/`.
- Không ghi đè audio gốc.
- Không dùng trong microphone mặc định; microphone hiện dùng raw audio trực tiếp, VAD chỉ cắt đoạn nói.

### 10.1. Kết quả NR end-to-end

| Model | Test audio | Best WER |
|---|---|---:|
| raw small/scratch30 rerun | raw | **3.40%** |
| nr small/nr_scratch30 | nr | 9.24% |

Raw vẫn tốt hơn NR khi train/test cùng domain.

### 10.2. Kiểm chéo raw/nr

| Train model | Test raw | Test NR | Best avg |
|---|---:|---:|---|
| raw small/scratch30 rerun | **3.40%** | 8.42% | avg=10 |
| NR small/nr_scratch30 | 57.07% | 9.24% | avg=7/10 |

Nhận xét:

- Raw model là tốt nhất trên raw test.
- Raw model test trên NR audio vẫn đạt 8.42%, tốt hơn NR model test trên NR audio 9.24%.
- NR model bị lệch domain mạnh: train NR nhưng test raw tăng lên 57.07% WER.
- NoiseReduce không tạo lợi thế rõ ràng trên corpus hiện tại.

Kết luận:

- Không nên dùng NR làm pipeline chính cho demo 3 câu.
- Nếu microphone thực tế rất nhiễu, hướng ít rủi ro hơn là giữ raw model và thử noise reduce ở inference.
- Train riêng model NR chưa đáng ưu tiên vì không thắng raw baseline.

## 11. Thời Gian Huấn Luyện

### 11.1. `vi_asr_corpus`

| Experiment | LR | Duration |
|---|---:|---:|
| base / scratch50 | 0.01 | ~28 min |
| base / finetune30 | 0.001 | ~21 min |
| base / streaming_scratch50 | 0.01 | ~30 min |
| base / streaming_finetune30 | 0.001 | ~21 min |
| small / scratch50 | 0.01 | ~19 min |
| small / scratch30 | 0.01 | ~13 min |
| small / scratch30 rerun | 0.01 | ~12 min |
| small / nr_scratch30 | 0.01 | ~12 min |
| small / finetune30 | 0.001 | ~13 min |
| small / streaming_scratch50 | 0.01 | ~20 min |
| small / streaming_finetune30 | 0.001 | ~13 min |

### 11.2. `VietnameseASR`

| Experiment | LR | Duration |
|---|---:|---:|
| base / scratch50 | 0.01 | ~54 min |
| base / finetune30 | 0.001 | ~34 min |
| base / streaming_scratch50 | 0.01 | ~59 min |
| base / streaming_finetune30 | 0.001 | ~34 min |
| small / scratch50 | 0.01 | ~41 min |
| small / finetune30 | 0.001 | ~24 min |
| small / streaming_scratch50 | 0.01 | ~40 min |
| small / streaming_finetune30 | 0.001 | ~23 min |

Nhận xét:

- `vi_asr_corpus` cho vòng thử nghiệm nhanh hơn nhiều.
- `small/scratch30` chỉ mất khoảng 12-13 phút, phù hợp để thử nhiều biến thể.
- `VietnameseASR` tốn thời gian hơn nhưng WER vẫn cao, nên chưa hiệu quả cho mục tiêu demo hiện tại.

## 12. Kết Quả Demo Microphone

Model hiện tại dùng cho microphone:

| Thành phần | Giá trị |
|---|---|
| Script | `mic_streaming_asr.py` |
| Mode chính | `full` |
| Model JIT | `ASR/zipformer/exp_bpe100_small_scratch30_20260506_194241/jit_script.pt` |
| Source checkpoint | epoch-30 avg=10 |
| Decode | beam search, beam=4 |

Smoke test JIT trên 3 câu unique trong test set:

| Câu | WER |
|---|---:|
| `trong phòng yên tĩnh...` | 34.48% |
| `tôi ghi âm dữ liệu thử nghiệm...` | 0.00% |
| `hôm nay tôi kiểm tra hệ thống...` | 0.00% |
| Tổng trên 3 mẫu unique | 11.36% |

Kết quả thử trực tiếp với microphone khi đọc các cụm đầu câu trong dữ liệu train:

| Prompt đọc vào mic | Một số hypothesis model sinh ra |
|---|---|
| `tôi ghi âm dữ liệu thử nghiệm` | `tôi ghi âm dữ liệu thử nghiệm vào lúc tám giờ ba mươi phút hôm nay`; `tôi ghi âm dữ liệu thử nghiệm vào lúc tám giờ ba mươi phút trong phòng yên tĩnh để kiểm tra hệ thống nhận dạng tiếng nói`; `tôi ghi âm dữ liệu thử nghiệm vào lúc tám giờ ba mươi phút hôm nay`; `tôi ghi âm dữ liệu thử nghiệm vào lúc tám giờ ba mươi phút hôm nay` |
| `hôm nay tôi kiểm tra hệ thống nhận dạng tiếng nói` | `hôm nay t`; `hôm nay`; `hôm nay`; `hôm nay` |
| `trong phòng yên tĩnh vào lúc tám giờ ba mươi phút` | `hôm nay`; `hôm nay`; `hôm nay tôi kiểm tra hệ thống nhận dạng`; `hôm nay`; `hôm nay tôi kiểm tra hệ thống nhận dạng` |

Nhận xét:

- Câu `trong phòng yên tĩnh...` là câu khó nhất.
- Ba câu có nhiều cụm từ trùng nhau, nên model dễ nhầm thứ tự cụm.
- Khi đọc cụm ngắn thay vì nguyên câu, model hay kéo kết quả về cụm có prior mạnh như `hôm nay...`.
- Cụm `trong phòng yên tĩnh vào lúc tám giờ ba mươi phút` thường bị nhận sai thành `hôm nay...`, cho thấy model chưa học tốt các prefix ngắn/partial utterance.
- Model hiện được train trên câu đầy đủ, nên đọc từng cụm rời không hoàn toàn cùng phân phối với dữ liệu train/test.
- Kết quả mic thực tế phụ thuộc cách nói, khoảng cách mic, VAD và noise môi trường.

## 13. Mạch Kết Luận Cho Báo Cáo

### Bước 1: Chạy trên dataset thực tế `VietnameseASR`

Kết quả tốt nhất chỉ đạt 74.87% WER. Điều này cho thấy dữ liệu vẫn quá nhỏ để train ASR tiếng Việt tổng quát. Dù dataset thực tế hơn, nó chưa phù hợp cho mục tiêu demo nhanh.

### Bước 2: Chuyển sang `vi_asr_corpus`

`vi_asr_corpus` chỉ có 3 câu unique, nên phù hợp để kiểm tra pipeline và tạo demo nhận dạng các câu mục tiêu. Training nhanh hơn nhiều và WER giảm mạnh.

### Bước 3: So sánh base và small

Small tốt hơn base trên cả hai dataset trong điều kiện dữ liệu hạn chế. Với `vi_asr_corpus`, small đạt 3.40-4.62% WER, còn base tốt nhất chỉ đạt 25.54%.

### Bước 4: So sánh scratch và finetune

Finetune từ LibriSpeech không hiệu quả trong setup hiện tại. Trên `vi_asr_corpus`, small finetune30 chỉ đạt 62.36% WER, kém xa small scratch30 avg=10 đạt 3.40%. Trên `VietnameseASR`, finetune cũng gần như thất bại với WER khoảng 98-99%. Vì vậy hướng chính là train from scratch.

### Bước 5: Chọn beam_search

Beam search tốt hơn greedy và modified beam trong các run chính. Vì vậy decode và export demo dùng beam search với beam size 4.

### Bước 6: Áp dụng checkpoint averaging

Checkpoint averaging là yếu tố cải thiện lớn nhất. Với `small/scratch30 rerun`, WER giảm từ 26.90% (`avg=1`) xuống 3.40% (`avg=10`).

### Bước 7: Thử NoiseReduce

NoiseReduce không thắng raw baseline:

- Raw→Raw: 3.40%
- Raw→NR: 8.42%
- NR→NR: 9.24%
- NR→Raw: 57.07%

Do đó raw pipeline được giữ làm hướng chính.

### Bước 8: Xuất model cho microphone

Model tốt nhất được export sang TorchScript/JIT và gắn vào `mic_streaming_asr.py` để thử nhận dạng từ microphone.

## 14. Kết Luận Cuối

Cấu hình tốt nhất hiện tại cho demo:

| Thành phần | Lựa chọn |
|---|---|
| Dataset | `vi_asr_corpus` |
| Audio variant | raw |
| Model | Zipformer small |
| Training | scratch30 |
| Learning rate | 0.01 |
| Checkpoint | epoch-30 avg=10 |
| Decode | beam_search, beam size 4 |
| Best WER trên full test | **3.40%** |
| Training time | ~12 min |

Thông điệp chính:

- Với dữ liệu nhỏ, model nhỏ hiệu quả hơn model lớn.
- Finetune LibriSpeech không phù hợp trong setup hiện tại.
- Checkpoint averaging rất quan trọng.
- NoiseReduce không cải thiện kết quả trong setup hiện tại.
- Pipeline đã chạy end-to-end và đã có model JIT dùng được cho microphone demo.
