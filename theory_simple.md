# Lý thuyết đơn giản: Pipeline ASR từ audio đến văn bản

> Tài liệu này giải thích pipeline nhận dạng tiếng nói (ASR) ở mức dễ hiểu,  
> phù hợp để đọc nhanh, hiểu bản chất và trình bày lại.

---

## Mục lục

1. [Tổng quan bài toán ASR](#1-tổng-quan-bài-toán-asr)
2. [Pipeline tổng quát](#2-pipeline-tổng-quát)
3. [Kiến trúc model](#3-kiến-trúc-model)
4. [Hàm loss](#4-hàm-loss)
5. [Quá trình huấn luyện](#5-quá-trình-huấn-luyện)
6. [Checkpoint và average nhiều epoch](#6-checkpoint-và-average-nhiều-epoch)
7. [Decode: từ model ra văn bản](#7-decode-từ-model-ra-văn-bản)
8. [Đánh giá kết quả](#8-đánh-giá-kết-quả)
9. [Ví dụ xuyên suốt: "hôm nay trời đẹp"](#9-ví-dụ-xuyên-suốt-hôm-nay-trời-đẹp)

---

## 1. Tổng quan bài toán ASR

**ASR (Automatic Speech Recognition)** là bài toán tự động chuyển đổi giọng nói thành văn bản.

```
Input:  [file âm thanh .wav — tiếng người nói]
Output: "hôm nay trời đẹp"
```

### Vì sao cần các thành phần dưới đây?

| Thành phần | Vai trò | Ví dụ |
|---|---|---|
| **Audio (.wav)** | Tín hiệu âm thanh thô | File ghi âm giọng nói |
| **Transcript (.txt/.tsv)** | Nhãn — "audio này nói gì" | `hôm nay trời đẹp` |
| **Fbank feature** | Biểu diễn âm thanh dạng model có thể học | Ma trận 80 chiều theo thời gian |
| **Tokenizer / BPE** | Chia văn bản thành các đơn vị nhỏ model có thể học | `hôm nay` → `▁hôm ▁nay` |
| **Manifest** | File index: liên kết audio ↔ transcript ↔ duration | Dùng để load dữ liệu hiệu quả |

---

## 2. Pipeline tổng quát

```
Audio (.wav) + Transcript (.txt)
        │
        ▼
   [Tạo Manifest]
   → Ghi lại đường dẫn audio, duration, text vào file index
        │
        ▼
   [Trích xuất Fbank feature]
   → Chuyển waveform → ma trận đặc trưng 80 mel bins
        │
        ▼
   [Train BPE Tokenizer]
   → Học cách chia văn bản thành tokens (ví dụ: vocab 100 tokens)
        │
        ▼
   [Huấn luyện Model]
   → Model học ánh xạ: Fbank feature → token sequence
        │
        ▼
   [Decode]
   → Dùng model đã train để chuyển audio mới → văn bản
        │
        ▼
   [Đánh giá WER]
   → So sánh văn bản model sinh ra với transcript gốc
```

### Bước 1: Chuẩn bị audio và transcript

Cần có cặp `(audio, text)` cho mỗi câu. Ví dụ:

```
audio/train/utt001.wav  →  "hôm nay trời đẹp quá"
audio/train/utt002.wav  →  "tôi đang học nhận dạng tiếng nói"
```

Dữ liệu được chia thành 3 phần:
- **train**: dùng để huấn luyện (~80%)
- **dev**: dùng để theo dõi quá trình train, không train trên đây
- **test**: dùng để đánh giá kết quả cuối (~10%)

### Bước 2: Tạo Manifest

Manifest là file index dạng JSON, chứa thông tin về từng câu:

```json
{"id": "utt001", "recording": {"path": "audio/train/utt001.wav", "duration": 3.2}, 
 "supervision": {"text": "hôm nay trời đẹp quá"}}
```

Ý nghĩa: thay vì load cả dataset vào RAM, model chỉ đọc index rồi load từng batch khi cần.

### Bước 3: Trích xuất Fbank feature

Model không học trực tiếp từ waveform âm thanh (dạng sóng số). Thay vào đó, ta chuyển sang **Fbank (Filter Bank)** — một dạng biểu diễn tần số theo thời gian, gần với cách tai người nghe.

```
Waveform (16000 samples/giây)
    ↓  chia thành frame 25ms, bước 10ms
    ↓  áp dụng FFT + mel filterbank
Ma trận (T × 80)   ← T = số frame theo thời gian, 80 = số mel bins
```

**Ví dụ:** Câu "hôm nay" dài ~0.8 giây → khoảng 80 frame → ma trận (80, 80).

Lý do không dùng waveform thô: waveform có quá nhiều thông tin không liên quan (noise, phase), trong khi Fbank giữ lại thông tin âm học quan trọng và giảm kích thước đáng kể.

### Bước 4: Train BPE Tokenizer

Thay vì học từng chữ cái (quá chi tiết) hoặc từng từ (quá nhiều từ vựng), ta dùng **BPE (Byte Pair Encoding)** để chia văn bản thành các đơn vị con phổ biến.

**Ví dụ với vocab_size=100:**
```
"hôm nay trời đẹp"
     ↓ BPE tokenize
[▁hôm, ▁nay, ▁trời, ▁đẹp]
     ↓ chuyển thành ID
[23, 17, 45, 8]
```

Dấu `▁` đánh dấu đầu từ, giúp phân biệt "đẹp" ở đầu từ với "đẹp" ở giữa từ ghép.

**Ảnh hưởng của vocab_size:**
- Nhỏ (50–100): mỗi token ≈ âm tiết/ký tự, phù hợp corpus nhỏ
- Lớn (500–5000): mỗi token ≈ từ/cụm từ, phù hợp corpus lớn

---

## 3. Kiến trúc model

Project này dùng **Zipformer-Transducer** — một trong những kiến trúc ASR hiện đại nhất.

### Ba thành phần chính

```
Audio feature (T×80)
        │
    [Encoder]  ← "tai của model"
        │
    Encoder output (T'×384)
        │─────────────────────┐
        │                     │
    [Joiner]  ← "bộ kết hợp"  │
        │                     │
    [Decoder]  ← "bộ nhớ ngôn ngữ"
    (dự đoán token tiếp theo dựa trên token đã sinh)
        │
    Output logits → chọn token → văn bản
```

### Encoder làm gì?

Encoder nhận vào chuỗi Fbank feature và trích xuất biểu diễn ngữ nghĩa:

- **Subsampling** (Conv2d): giảm độ phân giải thời gian (từ T frame → T/4 frame), giảm tính toán
- **Zipformer blocks**: kết hợp Attention (nhìn toàn bộ câu) + Convolution (nắm cấu trúc cục bộ)
- Output: mỗi frame được biểu diễn bởi vector 384 chiều mang thông tin âm học

**Hình dung:** Encoder giống như người nghe đang "phân tích" từng đoạn âm thanh và tóm tắt thành thông tin ngữ nghĩa.

### Decoder (Prediction Network) làm gì?

Decoder theo dõi chuỗi token đã được sinh ra, dự đoán token tiếp theo có thể xảy ra:

- Nhận vào: các token đã sinh (ví dụ: `[▁hôm, ▁nay]`)
- Output: xác suất cho token tiếp theo (ví dụ: `▁trời` có xác suất cao)

**Hình dung:** Decoder giống như người biết "sau khi nói hôm nay thì thường nói gì tiếp".

### Joiner làm gì?

Joiner kết hợp thông tin từ Encoder (âm thanh) và Decoder (ngôn ngữ) để ra quyết định cuối:

```
Encoder frame i  +  Decoder state j  →  Joiner  →  P(token | âm thanh + ngữ cảnh)
```

**Hình dung:** Joiner là "người phán xét" — dựa vào cả "âm thanh nghe được" và "ngữ cảnh đã nói", chọn ra token tốt nhất.

### Ví dụ: Audio "xin chào" qua model

```
1. Audio "xin chào" → Fbank (20 frame × 80 dims)
2. Subsampling → 5 frame × 384 dims
3. Zipformer encoder xử lý → 5 frame mang thông tin âm học "xin chào"
4. Decoder bắt đầu với [<blank>], dự đoán token đầu
5. Joiner: frame 1 + [<blank>] → "▁xin" (xác suất cao nhất)
6. Decoder cập nhật: đã có "▁xin", dự đoán tiếp
7. Joiner: frame 3 + [▁xin] → "▁chào"
8. Kết quả: "xin chào"
```

---

## 4. Hàm loss

### Loss là gì và dùng để làm gì?

Loss đo mức độ "sai" của model. Trong training, model cố gắng **tối thiểu hóa loss** bằng cách điều chỉnh hàng triệu tham số của mình.

```
Loss cao → model đang sai nhiều → cần học thêm
Loss thấp → model đang đúng nhiều → đang tiến bộ
```

### Các loss trong project này

Project dùng **Pruned Transducer Loss** làm loss chính, với các loss phụ trợ tùy chọn:

#### 1. Simple Loss (Transducer loss cơ bản)

Tính xác suất của toàn bộ chuỗi alignment có thể có giữa audio và text.

```
P("hôm nay" | audio) = tổng tất cả các cách align "hôm nay" với các frame
```

Ưu điểm: tổng quát. Nhược điểm: chậm vì phải tính tất cả alignment có thể.

#### 2. Pruned Loss (loss chính được dùng)

Phiên bản tối ưu của Simple Loss — chỉ tính loss trên một "vùng" giới hạn của alignment matrix, giảm tính toán đáng kể mà vẫn cho kết quả tốt.

Trong training: Simple Loss dùng ở đầu (giúp model học nhanh), Pruned Loss tăng dần sau khi model đã ổn định.

#### 3. CTC Loss (tùy chọn)

CTC đơn giản hơn Transducer: không có Decoder/Joiner, chỉ dùng Encoder trực tiếp dự đoán token.

```
Audio frames → Encoder → P(token | frame)
```

Khi bật (`--use-ctc`), CTC loss được cộng thêm vào tổng loss với hệ số 0.2, đóng vai trò **regularization** — giúp Encoder học tốt hơn.

#### 4. CR-CTC Loss (tùy chọn)

Consistency-Regularized CTC: chạy CTC với **hai phiên bản SpecAugment khác nhau** của cùng audio, rồi phạt nếu output của hai bản không nhất quán.

```
Audio → augment bản A → CTC output A ┐
Audio → augment bản B → CTC output B ┘→ KL divergence → phạt nếu khác nhau
```

Đặc biệt hữu ích với tiếng Việt có thanh điệu — giúp model "tự consistent" hơn.

#### 5. Attention Decoder Loss (tùy chọn)

Thêm một decoder dạng Transformer, tính cross-entropy loss giữa output của nó với transcript.

### Vì sao loss thấp chưa chắc WER thấp?

Loss là thước đo **toán học** trên training set — model học để giảm con số này.  
WER là thước đo **thực tế** trên test set — ta quan tâm cái này nhất.

Hai lý do loss ≠ WER:
1. **Overfit**: loss trên train thấp nhưng model "học vẹt", không generalize ra test set
2. **Khác đơn vị**: loss tính trên probability, WER tính trên lỗi từ rời rạc. Model có thể giảm loss mà vẫn sinh ra các từ sai.

---

## 5. Quá trình huấn luyện

### Các khái niệm cơ bản

**Epoch:** Một vòng đi qua toàn bộ training data một lần.

```
Epoch 1: model thấy tất cả câu train lần 1 → cập nhật tham số
Epoch 2: model thấy lại tất cả câu train → tiếp tục học
...
Epoch 50: sau 50 vòng, model (hy vọng) đã học tốt
```

**Batch:** Mỗi lần cập nhật tham số, model không xử lý từng câu một mà xử lý một **batch** nhiều câu cùng lúc. Project này dùng `max_duration=50` giây — tức mỗi batch chứa đủ audio để tổng duration ≈ 50 giây.

**Learning rate (LR):** Tốc độ học — mỗi lần cập nhật, thay đổi tham số bao nhiêu.

```
LR lớn → học nhanh nhưng hay "vọt" qua điểm tốt nhất
LR nhỏ → học chậm nhưng ổn định
```

Project dùng **Eden scheduler**: LR bắt đầu thấp, tăng dần trong giai đoạn warmup, sau đó giảm theo formula.

### Data Augmentation

**SpecAugment:** Che ngẫu nhiên một số frame (theo thời gian) hoặc một số mel bins (theo tần số) trong Fbank feature, buộc model học robust hơn.

```
Fbank bình thường:  [████████████████]  (80 mel bins × T frames)
Sau SpecAugment:    [██░░░░████░░████]  (một số bị che → 0)
```

**MUSAN:** Thêm tiếng ồn từ dataset MUSAN (âm nhạc, tiếng ồn nền, tiếng người) vào audio training, giúp model chịu được môi trường ồn ào.

### Vì sao cần validation?

Trong khi train, ta giữ lại một tập dev set không train trên đó. Sau mỗi N batch, chạy model trên dev set và tính loss.

```
Train loss ↓ nhưng Dev loss ↑ → model bắt đầu overfit → cần dừng hoặc thêm regularization
Train loss ↓ và Dev loss ↓ → model đang generalize tốt → tiếp tục train
```

---

## 6. Checkpoint và average nhiều epoch

### Checkpoint là gì?

Checkpoint là **ảnh chụp** toàn bộ trạng thái model tại một thời điểm, lưu vào file `.pt`:

```
exp_dir/
├── epoch-1.pt    ← model sau epoch 1
├── epoch-2.pt    ← model sau epoch 2
...
├── epoch-50.pt   ← model sau epoch 50
└── best-valid-loss.pt  ← model tốt nhất theo dev loss
```

### Vì sao không chỉ lấy epoch cuối?

Quá trình training không phẳng — loss dao động lên xuống:

```
Loss
│╲        
│ ╲   ╱╲  /╲
│  ╲ /  \/  \___
│   ╲
└────────────────── Epoch
       ↑ epoch cuối chưa chắc là tốt nhất
```

### Average nhiều epoch — trực giác

Thay vì lấy 1 checkpoint, ta lấy trung bình tham số của nhiều checkpoint liên tiếp:

```
avg_model = (epoch-46 + epoch-47 + epoch-48 + epoch-49 + epoch-50) / 5
```

**Tại sao điều này giúp ích?**  
Hình dung model như người đang luyện thi: mỗi ngày ôn có ngày tốt ngày xấu. Lấy "kiến thức trung bình" của 5 ngày gần nhất thường ổn định hơn là chỉ lấy 1 ngày bất kỳ.

**Khi nào avg nhiều epoch có lợi?**
- Model đã hội tụ (loss ổn định ở giai đoạn cuối)
- Corpus nhỏ, loss dao động nhiều

**Khi nào avg có thể không giúp?**
- Model chưa hội tụ — avg các checkpoint kém sẽ kéo model xuống
- Avg quá nhiều checkpoint cũ — "pha loãng" kết quả học tốt về sau

---

## 7. Decode: từ model ra văn bản

Sau khi train xong, dùng model để chuyển audio mới → văn bản. Quá trình này gọi là **decode** hoặc **inference**.

### Greedy Search — đơn giản nhất

Tại mỗi bước, chọn **token có xác suất cao nhất**:

```
Frame 1: P(▁hôm)=0.8, P(▁nay)=0.1, P(<blank>)=0.1 → chọn ▁hôm
Frame 2: P(▁nay)=0.7, P(<blank>)=0.2, P(▁này)=0.1 → chọn ▁nay
...
```

| | Greedy |
|---|---|
| Tốc độ | Nhanh nhất |
| Độ chính xác | Thấp nhất trong các method |
| Dùng khi | Smoke test, kiểm tra model có học không |

### Beam Search — giữ nhiều giả thuyết

Thay vì chỉ theo đường "ngon nhất" ngay lập tức, giữ lại **beam_size=4** giả thuyết cùng lúc:

```
Bước 1: Giả thuyết 1: "▁hôm..."  (score: -0.2)
         Giả thuyết 2: "▁hôn..."  (score: -0.5)
         Giả thuyết 3: "▁hom..."  (score: -0.8)
         Giả thuyết 4: "▁hóm..."  (score: -1.1)

Bước 2: Mở rộng cả 4, giữ lại 4 tốt nhất...
```

| | Beam Search |
|---|---|
| Tốc độ | Chậm hơn greedy ~beam_size lần |
| Độ chính xác | Thường tốt hơn greedy |
| Dùng khi | Khi cần kết quả tốt hơn, có đủ thời gian |

### Modified Beam Search — beam search tối ưu cho Transducer

Phiên bản beam search được tối ưu cho kiến trúc Transducer, xử lý theo batch hiệu quả hơn.

| | Modified Beam Search |
|---|---|
| Tốc độ | Nhanh hơn beam_search thường do batch |
| Độ chính xác | Thường tốt hơn greedy, tương đương beam |
| Dùng khi | **Lựa chọn đầu tiên sau greedy trong thực nghiệm** |

### So sánh tổng hợp

| Method | Tốc độ | WER | Khi nào dùng |
|---|---|---|---|
| `greedy_search` | ⚡⚡⚡ Nhanh nhất | Cao nhất | Smoke test, debug |
| `modified_beam_search` | ⚡⚡ Trung bình | Tốt | Thực nghiệm chính |
| `beam_search` | ⚡ Chậm | Tốt | So sánh với modified_beam |

---

## 8. Đánh giá kết quả

### WER (Word Error Rate)

WER đo tỉ lệ lỗi trên **cấp độ từ**:

```
WER = (Insertions + Deletions + Substitutions) / Số từ trong câu gốc × 100%
```

**Ví dụ:**
```
Ground truth:  "hôm nay trời rất đẹp"   (5 từ)
Model output:  "hôm này trời đẹp"       (4 từ)

Lỗi:
  - "nay" → "này": Substitution (S=1)
  - "rất" bị mất:  Deletion (D=1)

WER = (0 + 1 + 1) / 5 × 100% = 40%
```

### CER (Character Error Rate)

CER đo tỉ lệ lỗi trên **cấp độ ký tự**. Với tiếng Việt:

- Nhiều từ đơn âm tiết → WER và CER thường gần nhau
- CER thường thấp hơn WER (vì 1 từ sai chỉ tính 1 lỗi ở WER, nhưng nhiều ký tự đúng ở CER)

### Chú ý với tiếng Việt

Tiếng Việt có dấu thanh điệu (sắc, huyền, hỏi, ngã, nặng) và dấu phụ (ă, â, ê, ô, ơ, ư, đ). Một từ sai dấu = 1 lỗi WER:

```
"trời" vs "troi" → 1 substitution (WER tính 1 lỗi)
```

BPE vocab_size=100 trong project này khá nhỏ — mỗi token gần với âm tiết/ký tự — nên model cần học từng âm tiết, bao gồm cả dấu.

### WER bao nhiêu là tốt?

| WER | Đánh giá |
|---|---|
| < 5% | Rất tốt (mức thương mại) |
| 5–15% | Tốt, dùng được trong nhiều ứng dụng |
| 15–30% | Trung bình, vẫn có giá trị nghiên cứu |
| > 30% | Kém, cần cải thiện data/model/training |

Với corpus nhỏ (< 10h audio), WER 20–40% là bình thường.

---

## 9. Ví dụ xuyên suốt: "hôm nay trời đẹp"

Giả sử có file audio `utt001.wav` người nói câu "hôm nay trời đẹp".

### Bước 1: Chuẩn bị dữ liệu

```
utt001.wav  (2.1 giây, 16kHz, mono)
utt001.txt  → "hôm nay trời đẹp"
```

Được ghi vào `transcripts/train.tsv`:
```
audio_path             text               duration  speaker
audio/train/utt001.wav hôm nay trời đẹp  2.1       trung
```

### Bước 2: Tạo manifest

```json
{"id": "utt001", "duration": 2.1, "channel": 0,
 "source": {"path": "audio/train/utt001.wav"}}
{"id": "utt001", "text": "hôm nay trời đẹp", "speaker": "trung"}
```

### Bước 3: Trích xuất Fbank

```
utt001.wav (2.1s × 16000 = 33600 samples)
    ↓ chia frame 25ms, bước 10ms
    ↓ 209 frames
    ↓ 80 mel bins mỗi frame
→ Ma trận (209, 80) — lưu vào fbank/train/utt001.llc
```

### Bước 4: Tokenize (BPE vocab_size=100)

```
"hôm nay trời đẹp"
    ↓ BPE tokenize
["▁hôm", "▁nay", "▁trời", "▁đẹp"]
    ↓ map to ID
[23, 17, 45, 8]
```

### Bước 5: Đưa vào model (Training)

```
Input: Ma trận (209, 80)
    ↓ Subsampling: (52, 192)   ← giảm 4x theo thời gian
    ↓ Zipformer encoder: (52, 384)
    ↓ Joiner + Decoder
Loss: so sánh output với target [23, 17, 45, 8]
    ↓ Backpropagation → cập nhật 22 triệu tham số
```

### Bước 6: Decode (sau khi train xong)

```
Audio mới: "hôm nay trời đẹp" (người nói khác/lần khác)
    ↓ Fbank → Encoder
    ↓ Modified beam search (beam=4)
Giả thuyết 1: "hôm nay trời đẹp"  score: -2.1
Giả thuyết 2: "hôm này trời đẹp"  score: -3.4
Giả thuyết 3: "hôm nay trời đep"  score: -3.8
Giả thuyết 4: "hom nay trời đẹp"  score: -4.2
    ↓ chọn score tốt nhất
Output: "hôm nay trời đẹp" ✓
```

### Bước 7: Tính WER

```
Ground truth:  "hôm nay trời đẹp"  (4 từ)
Model output:  "hôm nay trời đẹp"
Lỗi: 0
WER = 0/4 × 100% = 0%  ← câu này đúng hoàn toàn
```

Sau khi decode toàn bộ test set (ví dụ 100 câu), lấy trung bình WER để ra WER cuối.

---

*Tài liệu này được tạo dựa trên code trong `vi_asr_corpus/ASR/zipformer/` và `local/`.*  
*Xem `theory_code_practical.md` để hiểu chi tiết từng file code và tham số.*
