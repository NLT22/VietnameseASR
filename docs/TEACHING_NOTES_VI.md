# Hiểu dự án này trong 20 phút (tiếng Việt, dễ hiểu)

Viết cho **chính bạn vài tháng sau**, khi đã quên gần hết. Mục tiêu: đọc xong là
**hiểu**, không cần nền tảng machine learning.

Thuật ngữ tiếng Anh (encoder, decoder, joiner, beam search, WER…) **giữ nguyên**,
nhưng lần đầu xuất hiện sẽ có giải thích bằng ví dụ đời thường.

- File này = **hiểu ý tưởng**.
- `TEACHING_NOTES_VI_chi_tiet.md` = bản chi tiết, đầy đủ config (để tra cứu).
- `RESULTS.md` = các con số.

---

## 1. Dự án này làm gì?

5 người (Dung, Khoi, Quan, Trung, Hieu) mỗi người đọc 200 câu tiếng Việt ngắn →
**1.000 bản ghi âm thật**. Pipeline hiện tại còn thêm voice clone bằng Gwen-TTS
vào tập train, nên bộ `main` có khoảng **16.000 dòng train**. Ta huấn luyện
(train) một mô hình AI để **nghe audio và ghi ra chữ** (đây gọi là ASR — nhận
dạng tiếng nói).

**Mục tiêu thật sự (rất quan trọng phải nhớ):** ta chỉ cần model nhận ra **đúng
những câu đã có trong dataset**. Không cần nó hiểu tiếng Việt nói chung. Giống như
dạy một đứa trẻ **học thuộc lòng** 1.000 câu, chứ không dạy nó đọc thông viết thạo.

Trong tài liệu, cách làm này được gọi là **"học vẹt"**. Nhớ chữ này — nó giải
thích rất nhiều thứ ở phần sau.

Model sau khi train được đóng gói (export) để chạy trên **Jetson Nano** (một máy
tính nhỏ) và trên **trình duyệt web** ở máy này.

---

## 2. Model hoạt động thế nào? (ví dụ: một nhóm 3 người)

Model không phải một khối duy nhất. Nó là **3 mạng nhỏ làm việc cùng nhau**, giống
một nhóm 3 người phiên dịch:

| tên | ví như | việc |
| --- | --- | --- |
| **encoder** | người **nghe** | nghe audio, tóm tắt lại thành "ý" |
| **decoder** | người **nhớ** | nhớ 2 chữ vừa nói ra |
| **joiner** | người **quyết định** | nhìn "ý" + "2 chữ vừa nói" → chọn chữ tiếp theo |

Cứ thế lặp lại: nghe → nhớ → chọn chữ → nghe tiếp… cho tới hết câu.

### encoder — người nghe

Đầu vào của encoder không phải audio thô, mà là **fbank** — hãy hình dung fbank
như "bảng năng lượng âm thanh theo tần số", giống thanh nhạc nhảy nhót trên máy
nghe nhạc. Cứ mỗi 10 mili-giây (ms) audio → một cột số.

encoder xử lý các cột đó và **tóm tắt lại**: khoảng mỗi 40 ms audio → một "ý"
(một vector 512 số). Nó nén ~4 lần.

Một điểm quan trọng: encoder là **causal** — nó **chỉ được nghe quá khứ và một chút
hiện tại, không nghe tương lai**. Nhờ vậy ta có thể chạy **streaming** (chữ hiện ra
ngay khi đang nói, không phải chờ nói xong). Cái giá phải trả: hơi kém chính xác
hơn so với model được nghe cả câu.

### decoder — người nhớ (không phải "bộ giải mã" như tên gọi)

Đừng để chữ "decoder" đánh lừa. Nó **không giải mã gì cả**. Nó là một cái **gợi ý
từ tiếp theo bé xíu**, kiểu như bàn phím điện thoại đoán chữ — nhưng **chỉ nhớ đúng
2 chữ gần nhất** (`context_size=2`). Vì chỉ nhớ 2 chữ nên nó rất nhẹ và có thể
**lưu sẵn kết quả** (cache) để chạy nhanh.

### joiner — người quyết định

Nó cộng "ý từ encoder" với "2 chữ vừa nói từ decoder", rồi cho **điểm cho từng chữ
có thể** trong bộ từ vựng (100 chữ). Chữ điểm cao nhất được chọn.

### Vì sao "beam_search" quan trọng hơn bạn nghĩ

Ở mỗi bước, model có thể:
- Chọn **blank** (nghĩa là "chưa có chữ mới, đi tiếp") → sang đoạn audio kế.
- Hoặc chọn một **chữ thật** → ghi chữ đó ra, và **ở nguyên chỗ cũ** để có thể ghi
  thêm chữ nữa.

Cái vòng "ở nguyên chỗ để ghi tiếp nhiều chữ" là **linh hồn của model này**.

Có nhiều cách để "dò" ra câu cuối cùng:
- **greedy** = luôn chọn chữ điểm cao nhất, không nghĩ lại. Nhanh, hay sai.
- **beam_search** (classic) = giữ vài phương án song song rồi chọn cái tốt nhất. Đây
  là cách **của chúng ta**, và nó cho kết quả tốt nhất.

> **Bài học đắt giá:** có một thư viện tên **sherpa-onnx** (runtime chuẩn của
> ngành). Nhưng nó **không có** classic beam_search — nó chỉ ghi được **tối đa 1
> chữ mỗi bước**, trong khi model của ta cần ghi nhiều chữ. Kết quả: dùng
> sherpa-onnx WER là ~3,3%, dùng beam_search của ta là **0,7%** — chênh gần 5 lần,
> chỉ vì cách dò chữ. Vì thế ta **tự viết** bộ decode (`jetson_beam_decode.py`,
> chỉ dùng numpy + onnxruntime, không cần torch).

*(WER = Word Error Rate = tỉ lệ chữ sai. Càng thấp càng tốt.)*

---

## 3. Vì sao "điểm đẹp nhưng đừng vội mừng" — hiểu cho kỹ

Đây là phần **quan trọng nhất** của cả tài liệu.

### 3.1 Tập test của ta = tập train (đề thi giống hệt bài tập về nhà)

Ba tập `train / dev / test` của pipeline hiện tại thực chất **là cùng một tập
1.000 bản ghi âm thật** — dev và test **giống hệt nhau từng byte**, và cả hai đều
**nằm trong train**.

Nghĩa là: khi ta đo "WER trên test", ta đang hỏi *"model có nhớ đúng cái nó đã học
thuộc không?"* Tất nhiên là nhớ → điểm gần như hoàn hảo. **Con số này không cho
biết model có giỏi thật hay không.** Nó giống như cho học sinh thi lại đúng đề đã
ôn.

Đây chính là "học vẹt" ở mục 1. Nó **cố ý** như vậy (vì mục tiêu của ta là nhận ra
câu có sẵn). Nhưng phải luôn nhớ: **điểm test đẹp ≠ model giỏi.**

### 3.2 Tập "held-out" — và vì sao nó vẫn đẹp dù ta không phủ hết tiếng Việt

Ta có một tập khác gọi là **held-out** (`eval_heldout_speaker.py`): lấy **câu đã
biết** nhưng cho **giọng lạ** (yen_nhi, khanh_toan — 2 giọng tổng hợp chưa từng
train) đọc.

Bạn có thể thắc mắc: *"dataset ta bé tí, đâu phủ hết tiếng Việt, sao held-out vẫn
được WER thấp?"* Câu hỏi rất đúng, và đây là câu trả lời:

**Held-out chỉ đổi GIỌNG, không đổi CÂU.** Cả 25 câu trong held-out đều **đã có
trong train** (chỉ khác người đọc). Model đã "học thuộc" 25 câu này rồi, giờ chỉ
cần nghe ra chúng trong một giọng mới — dễ hơn nhiều so với nghe một câu **hoàn
toàn mới**.

→ Nếu cho model một câu **chưa từng thấy** (chữ sắp xếp theo thứ tự mới), nó sẽ sai
**rất nặng**. Model **không biết** tiếng Việt tổng quát; nó chỉ thuộc lòng một bộ
câu cố định. Điểm đẹp chỉ nói lên "nó thuộc bài", không phải "nó biết đọc".

### 3.3 Cỡ tập đo quá nhỏ → dễ bị "may rủi"

Tập held-out chỉ có **578 chữ**. Chênh 3–4 chữ đã là 0,5–0,7 điểm WER. Nên khi hai
model chênh nhau 1 điểm, rất có thể chỉ là **may rủi**, không phải model nào giỏi
hơn.

> **Quy tắc vàng:** trước khi tin "model A tốt hơn model B", hãy **bootstrap**
> (lấy mẫu lại nhiều lần để xem khoảng tin cậy). Trên tập 578 chữ, chênh dưới ~2
> điểm coi như **nhiễu**, đừng kết luận.
>
> *(Đang mở rộng tập này lên ~5000 chữ để đo cho chắc.)*

---

## 4. Những cú vấp lớn (kể lại để không dẫm lại)

### Bug "lệch một dòng" (nghiêm trọng nhất, ẩn suốt 2 tháng)

Máy ghi âm Windows đặt tên file hơi lạ: bản đầu là `Recording.wav`, các bản sau là
`Recording (2).wav`, `(3)`… Khi sắp xếp theo thứ tự, file **không có số** bị đẩy
**xuống cuối** thay vì lên đầu.

Hậu quả: audio bị **ghép lệch với câu chữ** — audio số 1 bị gán nhãn câu số 2, v.v.
Toàn bộ của **Trung và Dung** bị sai nhãn (330/730 bản ghi). Quan và Khoi may mắn
không dính (tên file của họ không có kiểu "không số").

**Vì sao ẩn được 2 tháng?** Vì tập test = tập train (mục 3.1). Model học thuộc luôn
cả nhãn sai một cách nhất quán, nên điểm test vẫn ~0,5%, chẳng ai nghi ngờ. Chỉ khi
**người thật nói vào mic** và ra chữ sai thì mới lộ.

**Cách phát hiện (nhớ làm sau mỗi lần nạp data):** câu dài thì đọc lâu hơn. Nên
kiểm tra tương quan giữa **độ dài audio** và **số chữ** cho từng người. Nếu ghép
đúng, tương quan phải **dương rõ ràng**. Nếu bị lệch, nó tụt về 0. Ba dòng code này
lẽ ra đã tiết kiệm 2 tháng.

### Bug learning rate (cú lớn thứ hai — chỉ một con số sai)

`learning rate` (tốc độ học) là "model học nhanh hay chậm mỗi bước". `run.sh` bị đặt
cứng thành **0,01**, trong khi mặc định của icefall là **0,045**.

Ta tưởng 0,01 là ổn. Hoá ra nó **quá chậm**: đổi lên 0,045 giúp WER held-out từ
**10,9% xuống 2,1%** — tốt hơn ~4,5 lần, hơn tất cả các cải tiến về dữ liệu cộng
lại. **Một dòng config sai âm thầm làm hỏng mọi model.** Giờ đã sửa về 0,045.

Bài học kèm theo: **60 epoch là thừa.** Với learning rate đúng, model đạt đỉnh ở
khoảng **epoch 40**, rồi sau đó **học quá đà** (overfit) và tệ đi. (epoch = một
vòng học hết toàn bộ dữ liệu.)

### Cái bẫy fbank (im lặng nhưng chết người)

Khi model chạy thật, audio phải được biến thành fbank **y hệt** lúc train. Đặc biệt:
sóng âm phải nằm trong khoảng **[-1, 1]**. Nếu để nguyên số nguyên kiểu
[-32768, 32767], **mọi con số bị lệch đi ~20,8** và model nghe thấy thứ nó chưa
từng thấy → **ra rác, không báo lỗi gì cả**. Nên nhớ: `chia cho 32768.0`.

### Clone giọng bị clip (Trung)

Bản ghi bị **clip** nghĩa là thu quá to, đỉnh sóng bị cắt phẳng — không sửa được
bằng cách giảm âm lượng vì phần đỉnh đã mất hẳn. Bài học thực tế: **thu vừa phải,
đừng để đỉnh chạm trần**, vì lỗi này lan sang cả dữ liệu clone và làm model học
từ tín hiệu méo.

---

## 5. Vài con số config quan trọng nhất

(Bảng đầy đủ ở `TEACHING_NOTES_VI_chi_tiet.md` mục 4. Đây chỉ là những cái then
chốt.)

| tham số | giá trị đúng | ghi chú |
| --- | --- | --- |
| `--base_lr` | **0,045** | quan trọng nhất; đừng để 0,01 |
| số epoch | **~40** | 60 là thừa và bắt đầu overfit |
| `--vocab_size` | 100 | đủ cho tiếng Việt đơn âm tiết |
| `--causal` | 1 | bắt buộc, để streaming |
| `--avg` | 10 | trung bình 10 checkpoint cuối |
| `--max_duration` | 700 | **tổng giây audio mỗi batch**, không phải số câu. (Các run cũ tháng 7 dùng 450; đây chỉ là mức tận dụng GPU, không đổi chất lượng. LR 0,045 đã kiểm ở 700.) |
| decode | `beam_search` | KHÔNG dùng greedy/sherpa |
| int8 | mặc định | nhanh hơn, nhỏ hơn, sai số y hệt fp32 |
| fp16 | **không dùng được** | graph lỗi khi load |

---

## 6. Khi gặp lỗi thì làm gì (sổ tay nhanh)

**Model ra chữ vô nghĩa / chuỗi rỗng:**
1. Audio có ở [-1,1] chưa? 16 kHz mono chưa? fbank có đúng cấu hình không?
2. Đã `reset()` bộ nhớ encoder giữa 2 câu chưa? (quên → model "nghe" cả câu trước)
3. Clip có dài hơn ~15 giây không? Model chỉ quen câu 4–5 s → câu dài ra rỗng.

**Điểm test đẹp nhưng nói thật thì trượt:** bạn đang cầm chỉ số "học vẹt". Chạy
kiểm tra tương quan độ-dài/số-chữ (xem bug lệch dòng ở mục 4).

**Ra toàn khoảng trống / mất chữ:** chắc đang dùng greedy hoặc sherpa. Đổi sang
classic `beam_search`.

**Hai kết quả chênh nhau ít:** bootstrap trước khi tin. Dưới ~2 điểm là nhiễu.

**Cảnh báo về `pkill -f` / `pgrep -f`:** lệnh này **tự khớp với chính nó** → có thể
tự giết mình hoặc chạy vô tận. Đã cắn 3 lần. Hãy grep **file log**, hoặc dùng
`serve[r].py` (thêm ngoặc vuông để không tự khớp).

**Bẫy khác:**
- `/tmp` bị xoá khi khởi động lại → giữ log train ở đĩa dự án.
- Mất điện làm checkpoint đang ghi thành 0 byte → xoá nó, chạy lại với
  `--start_epoch N` (tự nạp `epoch-(N-1).pt`).
- Export hiện nằm trong `run.sh` stage 16; script thật bên dưới là
  `local/export_for_jetson.sh`.

---

## 7. Giới hạn cần biết

- **Chỉ nhận được câu ngắn** (4–5 s). Câu dài >15 s → ra rỗng.
- **Chỉ thuộc các câu có sẵn.** Câu mới hoàn toàn → sai nặng. Đây là chủ ý (mục 1).
- **Giọng nữ khó hơn.** Thêm giọng nữ vào train giúp nhiều nhất.
- **Mọi thứ train trước 2026-07-09 dùng nhãn sai** (bug lệch dòng) → đã bỏ đi.

---

## 8. Muốn tìm hiểu sâu hơn?

- `TEACHING_NOTES.md` (tiếng Anh) — đầy đủ, tới tận shape của tensor.
- `TEACHING_NOTES_VI_chi_tiet.md` — bản tiếng Việt chi tiết + toàn bộ bảng config.
- `RESULTS.md` — tất cả con số và thí nghiệm.
