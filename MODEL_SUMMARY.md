# Giải thích model Zipformer/RNN-T và kết quả hiện tại

Tài liệu này tập trung vào 3 phần:

| Phần | Nội dung |
|---|---|
| Kiến trúc model | Model đang dùng là gì, các khối trong model làm nhiệm vụ gì. |
| Phương pháp decode | `greedy_search`, `modified_beam_search`, `beam_search` khác nhau như thế nào. |
| Kết quả hiện tại | Tổng hợp WER đã đạt được và nhận xét nhanh. |

## 1. Tổng quan model

Recipe này dùng mô hình **Transducer**, thường gọi là **RNN-T**. Đây là kiến trúc rất phổ biến cho ASR streaming hoặc near-streaming.

Model gồm 3 thành phần chính:

```text
                 audio
                   |
                 fbank
                   |
            Zipformer encoder
                   |
                   v
previous tokens -> predictor/decoder -> joiner -> next token / blank
```

| Thành phần | Nhiệm vụ |
|---|---|
| **Zipformer encoder** | Đọc chuỗi đặc trưng âm thanh, ví dụ fbank, rồi biến nó thành representation giàu thông tin hơn. |
| **Stateless predictor / decoder** | Nhìn vào các token đã sinh trước đó để cung cấp ngữ cảnh text. |
| **Joiner** | Kết hợp thông tin âm thanh từ encoder và ngữ cảnh token từ decoder để dự đoán token tiếp theo. |

Khác với CTC, Transducer không chỉ nhìn audio độc lập để dự đoán từng frame. Nó vừa nhìn audio, vừa nhìn lịch sử token đã sinh. Vì vậy model có khả năng học quan hệ giữa âm thanh và chuỗi ký tự/từ tốt hơn, nhưng decode cũng phức tạp hơn.

## 2. Zipformer encoder

Phần encoder của model là **Zipformer**. Đây là biến thể Transformer được thiết kế riêng cho ASR trong icefall/k2.

Input audio sau khi trích xuất fbank thường có frame-rate khoảng **100 Hz**, tức khoảng 100 frame mỗi giây. Nếu đưa nguyên chuỗi dài này qua Transformer lớn thì rất tốn tính toán. Zipformer giải quyết bằng cách xử lý chuỗi ở nhiều độ phân giải thời gian khác nhau.

Luồng tổng quát:

```text
100 Hz fbank
  -> Conv2d subsampling
  -> 50 Hz
  -> Zipformer stacks ở 50 Hz, 25 Hz, 12.5 Hz, 6.25 Hz
  -> upsample/bypass dần
  -> output khoảng 25 Hz
```

Ý tưởng chính là:

| Ý tưởng | Giải thích |
|---|---|
| **Downsample** | Giảm số frame để model nhìn được ngữ cảnh dài hơn với chi phí thấp hơn. |
| **Upsample** | Đưa representation quay lại frame-rate cao hơn để giữ thông tin thời gian chi tiết. |
| **Bypass connection** | Truyền thông tin qua các nhánh tắt, giúp train ổn định hơn. |
| **Multi-scale encoder** | Các tầng khác nhau nhìn audio ở các độ phân giải khác nhau. |

Với cấu hình `base`, số block theo từng stack là:

| Stack | Frame-rate | Số Zipformer block |
|---|---:|---:|
| Stack 1 | 50 Hz | 2 |
| Stack 2 | 25 Hz | 2 |
| Stack 3 | 12.5 Hz | 3 |
| Stack 4 | 6.25 Hz | 4 |
| Stack 5 | 12.5 Hz | 3 |
| Stack 6 | 25 Hz | 2 |

Điểm quan trọng: phần giữa chạy ở frame-rate thấp nhất, nên mỗi block có thể học ngữ cảnh dài hơn mà không làm chi phí attention tăng quá mạnh.

## 3. Bên trong một Zipformer block

Một Zipformer block có thể hiểu như một block Transformer được bổ sung các module riêng cho ASR.

Các thành phần chính:

| Module | Vai trò |
|---|---|
| **Feed-forward** | Biến đổi representation tại từng frame. |
| **Self-attention** | Cho phép mỗi frame nhìn sang các frame khác để học ngữ cảnh dài. |
| **Non-linear attention** | Biến thể attention trong Zipformer, giúp attention linh hoạt hơn attention tuyến tính đơn giản. |
| **Convolution** | Bắt pattern cục bộ, ví dụ formant, chuyển âm, nhiễu ngắn. |
| **Bypass** | Tương tự residual connection, giúp thông tin và gradient đi qua model dễ hơn. |
| **Balancer / BasicNorm** | Các module ổn định training trong icefall. |

Vì ASR vừa cần ngữ cảnh dài vừa cần chi tiết cục bộ, Zipformer kết hợp cả attention và convolution. Attention giỏi học phụ thuộc dài, còn convolution giỏi học pattern gần nhau theo thời gian.

## 4. Predictor/decoder và joiner

Trong Transducer, encoder chỉ xử lý audio. Để sinh chữ/token, model cần thêm ngữ cảnh từ các token đã sinh trước đó.

### Stateless predictor

Predictor trong recipe này là **stateless decoder**, không phải LSTM decoder truyền thống. Nó thường dùng embedding của vài token trước đó.

Với `context_size=2`, decoder nhìn 2 token gần nhất:

```text
[token_{u-2}, token_{u-1}] -> decoder embedding -> decoder representation
```

Ưu điểm là nhẹ, nhanh, dễ train hơn decoder có recurrent state.

### Joiner

Joiner nhận 2 luồng:

| Input | Từ đâu ra |
|---|---|
| Encoder representation | Thông tin audio tại một frame/time step. |
| Decoder representation | Thông tin text context tại một output step. |

Sau đó joiner chiếu 2 representation này về cùng dimension, cộng/kết hợp chúng, rồi sinh phân phối xác suất trên vocab BPE cộng thêm blank.

```text
encoder_proj(audio_state) + decoder_proj(text_state)
  -> non-linearity
  -> output_linear
  -> token probabilities
```

Nếu model sinh **blank**, decode sẽ đi tiếp trên audio frame tiếp theo. Nếu model sinh **token**, decode giữ nguyên audio frame và cập nhật text context.

## 5. Các cấu hình model đã dùng

Hiện tại recipe có 3 preset chính trong `run.sh`: `base`, `small`, `tiny`.

| Preset | Ý nghĩa | Params |
|---|---|---:|
| `base` | Config gốc từ recipe Zipformer hiện tại. | 64,728,611 |
| `small` | Bản custom thu nhỏ để hợp hơn với tập dữ liệu nhỏ. | 17,733,695 |
| `tiny` | Bản custom nhỏ hơn nữa để thử nhanh. | 8,151,940 |

Chi tiết config:

| Preset | `num_encoder_layers` | `encoder_dim` | `feedforward_dim` | `decoder_dim` | `joiner_dim` |
|---|---|---|---|---:|---:|
| `base` | `2,2,3,4,3,2` | `192,256,384,512,384,256` | `512,768,1024,1536,1024,768` | 512 | 512 |
| `small` | `2,2,2,2,2,2` | `128,192,256,384,256,192` | `256,384,512,768,512,384` | 256 | 256 |
| `tiny` | `1,1,2,2,2,1` | `96,128,192,256,192,128` | `192,256,384,512,384,256` | 192 | 192 |

Lưu ý: `small` và `tiny` ở đây là **preset custom trong repo này**, không phải exact official config `Zipformer-small` từ icefall. Chúng được scale xuống từ `base` bằng cách giảm số block, encoder dimension, feed-forward dimension, decoder dimension và joiner dimension.

Số tham số được tính bằng:

```python
num_param = sum(p.numel() for p in model.parameters())
```

## 6. Các phương pháp decode

Decode là quá trình biến output xác suất của model thành chuỗi token/text cuối cùng.

### 6.1. Greedy search

`greedy_search` là cách decode đơn giản nhất.

Ở mỗi bước, model chọn token có xác suất cao nhất:

```text
current state -> chọn token tốt nhất -> đi tiếp
```

Ưu điểm:

| Ưu điểm | Giải thích |
|---|---|
| Nhanh | Chỉ giữ 1 hypothesis. |
| Dễ debug | Ít biến số, phù hợp kiểm tra nhanh model có học không. |
| Ít tốn RAM/GPU | Không mở rộng nhiều nhánh decode. |

Nhược điểm:

| Nhược điểm | Giải thích |
|---|---|
| Dễ mắc lỗi cục bộ | Nếu chọn sai token sớm, các bước sau bị kéo theo. |
| WER thường cao hơn beam search | Vì không thử các phương án thay thế. |

### 6.2. Modified beam search

`modified_beam_search` giữ nhiều hypothesis cùng lúc thay vì chỉ giữ một.

Ví dụ với `beam_size=4`, decode sẽ giữ một nhóm phương án tốt nhất trong quá trình sinh token. Nếu một nhánh tạm thời chưa tốt nhất nhưng về sau có thể tốt hơn, beam search vẫn có cơ hội giữ lại.

Ưu điểm:

| Ưu điểm | Giải thích |
|---|---|
| WER thường tốt hơn greedy | Vì không quyết định quá sớm. |
| Tối ưu cho Transducer trong icefall | Đây thường là lựa chọn tốt để báo cáo kết quả. |
| Tốc độ vẫn chấp nhận được | Nhẹ hơn beam search đầy đủ trong nhiều trường hợp. |

Nhược điểm:

| Nhược điểm | Giải thích |
|---|---|
| Chậm hơn greedy | Phải giữ nhiều hypothesis. |
| Cần chọn beam size | Beam lớn hơn chưa chắc luôn tốt hơn, nhưng chắc chắn tốn hơn. |

### 6.3. Beam search

`beam_search` là beam search đầy đủ hơn. Nó cũng giữ nhiều hypothesis, nhưng cách mở rộng/trộn hypothesis có thể khác `modified_beam_search`.

Ưu điểm:

| Ưu điểm | Giải thích |
|---|---|
| Có thể cho WER rất thấp | Vì tìm kiếm rộng hơn greedy. |
| Hữu ích để so sánh decode strategy | Cho biết model có tiềm năng tốt hơn khi search kỹ hơn không. |

Nhược điểm:

| Nhược điểm | Giải thích |
|---|---|
| Chậm hơn | Tốn compute hơn greedy và thường hơn modified beam. |
| Kết quả bất thường cần kiểm chứng | Nếu WER quá thấp như `0.00`, cần kiểm tra test set/leakage/log decode. |

Với kết quả hiện tại, `beam_search` đang có vài WER `0.00`. Đây là tín hiệu phải cẩn thận: có thể do test set nhỏ/quá lặp, hoặc có vấn đề chia train/test, hoặc decode đang chạy trên tập quá dễ. Không nên xem `0.00` là kết luận cuối cùng trước khi audit lại dữ liệu.

## 7. Cách đọc chuỗi thí nghiệm hiện tại

Thay vì chỉ nhìn một bảng WER rời rạc, có thể đọc quá trình cải thiện theo 4 bước:

| Bước | Câu hỏi | Kết luận hiện tại |
|---|---|---|
| 1 | Với model `base`, train có học không? | Có. `base raw epoch30 greedy` đạt WER 65.35, tốt hơn epoch 1 và epoch 5. |
| 2 | Giảm kích thước model có giúp không? | Có tín hiệu, nhưng không tuyệt đối. Raw epoch30 thì `base/tiny` tốt hơn `small`; NR epoch30 thì `small` tốt nhất theo greedy. |
| 3 | Train `small` lâu hơn có giúp không? | Có. Với raw, `small epoch50 greedy` đạt 55.30, tốt hơn `small epoch30 greedy` 77.85 và tốt hơn `base raw epoch30 greedy` 65.35. |
| 4 | Đổi decode method có cải thiện không? | Có rất mạnh. `base raw epoch30 beam_search` đạt 23.51, `small_nr epoch30 modified_beam_search` đạt 31.11. |

Điểm quan trọng: câu chuyện hiện tại không nên viết kiểu “giảm model là thắng ngay”. Kết quả bổ sung cho thấy ở **raw epoch30**, `small` chưa hội tụ tốt. Nhưng khi train `small` lâu hơn hoặc dùng noisereduce, nó trở thành hướng tốt nhất.

Vì vậy cách kể hợp lý nhất là:

```text
base raw epoch30 là baseline tốt ban đầu
  -> thử giảm model size
  -> raw epoch30: base/tiny tốt hơn small, tiny nhỉnh hơn base rất nhẹ khi dùng modified beam
  -> raw epoch30 beam_search: base tốt nhất với WER 23.51
  -> noisereduce epoch30: small tốt nhất giữa base/small/tiny theo greedy
  -> noisereduce epoch30 modified beam: small vẫn tốt nhất với WER 31.11
  -> train small raw lên 50 epoch: greedy cải thiện xuống 55.30
  -> decode small bằng beam methods: modified beam xuống 43.34 trên raw, 31.11 trên NR
  -> beam_search cho số tốt nhất nhưng cần audit nếu quá thấp
```

## 8. So sánh model size ở epoch 30

### 8.1. Raw data, greedy search

| Model | Data | Epoch | Decode | Params | WER | Nhận xét |
|---|---|---:|---|---:|---:|---|
| `base` | raw | 30 | `greedy_search` | 64,728,611 | 65.35 | Baseline tốt nhất ở raw epoch30. |
| `tiny` | raw | 30 | `greedy_search` | 8,151,940 | 69.29 | Nhỏ hơn nhiều, WER cao hơn base một chút. |
| `small` | raw | 30 | `greedy_search` | 17,733,695 | 77.85 | Chưa hội tụ tốt ở epoch30 trên raw. |

Kết luận cho raw epoch30: `base` đang tốt nhất. Nếu chỉ nhìn raw tại epoch30 thì chưa nên nói `small` thắng.

### 8.2. Raw data, modified beam search

| Model | Data | Epoch | Decode | Params | WER | Nhận xét |
|---|---|---:|---|---:|---:|---|
| `tiny` | raw | 30 | `modified_beam_search`, beam 4 | 8,151,940 | 53.40 | Tốt nhất ở raw epoch30 theo modified beam, nhưng chỉ nhỉnh hơn base rất nhẹ. |
| `base` | raw | 30 | `modified_beam_search`, beam 4 | 64,728,611 | 53.53 | Gần như hòa với tiny. |
| `small` | raw | 30 | `modified_beam_search`, beam 4 | 17,733,695 | 82.07 | Chưa hội tụ tốt ở epoch30. |

Kết luận cho raw epoch30 khi decode bằng modified beam: `tiny` nhỉnh nhất, nhưng khoảng cách với `base` chỉ 0.13 WER nên chưa đủ mạnh để kết luận tiny “thắng hẳn”. Điều chắc hơn là `small raw epoch30` chưa phải checkpoint tốt.

### 8.3. Raw small epoch30, thêm beam search

| Model | Data | Epoch | Decode | Beam | WER | Nhận xét |
|---|---|---:|---|---:|---:|---|
| `small` | raw | 30 | `greedy_search` | - | 77.85 | Greedy yếu. |
| `small` | raw | 30 | `modified_beam_search` | 4 | 82.07 | Không cải thiện, thậm chí xấu hơn greedy. |
| `small` | raw | 30 | `beam_search` | 4 | 61.68 | Beam search cải thiện rõ, nhưng vẫn chưa bằng base/tiny modified beam. |

Kết luận: `small raw epoch30` không phải điểm tốt để chọn model. Việc `small` tốt lên ở epoch50 là do train lâu hơn, không phải do ngay từ epoch30 nó đã tốt.

### 8.4. Raw data, beam search

| Model | Data | Epoch | Decode | Params | WER | Nhận xét |
|---|---|---:|---|---:|---:|---|
| `base` | raw | 30 | `beam_search`, beam 4 | 64,728,611 | 23.51 | Tốt nhất trong nhóm raw epoch30 khi dùng beam search. |
| `tiny` | raw | 30 | `beam_search`, beam 4 | 8,151,940 | 41.71 | Cải thiện mạnh so với modified beam, nhưng vẫn kém base. |
| `small` | raw | 30 | `beam_search`, beam 4 | 17,733,695 | 61.68 | Beam search giúp hơn greedy/modified beam, nhưng vẫn chưa tốt. |

Kết luận cho raw epoch30 khi dùng beam search: `base` đang là lựa chọn mạnh nhất. Đây là một điểm rất quan trọng vì nó cho thấy model lớn vẫn có lợi nếu decode search đủ mạnh.

### 8.5. Noisereduce data, greedy search

| Model | Data | Epoch | Decode | Params | WER | Nhận xét |
|---|---|---:|---|---:|---:|---|
| `small` | noisereduce | 30 | `greedy_search` | 17,733,695 | 55.57 | Tốt nhất trong nhóm NR epoch30. |
| `base` | noisereduce | 30 | `greedy_search` | 64,728,611 | 62.64 | Tốt hơn raw base, nhưng vẫn kém small NR. |
| `tiny` | noisereduce | 30 | `greedy_search` | 8,151,940 | 64.54 | Nhẹ nhất nhưng thiếu capacity hơn small. |

Kết luận cho noisereduce epoch30: `small` là lựa chọn hợp lý nhất giữa `base/small/tiny`. Đây là điểm mạnh để chọn `small` làm hướng thí nghiệm tiếp theo.

### 8.6. Noisereduce data, modified beam search

| Model | Data | Epoch | Decode | Params | WER | Nhận xét |
|---|---|---:|---|---:|---:|---|
| `small` | noisereduce | 30 | `modified_beam_search`, beam 4 | 17,733,695 | 31.11 | Tốt nhất trong nhóm NR epoch30. |
| `tiny` | noisereduce | 30 | `modified_beam_search`, beam 4 | 8,151,940 | 38.18 | Nhẹ hơn small, WER cao hơn khoảng 7 điểm. |
| `base` | noisereduce | 30 | `modified_beam_search`, beam 4 | 64,728,611 | 49.05 | Kém small/tiny trong nhánh NR. |

Kết luận cho noisereduce epoch30 khi dùng modified beam: `small` là lựa chọn tốt nhất, và kết luận này giờ chặt hơn vì đã có đủ `base/small/tiny`.

### 8.7. Noisereduce data, beam search

| Model | Data | Epoch | Decode | Params | WER | Nhận xét |
|---|---|---:|---|---:|---:|---|
| `small` | noisereduce | 30 | `beam_search`, beam 4 | 17,733,695 | 5.98 | Tốt nhất trong nhóm NR epoch30, nhưng thấp bất thường nên cần audit. |
| `tiny` | noisereduce | 30 | `beam_search`, beam 4 | 8,151,940 | 20.52 | Tốt hơn base NR và raw tiny beam. |
| `base` | noisereduce | 30 | `beam_search`, beam 4 | 64,728,611 | 25.27 | Gần raw base beam, nhưng kém tiny/small NR. |

Kết luận cho noisereduce epoch30 khi dùng beam search: `small_nr` vẫn tốt nhất, `tiny_nr` cũng rất đáng chú ý vì ít tham số nhưng đạt WER 20.52. Tuy nhiên các WER beam search quá thấp cần kiểm tra leakage/test-set trước khi coi là kết luận chính thức.

## 9. Train small lên 50 epoch

Sau khi chọn hướng `small`, có hai nhánh quan trọng:

| Model | Data | Epoch | Decode | WER | So với mốc trước |
|---|---|---:|---|---:|---|
| `base` | raw | 30 | `greedy_search` | 65.35 | Baseline raw ban đầu. |
| `small` | raw | 30 | `greedy_search` | 77.85 | Chưa tốt ở epoch30. |
| `small` | raw | 50 | `greedy_search` | 55.30 | Tốt hơn base raw epoch30. |
| `small` | noisereduce | 30 | `greedy_search` | 55.57 | Gần ngang raw small epoch50. |
| `small` | noisereduce | 50 | `greedy_search` | 55.57 | Không cải thiện thêm theo greedy. |

Kết luận: `small` cần train lâu hơn trên raw để vượt baseline. Với noisereduce, `small` đã đạt mức tốt từ epoch30, nhưng train lên epoch50 không cải thiện theo greedy.

Điều này gợi ý:

| Gợi ý | Lý do |
|---|---|
| Giữ `small` làm model chính | Nó cho kết quả tốt hơn khi train đủ lâu hoặc dùng NR. |
| Không mặc định lấy checkpoint cuối | `small_nr epoch30` có thể tốt hơn epoch50 với modified beam. |
| Nên so sánh nhiều checkpoint | Với data nhỏ, epoch tốt nhất không nhất thiết là epoch cuối. |

## 10. Thời gian huấn luyện

Thời gian dưới đây lấy từ timestamp trong các file `log-train-*`, tính từ dòng `Training started` đến `Done!`.

| Exp | Data | Model | Epoch range | Device log | Params | Thời gian train |
|---|---|---|---|---|---:|---:|
| `exp_bpe100` | raw | base | 1 -> 30 | `cuda:0` | 64,728,611 | 20 phút 11 giây |
| `exp_bpe100_small` | raw | small | 1 -> 50 | `cuda:0` | 17,733,695 | 20 phút 00 giây |
| `exp_bpe100_tiny` | raw | tiny | 1 -> 30 | `cuda:0` | 8,151,940 | 9 phút 24 giây |
| `exp_bpe100_nr` | noisereduce | base | 1 -> 30 | `cuda:0` | 64,728,611 | 18 phút 53 giây |
| `exp_bpe100_small_nr` | noisereduce | small | 1 -> 30 | `cuda:0` | 17,733,695 | 11 phút 45 giây |
| `exp_bpe100_small_nr` | noisereduce | small | 31 -> 50 | `cuda:0` | 17,733,695 | 7 phút 45 giây |
| `exp_bpe100_small_nr` | noisereduce | small | 1 -> 50 total | `cuda:0` | 17,733,695 | khoảng 19 phút 30 giây |
| `exp_bpe100_tiny_nr` | noisereduce | tiny | 1 -> 30 | `cuda:0` | 8,151,940 | 9 phút 26 giây |

Nhận xét về thời gian:

| Nhận xét | Ý nghĩa |
|---|---|
| `tiny` nhanh nhất. | Khoảng 9.5 phút cho 30 epoch, nhưng WER chưa tốt nhất. |
| `small` có trade-off tốt. | Ít tham số hơn base nhiều, thời gian train 50 epoch vẫn khoảng 20 phút trên data hiện tại. |
| `base` không quá chậm trên data nhỏ, nhưng nhiều tham số hơn. | 64.7M params, WER không tốt hơn small sau khi small train đủ lâu. |
| `small_nr` epoch30 là điểm rất đáng chú ý. | Train khoảng 11 phút 45 giây, modified beam đạt WER 31.11. |
| `base raw` vẫn rất mạnh nếu dùng beam search. | Train 30 epoch khoảng 20 phút 11 giây, beam search đạt WER 23.51. |

## 11. So sánh decode methods trên small

### 11.1. Small raw epoch50

| Decode method | Beam | WER | Nhận xét |
|---|---:|---:|---|
| `greedy_search` | - | 55.30 | Nhanh, baseline decode. |
| `modified_beam_search` | 4 | 43.34 | Cải thiện rõ so với greedy. |
| `beam_search` | 4 | 0.00 | Tốt nhất về số, nhưng quá bất thường, cần audit. |

Nếu bỏ qua rủi ro `0.00`, kết luận chắc nhất là `modified_beam_search` giúp giảm WER từ 55.30 xuống 43.34.

### 11.2. Small noisereduce epoch30

| Decode method | Beam | WER | Nhận xét |
|---|---:|---:|---|
| `greedy_search` | - | 55.57 | Gần bằng raw small epoch50 greedy. |
| `modified_beam_search` | 4 | 31.11 | Kết quả đáng tin nhất hiện tại nếu chưa chấp nhận beam 0.00. |
| `beam_search` | 4 | 5.98 | Rất tốt, nhưng vẫn nên kiểm chứng vì thấp bất thường. |

### 11.3. Small noisereduce epoch50

| Decode method | Beam | WER | Nhận xét |
|---|---:|---:|---|
| `greedy_search` | - | 55.57 | Không đổi so với epoch30. |
| `modified_beam_search` | 4 | 37.50 | Xấu hơn epoch30. |
| `beam_search` | 4 | 0.00 | Tốt nhất về số, nhưng cần audit. |

Kết luận decode: Beam-based decoding tốt hơn greedy rất nhiều. Tuy nhiên, vì `beam_search` có kết quả `0.00`, phương pháp nên dùng để báo cáo tạm thời là `modified_beam_search`, còn `beam_search` nên đặt vào nhóm “best observed but needs verification”.

## 12. Tổng hợp WER hiện tại

WER càng thấp càng tốt. Bảng này được rút gọn để chỉ giữ các mốc quan trọng; các số cũ/ít liên quan hơn nằm trong các file `wer-summary-*` tương ứng trong từng `exp_dir`.

### 12.1. Các mốc tốt nhất

| Nhóm | Data | Model | Epoch | Decode | WER | Ghi chú |
|---|---|---|---:|---|---:|---|
| Best observed | raw / NR | small | 50 | `beam_search` | 0.00 | Cần audit, chưa nên báo cáo chính. |
| Best non-zero beam | noisereduce | small | 30 | `beam_search` | 5.98 | Rất mạnh nhưng vẫn cần kiểm tra leakage. |
| Best raw beam | raw | base | 30 | `beam_search` | 23.51 | Mốc raw tốt nhất không tính WER 0.00. |
| Best modified beam | noisereduce | small | 30 | `modified_beam_search` | 31.11 | Mốc nên dùng để báo cáo thận trọng. |
| Best raw modified beam | raw | small | 50 | `modified_beam_search` | 43.34 | Raw tốt nhất nếu không dùng full beam search. |
| Best greedy | raw | small | 50 | `greedy_search` | 55.30 | Mốc greedy tốt nhất. |

### 12.2. So sánh model size ở epoch30

| Data | Decode | Base | Small | Tiny | Model tốt nhất |
|---|---|---:|---:|---:|---|
| raw | `greedy_search` | 65.35 | 77.85 | 69.29 | base |
| raw | `modified_beam_search` | 53.53 | 82.07 | 53.40 | tiny, gần hòa base |
| raw | `beam_search` | 23.51 | 61.68 | 41.71 | base |
| noisereduce | `greedy_search` | 62.64 | 55.57 | 64.54 | small |
| noisereduce | `modified_beam_search` | 49.05 | 31.11 | 38.18 | small |
| noisereduce | `beam_search` | 25.27 | 5.98 | 20.52 | small |

### 12.3. Small khi train lâu hơn

| Data | Epoch | `greedy_search` | `modified_beam_search` | `beam_search` | Nhận xét |
|---|---:|---:|---:|---:|---|
| raw | 30 | 77.85 | 82.07 | 61.68 | Chưa hội tụ tốt. |
| raw | 50 | 55.30 | 43.34 | 0.00 | Train lâu hơn cải thiện rõ, beam 0.00 cần audit. |
| noisereduce | 30 | 55.57 | 31.11 | 5.98 | Mốc tốt nhất nếu ưu tiên small/NR. |
| noisereduce | 50 | 55.57 | 37.50 | 0.00 | Không tốt hơn epoch30 với modified beam. |

## 13. Kết quả đạt được hiện tại

Nếu chỉ nhìn số thấp nhất, `beam_search` cho WER `0.00` ở một vài exp. Tuy nhiên đây là kết quả cần kiểm chứng thêm, không nên dùng làm kết luận chính ngay.

### 13.1. Beam search non-zero

| Rank | Data | Model | Epoch | WER | Nhận xét |
|---:|---|---|---:|---:|---|
| 1 | noisereduce | small | 30 | 5.98 | Mạnh nhất nếu chấp nhận beam search non-zero, vẫn cần audit. |
| 2 | noisereduce | tiny | 30 | 20.52 | Rất đáng chú ý vì chỉ 8.15M params. |
| 3 | raw | base | 30 | 23.51 | Raw tốt nhất hiện tại. |
| 4 | noisereduce | base | 30 | 25.27 | Gần raw base beam, nhưng kém tiny/small NR. |

### 13.2. Modified beam và greedy

| Nhóm | Data | Model | Epoch | Decode | WER | Nhận xét |
|---|---|---|---:|---|---:|---|
| Best modified beam | noisereduce | small | 30 | `modified_beam_search` | 31.11 | Mốc báo cáo thận trọng nhất. |
| Best raw modified beam | raw | small | 50 | `modified_beam_search` | 43.34 | Raw tốt nhất nếu không dùng full beam. |
| Raw epoch30 modified beam | raw | tiny | 30 | `modified_beam_search` | 53.40 | Gần hòa base raw epoch30. |
| Best greedy | raw | small | 50 | `greedy_search` | 55.30 | Mốc greedy tốt nhất. |

### 13.3. Nhận xét tổng quát

| Nhận xét | Ý nghĩa |
|---|---|
| Raw epoch30 không đủ để chọn `small`. | `base/tiny` tốt hơn `small` ở raw epoch30. |
| `small` là hướng tốt nhất sau khi thêm NR hoặc train đủ lâu. | NR epoch30 và raw epoch50 cho thấy small có tiềm năng hơn. |
| `base raw epoch30 beam_search` là mốc rất mạnh. | Nếu chấp nhận beam_search, base raw đạt 23.51, tốt hơn mọi modified beam hiện có. |
| `tiny_nr beam_search` cũng đáng chú ý. | Tiny chỉ 8.15M params nhưng đạt WER 20.52 với NR beam search. |
| `modified_beam_search` cải thiện rõ so với greedy. | Decode strategy ảnh hưởng rất mạnh tới WER của Transducer. |
| `noisereduce` có tín hiệu tốt ở `small_nr epoch30 modified_beam_search`. | Nên tiếp tục kiểm chứng bằng test set sạch và tránh leakage. |
| `small_nr epoch50` không tốt hơn epoch30 ở modified beam. | Có thể epoch30 là checkpoint tốt hơn, hoặc train tiếp bắt đầu overfit. |
| `beam_search 0.00` là cờ đỏ cần audit. | Nên kiểm tra trùng utterance giữa train/test, transcript, manifest, và log hypothesis/reference. |

## 14. Độ chặt chẽ hiện tại

Hiện tại câu chuyện đã chặt hơn trước khá nhiều vì đã có:

| Đã có | Ý nghĩa |
|---|---|
| Raw epoch30 greedy cho `base/small/tiny` | So sánh model size ở điều kiện decode nhanh. |
| Raw epoch30 modified beam cho `base/small/tiny` | So sánh model size công bằng hơn với decode mạnh hơn. |
| Raw epoch30 beam search cho `base/small/tiny` | Thấy rõ `base` thắng khi dùng full beam search. |
| NR epoch30 modified beam cho `base/small/tiny` | Thấy rõ `small` thắng trong nhánh noisereduce. |
| NR epoch30 beam search cho `base/small/tiny` | Thấy rõ `small` thắng, `tiny` cũng rất mạnh so với số tham số. |
| Raw small epoch30 đủ 3 decode methods | Biết được checkpoint này yếu thật, không chỉ do greedy. |
| Small raw epoch50 đủ 3 decode methods | Thấy rõ train lâu hơn và beam decode giúp cải thiện. |
| Small NR epoch30/50 đủ 3 decode methods | Thấy epoch30 có thể là checkpoint tốt hơn epoch50. |
| Thời gian train cho các exp chính | Có thể nói được trade-off WER vs params vs training time. |

Những điểm vẫn còn thiếu nếu muốn báo cáo thật kín:

| Thiếu gì | Vì sao nên chạy |
|---|---|
| Decode nhiều checkpoint `small_nr` như epoch20, epoch25, epoch35, epoch40 | Vì epoch30 đang tốt hơn epoch50 với modified beam. |
| Audit train/test leakage | Vì `beam_search` có WER `0.00`. |

## 15. Kết luận ngắn

Hiện tại hướng cải thiện hợp lý nhất là:

```text
1. Dùng base raw epoch30 làm baseline ban đầu: WER 65.35.
2. Decode baseline bằng modified beam: base raw epoch30 xuống 53.53.
3. Decode base raw epoch30 bằng beam search: xuống 23.51, mốc raw mạnh nhất hiện tại.
4. Thử model size ở raw epoch30: beam search cho thấy base 23.51, tiny 41.71, small 61.68.
5. Dùng noisereduce: small_nr epoch30 tốt nhất giữa base/small/tiny với modified beam 31.11 và beam search 5.98.
6. Train small lên 50 epoch: raw greedy giảm còn 55.30, modified beam giảm còn 43.34.
7. Beam search cho số tốt nhất, kể cả 0.00, nhưng phải audit trước khi báo cáo là kết quả chính.
```

Nếu chỉ báo cáo modified beam, mốc tốt nhất là **WER 31.11** với `small + noisereduce + epoch30 + modified_beam_search`, huấn luyện khoảng **11 phút 45 giây** cho 30 epoch. Nếu chấp nhận beam search non-zero, mốc mạnh nhất hiện tại là **WER 5.98** với `small + noisereduce + epoch30 + beam_search`, còn raw mạnh nhất là **WER 23.51** với `base + raw + epoch30 + beam_search`. Các kết quả `beam_search = 0.00` vẫn cần audit dữ liệu và log decode trước khi xem là kết luận cuối.
