# Benchmark

Đo ngày 22–23/11/2026 trên Apple M1 Max, CPU. Mọi hệ thống được chạy lại trên cùng máy và chấm bằng cùng một bộ chuẩn
hóa văn bản (`vnasr.text.normalize_text`, phiên bản 1.3: chữ thường, số đọc thành chữ, bỏ dấu câu), nên các con số so
được với nhau. Chúng có thể khác số trong model card của từng tác giả vài phần mười đến 1 điểm vì khác cách chuẩn hóa
(ví dụ hynt 30M streaming ghi 9,72% trên GigaSpeech2, chúng tôi đo được 8,87%).

**SyER** (syllable error rate) = (âm tiết thay + xóa + chèn) / số âm tiết của lời gốc. Tiếng Việt viết tách âm tiết
nên SyER gần như trùng với WER mà các tác giả khác báo cáo.

## Bộ dữ liệu

| Bộ | Số câu | Thời lượng | Với VietNote ASR | Ghi chú |
|---|---|---|---|---|
| [VietMed](https://huggingface.co/datasets/leduckhai/VietMed) test | 3.437 | 6,0 giờ | **zero-shot** | hội thoại y tế, nhiều vùng miền, 7 điều kiện thu |
| [GigaSpeech2](https://huggingface.co/datasets/speechcolab/gigaspeech2)-vi test | 4.422 | 11,0 giờ | **zero-shot** | video YouTube tiếng Việt |
| [FLEURS](https://huggingface.co/datasets/google/fleurs) vi test + [VIVOS](https://huggingface.co/datasets/AILAB-VNUHCM/vivos) test | 1.617 | 3,8 giờ | đã train trên tập train | giọng đọc |
| Dev pha tiếng Anh | 1.600 | 4,6 giờ | cùng nguồn với dữ liệu train | 4 nguồn × 400 câu: hội thoại pha tiếng Anh (YouTube), đa vùng miền, đọc, y tế pha tiếng Anh |

Model của hynt train trên VietMed_Labeled, Sub-GigaSpeech2-Vi và FLEURS (theo model card). Model card của VietASR không
ghi dữ liệu train.

## Kết quả (SyER %)

| Hệ thống | Loại | Tham số | VietMed ⭐ | GigaSpeech2 ⭐ | FLEURS+VIVOS | Dev | Từ tiếng Anh nhận đúng (dev) |
|---|---|---|---|---|---|---|---|
| **VietNote ASR streaming fp32** | streaming | 77,7M | **13,87** | 10,65 | **6,88** | **9,62** | **75,5%** |
| **VietNote ASR streaming int8** | streaming | 77,7M | – | – | 7,28 | – | – |
| VietNote ASR offline fp32 | offline | 77,7M | 13,44 | 9,67 | 5,69 | 8,64 | 76,5% |
| VietNote ASR offline int8 | offline | 77,7M | – | – | 6,09 | – | – |
| [hynt Zipformer 30M streaming](https://huggingface.co/hynt/Zipformer-30M-RNNT-Streaming-6000h) | streaming | 30M | 14,30 | **8,87** | 7,73 | 12,58 | 24,7% |
| [Moonshine streaming tiny vi](https://huggingface.co/moonshine-ai/moonshine-streaming-tiny-vi) | streaming | 27M | 19,74 | 14,38 | 10,50 | 22,10 | 20,8% |
| [hynt Zipformer 150M](https://huggingface.co/hynt/ZipFormer-150M-CR-CTC-RNNT-6000h) | offline | 153M | 11,90 | 7,20 | 5,14 | 10,81 | 35,5% |
| [hynt Zipformer 30M](https://huggingface.co/hynt/Zipformer-30M-RNNT-6000h) | offline | 30M | 12,93 | 7,40 | 6,06 | 11,01 | 28,9% |
| [VietASR Zipformer 70k giờ](https://huggingface.co/zzasdf/viet_iter3_pseudo_label) | offline | 68M | 14,19 | 6,75 | 7,50 | 13,22 | 14,3% |
| [PhoWhisper large](https://huggingface.co/vinai/PhoWhisper-large) | offline | 1,5B | 16,74 | – | 6,03 | 16,31 | 30,8% |
| [Whisper large-v3](https://huggingface.co/openai/whisper-large-v3) | offline | 1,5B | 36,64 | – | 8,08 | 15,21 | 67,8% |

⭐ zero-shot với VietNote ASR. Model streaming được đưa âm thanh vào từng 100 ms như khi nghe micro. Giải mã: beam
search 4 nhánh (modified beam search) cho các model transducer. Whisper/PhoWhisper chưa chạy trên GigaSpeech2. Bản int8
mới đo trên FLEURS+VIVOS.

## Theo vùng miền và điều kiện thu (VietMed)

| | Số câu | VietNote streaming | hynt 30M streaming | hynt 150M | PhoWhisper |
|---|---|---|---|---|---|
| Đông Nam Bộ | 750 | 8,10 | 8,43 | 6,36 | 11,89 |
| Bắc | 693 | 9,83 | 10,22 | 7,92 | 11,10 |
| Tây Nam Bộ | 1.164 | 16,18 | 16,57 | 15,01 | 18,80 |
| Nam Trung Bộ | 809 | 18,63 | 19,30 | 15,19 | 22,35 |
| Bắc Trung Bộ | 21 | 30,70 | 28,45 | 31,83 | 34,37 |
| Podcast | 766 | 8,90 | 9,00 | 7,98 | 13,55 |
| Điện thoại | 607 | 9,73 | 10,45 | 7,86 | 11,29 |
| Tin tức | 312 | 10,48 | 11,04 | 8,89 | 13,27 |
| Chẩn đoán | 192 | 9,23 | 9,95 | 6,92 | 10,45 |
| Tư vấn | 446 | 17,35 | 17,43 | 15,14 | 18,73 |
| Talkshow | 730 | 18,25 | 18,42 | 16,69 | 20,13 |
| Bài giảng | 384 | 20,84 | 22,15 | 15,97 | 27,62 |

## Tốc độ (1 lõi CPU, M1 Max)

RTF = thời gian xử lý / thời lượng âm thanh (càng nhỏ càng nhanh). Streaming: đưa từng 100 ms; offline: cả câu một lần.
Đo trên 200 câu FLEURS khi máy không chạy việc khác.

| Model | RTF | Nhanh gấp thời gian thực |
|---|---|---|
| VietNote ASR streaming int8 | 0,037 | 27× |
| VietNote ASR streaming fp32 | 0,057 | 17× |
| VietNote ASR offline int8 | 0,026 | 38× |
| VietNote ASR offline fp32 | 0,047 | 21× |
| hynt Zipformer 30M streaming (fp16) | 0,037 | 27× |
| hynt Zipformer 30M offline (int8) | 0,018 | 55× |

hynt 30M offline đo trên 60 câu FLEURS cùng điều kiện. Model hynt nhỏ hơn 2,5 lần nên offline nhanh hơn; ở chế độ
streaming hai model nhanh ngang nhau.

## Chấm lại

```bash
python scripts/evaluate.py --model models/streaming/fp32 --tsv your_test.tsv
```

`your_test.tsv`: mỗi dòng `đường_dẫn_audio<TAB>lời gốc`. Bộ dev pha tiếng Anh không công bố được vì một phần nguồn âm
thanh không cho phép phân phối lại.

## Lỗi đã biết của bản int8 mặc định

Bản int8 do bộ xuất ONNX của icefall tạo mặc định (lượng tử hóa theo cả tensor) làm model này dừng ra chữ sau vài giây:
SyER 82% trên FLEURS+VIVOS. Bản phát hành dùng lượng tử hóa theo từng kênh (`scripts/quantize_int8.py`), chỉ kém fp32
khoảng 0,4 điểm.
