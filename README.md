# VietNote ASR

**Nhận dạng giọng nói tiếng Việt pha tiếng Anh, chạy trực tiếp (streaming) trên CPU.**

[![Demo](https://img.shields.io/badge/demo-asr.vietnote.vn-0f6e56)](https://asr.vietnote.vn)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20models-Hugging%20Face-yellow)](https://huggingface.co/kaisye/vietnote-asr)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

Thử ngay trên trình duyệt: **https://asr.vietnote.vn** (nói vào micro, đọc kịch bản có sẵn, hoặc nghe đoạn mẫu).

- **Streaming:** chữ hiện ra trong lúc bạn nói, mỗi bước 640 ms âm thanh.
- **Tiếng Việt pha tiếng Anh (code-switching):** nhận đúng khoảng **75% từ tiếng Anh** trong câu nói pha, gấp 2–5 lần
  các model tiếng Việt mã nguồn mở khác mà chúng tôi đã thử.
- **Văn bản hoàn chỉnh:** số viết bằng chữ số (1,5 tỷ · 250.000 · 15/12 · 0912 345 678 · tỷ số 2-1), dấu câu, viết hoa,
  tên thương hiệu đúng chính tả; hiểu cả khi người nói vấp hoặc sửa giữa chừng ("ba… à không, bốn triệu" → 4 triệu).
- **Nhẹ, chạy trên CPU:** 83 MB (int8), không cần GPU; chạy bằng [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
  trên Linux, macOS, Windows, Android, iOS, Raspberry Pi.
- **Fine-tune được** trên dữ liệu của bạn: checkpoint PyTorch và recipe [icefall](https://github.com/k2-fsa/icefall)
  đi kèm, xem [docs/FINETUNE.md](docs/FINETUNE.md).

## Kết quả

Tỷ lệ âm tiết sai (SyER, %, càng thấp càng tốt). Tất cả hệ thống được chấm lại trên cùng máy, cùng bộ chuẩn hóa văn bản.
Chi tiết, cách đo và lưu ý: [docs/BENCHMARK.md](docs/BENCHMARK.md).

| Hệ thống | Loại | VietMed ⭐ | GigaSpeech2 ⭐ | FLEURS+VIVOS | Dev pha tiếng Anh | Từ tiếng Anh nhận đúng |
|---|---|---|---|---|---|---|
| **VietNote ASR (streaming)** | streaming, 77,7M | **13,87** | 10,65 | **6,88** | **9,62** | **75,5%** |
| hynt Zipformer 30M streaming | streaming, 30M | 14,30 | **8,87** | 7,73 | 12,58 | 24,7% |
| Moonshine streaming tiny vi | streaming, 27M | 19,74 | 14,38 | 10,50 | 22,10 | 20,8% |
| hynt Zipformer 150M | offline | 11,90 | 7,20 | 5,14 | 10,81 | 35,5% |
| hynt Zipformer 30M | offline | 12,93 | 7,40 | 6,06 | 11,01 | 28,9% |
| VietASR Zipformer (70k giờ) | offline | 14,19 | 6,75 | 7,50 | 13,22 | 14,3% |
| PhoWhisper large | offline | 16,74 | – | 6,03 | 16,31 | 30,8% |
| Whisper large-v3 | offline | 36,64 | – | 8,08 | 15,21 | 67,8% |

⭐ Zero-shot với VietNote ASR (không dùng phần nào của bộ này để train). hynt train trên VietMed_Labeled, Sub-GigaSpeech2
và FLEURS (theo model card của họ). VietNote ASR có dùng tập train của FLEURS và VIVOS; bộ dev cùng nguồn với dữ liệu train.

**Nói thẳng:** model offline lớn hơn (hynt 150M) chính xác hơn ở tiếng Việt thuần, và trên GigaSpeech2 chúng tôi kém các
model đã train trên bộ này. Điểm khác biệt của VietNote ASR là streaming, tiếng Anh trong câu và văn bản hoàn chỉnh.

Còn yếu: giọng Tây Nam Bộ và miền Trung (SyER 16–31% trên VietMed, so với 8–10% giọng Bắc và Đông Nam Bộ); hội thoại
nhiều người, xa micro; tên riêng và thuật ngữ tiếng Anh hiếm.

## Bắt đầu nhanh

```bash
git clone https://github.com/kaisye/vietnote-asr && cd vietnote-asr
pip install -e ".[asr]"
bash scripts/download_models.sh streaming/int8      # hoặc bỏ tham số để tải tất cả

python examples/transcribe_file.py audio.wav --model models/streaming/int8
```

```
raw : giá vé là hai trăm năm mươi nghìn một người tăng mười hai phần trăm so với năm hai nghìn không trăm hai mươi tư
itn : giá vé là 250.000 một người tăng 12% so với năm 2024
```

Thêm dấu câu và viết hoa (`pip install -e ".[punct]"`, tải `punct`):

```bash
python examples/transcribe_file.py audio.wav --punct models/punct --hotwords data/hotwords.txt
# text: Giá vé là 250.000 một người, tăng 12% so với năm 2024.
```

Phụ đề trực tiếp từ micro trong terminal: `pip install sounddevice && python examples/microphone.py`.

### Dùng trong code

```python
import sherpa_onnx, soundfile as sf
from vnasr.itn import spoken_to_digits

rec = sherpa_onnx.OnlineRecognizer.from_transducer(
    encoder="models/streaming/int8/encoder.int8.onnx",
    decoder="models/streaming/int8/decoder.int8.onnx",
    joiner="models/streaming/int8/joiner.int8.onnx",
    tokens="models/tokens.txt", num_threads=2,
    decoding_method="modified_beam_search", max_active_paths=4)

audio, sr = sf.read("audio.wav", dtype="float32")   # 16 kHz mono
s = rec.create_stream()
for i in range(0, len(audio), 1600):                 # 100 ms mỗi lần, như nhận từ micro
    s.accept_waveform(sr, audio[i:i + 1600])
    while rec.is_ready(s):
        rec.decode_stream(s)
    print("\r" + rec.get_result(s), end="")
s.input_finished()
while rec.is_ready(s):
    rec.decode_stream(s)
print("\n" + spoken_to_digits(rec.get_result(s).lower()))
```

## Các model

| Thư mục trên Hugging Face | Dùng khi | Kích thước | Tốc độ (1 lõi CPU) |
|---|---|---|---|
| `streaming/int8` | ứng dụng thời gian thực, điện thoại, máy yếu | 83 MB | nhanh gấp ~27 lần thời gian thực |
| `streaming/fp32` | thời gian thực, chính xác nhất (bản chạy trên web) | 274 MB | gấp ~17 lần |
| `offline/int8` | xử lý file, không cần streaming | 88 MB | gấp ~38 lần |
| `offline/fp32` | xử lý file, chính xác nhất | 280 MB | gấp ~21 lần |
| `pytorch/pretrained.pt` | fine-tune | 311 MB | – |
| `punct/` | dấu câu và viết hoa (XLM-RoBERTa base, ONNX) | 1,1 GB | 20–40 ms/câu |

Cùng một bộ trọng số: bản offline nhìn được cả câu nên chính xác hơn khoảng 1 điểm SyER.

## Văn bản hoàn chỉnh

Model trả ra chữ thường, số đọc thành chữ. Ba bước sau biến nó thành văn bản dùng được ngay; demo chạy theo đúng thứ tự:

```
ASR → ① ITN (số) → ② dấu câu + viết hoa → ③ hotword (tên riêng)
```

**① ITN** ([src/vnasr/itn.py](src/vnasr/itn.py)): luật viết tay, không tham số, khoảng 1,5 ms/câu.

```python
from vnasr.itn import spoken_to_digits
spoken_to_digits("chuyển khoản cho chị ba à không bốn triệu rưỡi nhé")   # 'chuyển khoản cho chị 4,5 triệu nhé'
spoken_to_digits("hẹn anh bảy giờ ba mươi ngày mười lăm tháng mười hai")  # 'hẹn anh 7 giờ 30 ngày 15 tháng 12'
spoken_to_digits("số điện thoại không chín một hai ba bốn năm sáu bảy tám")  # '... 0912345678'
spoken_to_digits("sau khi đạt tỷ số sáu sáu")                              # 'sau khi đạt tỷ số 6-6'
```

Đã kiểm tra trên [VietAI spoken_norm](https://huggingface.co/datasets/VietAI/spoken_norm_assignment): 78,7% số đúng giá
trị, chỉ 0,26% từ thường bị đổi nhầm; trên bản tin thời sự (lời đúng tuyệt đối) đúng 100% số lớn hơn 10. Chịu được người
nói vấp ("ừm", "à") và đọc sai rồi sửa.

**② Dấu câu và viết hoa** ([punct/](punct/)): XLM-RoBERTa base với hai đầu phân loại (dấu: không / phẩy / chấm / chấm hỏi;
chữ hoa: thường / Hoa đầu / HOA HẾT / hỗn hợp như iPhone). Viết hoa đúng 97%; F1 dấu chấm 0,73, chấm hỏi 0,67, phẩy 0,56.

**③ Hotword: tên riêng** ([src/vnasr/hotwords.py](src/vnasr/hotwords.py), [data/hotwords.txt](data/hotwords.txt)): danh sách
tên kèm cách đọc kiểu Việt, ví dụ `VinFast | vin phát`, `Nvidia | en vi đi a`. Tên được cộng điểm khi giải mã, cách đọc
được thay bằng tên viết đúng. Có sẵn khoảng 70 thương hiệu phổ biến ở Việt Nam; thêm tên của bạn vào file.

## Chạy demo web

```bash
pip install -e ".[demo]"
bash scripts/download_models.sh
python demo/server.py --model-dir models/streaming/fp32 --punct-dir models/punct \
    --vad models/silero_vad.onnx --hotwords data/hotwords.txt --bpe-model models/bpe.model
# mở http://localhost:8765
```

Máy chủ không lưu âm thanh trừ khi bật `--dump`. Triển khai lên VM (Caddy HTTPS, systemd): [deploy/](deploy/).
Một VM 4 vCPU phục vụ thoải mái 8–10 người nói cùng lúc (đo thật trên Google Cloud e2-standard-4).

## Model

- **Kiến trúc:** Zipformer2 causal (cỡ M, 77,7M tham số), stateless transducer, pruned RNN-T loss.
- **Token:** BPE 6.144, dùng chung tiếng Việt và tiếng Anh.
- **Streaming:** chunk 32 frame (640 ms), nhìn lại 128 frame; xuất ONNX cho sherpa-onnx.
- **Dữ liệu train:** khoảng 1.500 giờ, gồm các bộ dữ liệu tiếng Việt công khai, dữ liệu pha tiếng Anh thu thập thêm,
  dữ liệu tổng hợp (TTS) và một phần dữ liệu không thể chia sẻ; một phần nhãn được tạo tự động. Có dùng tập train của
  FLEURS và VIVOS. **Không dùng bất kỳ phần nào của VietMed hay GigaSpeech2.**
- **Train:** [icefall](https://github.com/k2-fsa/icefall), 1 GPU RTX 5060 Ti 16 GB, bf16, 45 epoch, lấy trung bình 2 checkpoint.

## Cấu trúc repo

```
src/vnasr/      ITN, hotword, chuẩn hóa văn bản, chấm điểm (SyER/CER)
examples/       nhận dạng file, micro
demo/           máy chủ web + giao diện (asr.vietnote.vn)
punct/          model dấu câu: suy luận, xuất ONNX, train
recipe/         tạo recipe icefall, chuẩn bị dữ liệu, fine-tune, xuất ONNX
scripts/        tải model, lượng tử hóa int8, đánh giá
deploy/         triển khai lên VM
tests/          ~1.500 test cho ITN, hotword, chuẩn hóa
```

Chạy test: `pip install -e ".[dev]" && pytest`.

## Giấy phép và ghi công

Code và trọng số: [Apache-2.0](LICENSE). Model dấu câu dựa trên
[XLM-RoBERTa base](https://huggingface.co/FacebookAI/xlm-roberta-base) (MIT). Âm thanh mẫu trong demo từ
[FLEURS](https://huggingface.co/datasets/google/fleurs) (CC BY 4.0). Xây dựng trên [k2](https://github.com/k2-fsa/k2),
[icefall](https://github.com/k2-fsa/icefall), [lhotse](https://github.com/lhotse-speech/lhotse) và
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx). Cảm ơn tác giả các bộ dữ liệu và model dùng để so sánh.

```bibtex
@misc{vietnote_asr_2026,
  title  = {VietNote ASR: streaming Vietnamese--English code-switching speech recognition},
  year   = {2026},
  url    = {https://github.com/kaisye/vietnote-asr}
}
```

---

## English summary

VietNote ASR is a **streaming Vietnamese–English (code-switching) speech recognizer** that runs on a CPU. It is a
77.7M-parameter causal Zipformer transducer (icefall), exported to ONNX for sherpa-onnx, plus a text pipeline that turns
the raw output into finished text: rule-based ITN for numbers, an XLM-R punctuation/casing model, and a hotword list for
brand names. On our code-switching dev set it recognises ~75% of English words (2–5× other open Vietnamese models); on
the zero-shot VietMed test it reaches 13.9% SyER, the best of the streaming models we tested. It is weaker than larger
offline models on pure Vietnamese (e.g. GigaSpeech2). Weights and code are Apache-2.0; a PyTorch checkpoint and an
icefall fine-tuning recipe are included. Live demo: https://asr.vietnote.vn
