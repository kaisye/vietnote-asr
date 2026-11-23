# Fine-tune VietNote ASR trên dữ liệu của bạn

Khi nào nên fine-tune: model cần nghe tốt hơn một **giọng vùng miền**, một **lĩnh vực** (y tế, pháp lý, tổng đài…),
các **tên riêng / thuật ngữ** lặp lại thường xuyên, hoặc một **môi trường thu âm** cụ thể. Với vài tên riêng thì dùng
[hotword](../README.md#hotword-tên-riêng) trước, không cần train.

Quy trình bên dưới đã được chạy thử từ đầu đến cuối (chuẩn bị dữ liệu → fine-tune → xuất ONNX → giải mã bằng sherpa-onnx).

## 1. Cần gì

| | |
|---|---|
| GPU | NVIDIA ≥ 12 GB (đã thử trên RTX 5060 Ti 16 GB). CPU-only chạy được nhưng rất chậm. |
| Phần mềm | Python 3.11–3.12, PyTorch + [k2](https://k2-fsa.github.io/k2/), [lhotse](https://github.com/lhotse-speech/lhotse), [icefall](https://github.com/k2-fsa/icefall) |
| Checkpoint | `pytorch/pretrained.pt` và `bpe.model`, `tokens.txt` trên Hugging Face |
| Dữ liệu | Tối thiểu khoảng 1–2 giờ có nhãn đúng; 10–50 giờ cho thay đổi rõ rệt. Câu dài 1–20 giây. |

Cài đặt icefall theo [hướng dẫn chính thức](https://k2-fsa.github.io/icefall/installation/index.html), rồi chốt đúng
commit đã thử:

```bash
git clone https://github.com/k2-fsa/icefall && cd icefall
git checkout 9c966762e058bd7ba79d9da71e42fc06b64b703e
export PYTHONPATH=$PWD:$PYTHONPATH
cd ..
```

Tải checkpoint:

```bash
bash scripts/download_models.sh pytorch      # -> models/pytorch/pretrained.pt, models/bpe.model, models/tokens.txt
```

## 2. Tạo recipe

`recipe/make_recipe.py` tạo thư mục recipe từ icefall: mã model được liên kết nguyên vẹn, chỉ phần đọc dữ liệu được vá
để dùng dữ liệu của bạn (đặc trưng tính trực tiếp khi train, có đổi tốc độ ngẫu nhiên 0,9/1,1).

```bash
python recipe/make_recipe.py --icefall /path/to/icefall --out zipformer_vi
```

## 3. Chuẩn bị dữ liệu

Tạo file TSV, mỗi dòng một câu: `đường_dẫn_audio<TAB>lời`.

```
/data/my/0001.wav	Khách hàng chuyển khoản 2,5 triệu cho Vietcombank lúc 9 giờ sáng.
/data/my/0002.wav	Bệnh nhân dùng Paracetamol 500 mg, ngày 3 lần.
```

```bash
python recipe/prepare_data.py --tsv my_data.tsv --out data/my_cuts --valid-fraction 0.05
```

**Quan trọng:** lời được chuẩn hóa giống hệt nhãn lúc train (`vnasr.text.normalize_text`): chữ thường, số đọc thành
chữ ("2,5 triệu" → "hai phẩy năm triệu"), bỏ dấu câu. Bạn cứ viết lời bình thường; đừng tự chuẩn hóa theo cách khác,
nếu không model sẽ học hai kiểu viết. Âm thanh ở mọi tần số lấy mẫu đều được đưa về 16 kHz mono.

Mẹo:
- Trộn thêm một phần dữ liệu tổng quát (ví dụ [FLEURS](https://huggingface.co/datasets/google/fleurs) hoặc
  [Common Voice](https://commonvoice.mozilla.org) tiếng Việt), khoảng 20–30% số giờ, để model không quên tiếng Việt
  chung và tiếng Anh. Chỉ cần gộp các dòng vào cùng một file TSV.
- Giữ một tập kiểm tra riêng (không nằm trong TSV) để so trước/sau.

## 4. Fine-tune

```bash
bash recipe/finetune.sh $PWD/zipformer_vi $PWD/data/my_cuts \
    $PWD/models/pytorch/pretrained.pt $PWD/models/bpe.model $PWD/exp/my_ft 10
```

Tham số trong `finetune.sh`:
- Cấu hình model (`--num-encoder-layers`, `--encoder-dim`, `--causal 1`, `--chunk-size`…) **phải giữ nguyên**, vì đó là
  kiến trúc của checkpoint.
- `--base-lr 0.0045 --lr-batches 100000 --lr-epochs 100`: learning rate thấp và gần như không giảm, theo khuyến nghị
  fine-tune của icefall. Dữ liệu ít (dưới 5 giờ) thì giảm còn `0.002` và train 3–5 epoch.
- `MAX_DURATION=300` (giây âm thanh mỗi batch) vừa 16 GB; GPU 12 GB thì dùng `MAX_DURATION=200`.
- Train lại từ mô hình gốc thay vì train tiếp nhiều vòng: theo dõi `validation loss` trong `exp/my_ft/log`; khi nó
  không giảm nữa thì dừng.

Thời gian (RTX 5060 Ti, đo thật): khoảng 6 giây cho mỗi giờ dữ liệu mỗi epoch, cộng khoảng 1 phút khởi động. 10 giờ dữ liệu × 10 epoch ≈ 15 phút; 100 giờ × 10 epoch ≈ 2 giờ.

## 5. Xuất ONNX và dùng

```bash
bash recipe/export_onnx.sh $PWD/zipformer_vi $PWD/exp/my_ft $PWD/models/tokens.txt 10 3   # epoch 10, trung bình 3 epoch
```

Lấy trung bình vài epoch cuối (`avg` 2–4) thường tốt hơn một epoch đơn lẻ. Kết quả là encoder/decoder/joiner `.onnx`
(fp32) và `.int8.onnx`, dùng y như model phát hành:

```bash
python examples/transcribe_file.py test.wav --model exp/my_ft
```

**Bản int8:** bản int8 mà icefall tạo mặc định có thể làm model streaming dừng ra chữ sau vài giây. Hãy lượng tử hóa
lại từ bản fp32 theo từng kênh (per-channel), như bản phát hành:

```bash
python scripts/quantize_int8.py exp/my_ft
```

## 6. Kiểm tra trước/sau

```bash
python scripts/evaluate.py --model exp/my_ft --tsv my_test.tsv        # SyER của model fine-tune
python scripts/evaluate.py --model models/streaming/fp32 --tsv my_test.tsv   # model gốc
```

SyER (tỷ lệ âm tiết sai) tính trên văn bản đã chuẩn hóa ở cả hai phía, giống bảng benchmark trong README.

## Câu hỏi thường gặp

- **Thêm từ mới vào từ điển?** Không cần. Bộ token BPE 6.144 dùng chung Việt–Anh, ghép được mọi từ. Chỉ cần có từ đó
  trong lời của dữ liệu fine-tune.
- **Model quên tiếng Anh sau khi fine-tune?** Trộn thêm dữ liệu pha tiếng Anh hoặc giảm số epoch / learning rate.
- **Muốn bản offline (chính xác hơn khi không cần streaming)?** Dùng `export-streaming-as-non-streaming-onnx.py` trong
  thư mục recipe với `--chunk-size -1 --left-context-frames -1`.
