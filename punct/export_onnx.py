"""Export the punctuation/casing restorer to ONNX and quantise it to int8 (2026-11-22).

    python punct/export_onnx.py --model-dir <dir with best.pt from train_punct.py>
writes <model-dir>/onnx/{model.onnx, model.int8.onnx} plus tokenizer/, config.json, mixed_case.json, so the
directory runs without torch (punctuate.Punctuator picks model.onnx when best.pt is absent or onnx=True; int8 loses about 7 points of punctuation agreement).
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from train_punct import Restorer  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    a = p.parse_args()
    out = a.model_dir / "onnx"
    out.mkdir(exist_ok=True)
    cfg = json.loads((a.model_dir / "config.json").read_text())
    m = Restorer(cfg["model"])
    m.load_state_dict(torch.load(a.model_dir / "best.pt", map_location="cpu"))
    m.eval()
    ids = torch.randint(5, 1000, (1, 24))
    mask = torch.ones_like(ids)
    torch.onnx.export(m, (ids, mask), out / "model.onnx", input_names=["input_ids", "attention_mask"],
                      output_names=["punct", "case"], opset_version=17, dynamo=False,
                      dynamic_axes={k: {0: "b", 1: "t"} for k in ["input_ids", "attention_mask", "punct", "case"]})
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(out / "model.onnx", out / "model.int8.onnx", weight_type=QuantType.QInt8)
    for f in ("config.json", "mixed_case.json"):
        shutil.copy(a.model_dir / f, out / f)
    shutil.copytree(a.model_dir / "tokenizer", out / "tokenizer", dirs_exist_ok=True)
    for f in sorted(out.glob("*.onnx")):
        print(f, f"{f.stat().st_size / 2**20:.0f} MB")


if __name__ == "__main__":
    main()
