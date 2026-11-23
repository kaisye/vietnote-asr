"""Quantise fp32 ONNX encoder/decoder/joiner to int8 the way the release does (per-channel MatMul weights).

The default dynamic int8 (per-tensor) of icefall's exporter makes this streaming model stop emitting text after a few
seconds (82% SyER on FLEURS+VIVOS). Per-channel int8 stays within about 0.3 points of fp32.

    python scripts/quantize_int8.py <model_dir>      # writes <name>.int8.onnx next to each fp32 file
"""

import argparse
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model_dir", type=Path)
    a = p.parse_args()
    for part in ("encoder", "decoder", "joiner"):
        for f in sorted(a.model_dir.glob(f"{part}*.onnx")):
            if ".int8." in f.name:
                continue
            out = f.with_name(f.name.replace(".onnx", ".int8.onnx"))
            quantize_dynamic(str(f), str(out), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"],
                             per_channel=True)
            print(f"{f.name} -> {out.name} ({out.stat().st_size / 2**20:.0f} MB)")


if __name__ == "__main__":
    main()
