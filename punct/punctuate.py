"""Add punctuation and casing to lowercase ASR text with a model from train_punct.py.

    python punct/punctuate.py --model-dir models/punct "hôm nay mình sẽ hướng dẫn các bạn cách train model"
    python punct/punctuate.py --model-dir models/punct --file transcript.txt     # lines "[mm:ss] text"
"""

import argparse
import json
import re
from pathlib import Path

MARK = {"O": "", "COMMA": ",", "PERIOD": ".", "QUESTION": "?"}


class Punctuator:
    """model_dir holds best.pt (torch) or model.int8.onnx / model.onnx (onnxruntime, no torch needed).
    The ONNX export (export_onnx.py) is used when the directory has no best.pt or when onnx=True. fp32 ONNX gives
    the same output as torch. The pruned-vocabulary export (prune_vocab.py) gives the same output at 399 MB; its int8
    file (int8=True, 206 MB, about 2x faster) is within 0.006 F1 of fp32."""

    def __init__(self, model_dir: Path, device: str = "cpu", onnx: bool | None = None, threads: int = 2,
                 int8: bool = False):
        from transformers import AutoTokenizer
        model_dir = Path(model_dir)
        if onnx is None:
            onnx = not (model_dir / "best.pt").exists()
        if onnx and (model_dir / "onnx").is_dir() and not list(model_dir.glob("*.onnx")):
            model_dir = model_dir / "onnx"
        cfg = json.loads((model_dir / "config.json").read_text())
        self.punct, self.case = cfg["punct"], cfg["case"]
        self.tok = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
        self.lexicon = json.loads((model_dir / "mixed_case.json").read_text())
        if onnx:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = threads
            f = model_dir / ("model.int8.onnx" if int8 else "model.onnx")
            self.sess = ort.InferenceSession(str(f), so,
                                             providers=["CPUExecutionProvider"])
            self.model = None
        else:
            import torch
            from train_punct import Restorer
            self.model = Restorer(cfg["model"])
            self.model.load_state_dict(torch.load(model_dir / "best.pt", map_location="cpu"))
            self.model.to(device).eval()
            self.device = device

    def _predict(self, chunk):
        """(punct ids, case ids, first sub-word index per word) for one chunk of words."""
        if self.model is None:
            enc = self.tok([chunk], is_split_into_words=True, truncation=True, max_length=192, return_tensors="np")
            op, oc = self.sess.run(None, {"input_ids": enc["input_ids"].astype("int64"),
                                          "attention_mask": enc["attention_mask"].astype("int64")})
            pp, pc = op.argmax(-1)[0].tolist(), oc.argmax(-1)[0].tolist()
        else:
            import torch
            enc = self.tok([chunk], is_split_into_words=True, truncation=True, max_length=192, return_tensors="pt")
            with torch.no_grad():
                op, oc = self.model(enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device))
            pp, pc = op.argmax(-1)[0].cpu().tolist(), oc.argmax(-1)[0].cpu().tolist()
        first = {}
        for j, w in enumerate(enc.word_ids(0)):
            if w is not None and w not in first:
                first[w] = j
        return pp, pc, first

    def __call__(self, text: str, capitalise_first: bool = True) -> str:
        words = text.split()
        if not words:
            return ""
        out = []
        for start in range(0, len(words), 120):  # keep inputs within the encoder's length
            chunk = words[start:start + 120]
            pp, pc, first = self._predict(chunk)
            for i, w in enumerate(chunk):
                j = first.get(i)
                p, c = (self.punct[pp[j]], self.case[pc[j]]) if j is not None else ("O", "LOWER")
                if c == "CAP":
                    w = w[:1].upper() + w[1:]
                elif c == "UPPER":
                    w = w.upper()
                elif c == "MIXED":
                    w = self.lexicon.get(w, w[:1].upper() + w[1:])
                out.append([w, MARK[p]])
        # Sentence starts are capitalised regardless of the casing head.
        for i, (w, _) in enumerate(out):
            if (i == 0 and capitalise_first) or (i > 0 and out[i - 1][1] in (".", "?")):
                out[i][0] = w[:1].upper() + w[1:]
        return " ".join(w + m for w, m in out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--file", type=Path)
    p.add_argument("--onnx", action="store_true")
    p.add_argument("text", nargs="?")
    a = p.parse_args()
    punct = Punctuator(a.model_dir, onnx=a.onnx or None)
    if a.file:
        for line in a.file.read_text(encoding="utf-8").splitlines():
            m = re.match(r"(\[\d+:\d+\]\s*)?(.*)", line)
            print((m.group(1) or "") + punct(m.group(2)))
    else:
        print(punct(a.text))


if __name__ == "__main__":
    main()
