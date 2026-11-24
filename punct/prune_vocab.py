"""Shrink the punctuation restorer: keep only the XLM-R vocabulary that Vietnamese/English ASR output uses (2026-11-24).

XLM-R base has 250,002 tokens; the embedding table is 192M of its 278M parameters, but the restorer only ever sees
lowercase Vietnamese/English words written by vnasr.text.normalize_text. We tokenise a text collection with the
original tokenizer, keep the ids that occur, and export an ONNX model whose first op maps original ids to the
smaller table (ids not kept become <unk>). The tokenizer and Punctuator code stay unchanged.

    python scripts/punct/prune_vocab.py --model-dir exp/punct --out exp/punct_small \
        --text data/punct/punct_text.jsonl data/soniox_cs/sxcs-train.jsonl --words /usr/share/dict/words
writes <out>/{model.onnx (399 MB), model.int8.onnx (206 MB), tokenizer/, config.json, mixed_case.json, kept_ids.json}
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
from train_punct import Restorer  # noqa: E402
from vnasr.text import normalize_text  # noqa: E402


def collect_ids(tok, texts, min_count):
    counts = {}
    words = set()
    for t in texts:
        words.update(normalize_text(t).split())
    words = sorted(words)
    for i in range(0, len(words), 4096):
        enc = tok(words[i:i + 4096], add_special_tokens=False)
        for ids in enc["input_ids"]:
            for x in ids:
                counts[x] = counts.get(x, 0) + 1
    return {x for x, c in counts.items() if c >= min_count}, len(words)


def read_texts(paths, split_filter=True):
    for p in paths:
        for line in Path(p).open(encoding="utf-8"):
            r = json.loads(line)
            if split_filter and r.get("split") not in (None, "train"):
                continue  # dev/test sentences stay unseen, so unk rates measured on them are honest
            if r.get("text"):
                yield r["text"]


class Pruned(nn.Module):
    def __init__(self, model: Restorer, kept: list[int], vocab_size: int, unk: int):
        super().__init__()
        remap = torch.full((vocab_size,), kept.index(unk), dtype=torch.long)
        remap[torch.tensor(kept)] = torch.arange(len(kept))
        self.register_buffer("remap", remap)
        emb = model.encoder.embeddings.word_embeddings
        small = nn.Embedding(len(kept), emb.embedding_dim, padding_idx=kept.index(emb.padding_idx))
        small.weight.data = emb.weight.data[kept].clone()
        model.encoder.embeddings.word_embeddings = small
        model.encoder.config.vocab_size = len(kept)
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(self.remap[input_ids], attention_mask)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--text", nargs="+", default=[], help="jsonl files with a 'text' field (train split only)")
    p.add_argument("--words", nargs="*", default=[], help="plain word lists, one word per line")
    p.add_argument("--min-count", type=int, default=1)
    a = p.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model_dir / "tokenizer")
    texts = list(read_texts(a.text))
    for f in a.words:
        texts += Path(f).read_text(encoding="utf-8", errors="ignore").split()
    ids, nwords = collect_ids(tok, texts, a.min_count)
    # Special tokens keep their positions 0..3 (<s>, <pad>, </s>, <unk>): XLM-R derives position ids from <pad>=1.
    kept = [0, 1, 2, 3] + sorted(x for x in ids if x > 3)
    print(f"{nwords} distinct words -> keep {len(kept)} of {len(tok)} tokens")

    cfg = json.loads((a.model_dir / "config.json").read_text())
    m = Restorer(cfg["model"])
    m.load_state_dict(torch.load(a.model_dir / "best.pt", map_location="cpu"))
    m.eval()
    pm = Pruned(m, kept, len(tok), tok.unk_token_id).eval()
    a.out.mkdir(parents=True, exist_ok=True)
    ids_ = torch.randint(5, 1000, (1, 24))
    torch.onnx.export(pm, (ids_, torch.ones_like(ids_)), a.out / "model.onnx",
                      input_names=["input_ids", "attention_mask"], output_names=["punct", "case"],
                      opset_version=17, dynamo=False,
                      dynamic_axes={k: {0: "b", 1: "t"} for k in ["input_ids", "attention_mask", "punct", "case"]})
    from onnxruntime.quantization import QuantType, quantize_dynamic
    # Per-channel int8 on MatMul weights, except the FFN output projection of layers 0-5, which carries activation
    # outliers: quantising those costs ~6 points of comma F1. Embeddings stay fp32 (int8 Gather costs ~2 points of
    # question F1). Result: 206 MB, comma/period/question F1 within 0.006 of fp32 on the dev split.
    keep_fp32 = [f"/model/encoder/encoder/layer.{i}/output/dense/MatMul" for i in range(6)]
    quantize_dynamic(a.out / "model.onnx", a.out / "model.int8.onnx", weight_type=QuantType.QInt8,
                     per_channel=True, op_types_to_quantize=["MatMul"], nodes_to_exclude=keep_fp32)
    for f in ("config.json", "mixed_case.json"):
        shutil.copy(a.model_dir / f, a.out / f)
    shutil.copytree(a.model_dir / "tokenizer", a.out / "tokenizer", dirs_exist_ok=True)
    (a.out / "kept_ids.json").write_text(json.dumps(kept))
    for f in sorted(a.out.glob("*.onnx")):
        print(f, f"{f.stat().st_size / 2**20:.0f} MB")


if __name__ == "__main__":
    main()
