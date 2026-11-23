"""Train a punctuation and capitalisation restorer for the ASR output.

Input is what the recogniser emits: lowercase words, no punctuation, numbers spelled out
(vnasr.text.normalize_text). For every input word the model predicts
  * the punctuation that follows it: none, comma, period, question;
  * its casing: lower, Capitalised, UPPER, or MIXED (ChatGPT, OpenClaw), where the mixed
    surface form comes from a lexicon of the most frequent spelling in the training text.
Labels come from punctuated transcripts (csom, ViMD, ViMedCSS, LSVSC), using their own
train/dev/test splits. One XLM-RoBERTa encoder with two token-classification heads; the
label of a word sits on its first sub-word.

    python punct/train_punct.py --data <jsonl of {"text": ...} sentences> --out exp/punct
"""

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import time

import torch
from torch import nn

from vnasr.text import normalize_text

PUNCT = ["O", "COMMA", "PERIOD", "QUESTION"]
CASE = ["LOWER", "CAP", "UPPER", "MIXED"]
EDGE = re.compile(r"^[\W_]+|[\W_]+$")


def punct_of(trail: str) -> int:
    if "?" in trail:
        return 3
    if any(c in trail for c in ".!…"):
        return 2
    if any(c in trail for c in ",;:—–-"):
        return 1
    return 0


def case_of(word: str) -> int:
    if word == word.lower():
        return 0
    if word[0].isupper() and word[1:] == word[1:].lower():
        return 1
    if word.isupper() and len(word) > 1:
        return 2
    return 3


def to_example(text: str):
    """Return (words, punct labels, case labels, surface forms) for one punctuated sentence."""
    words, punct, case, surface = [], [], [], []
    for raw in text.split():
        core = EDGE.sub("", raw)
        trail = raw[len(raw.rstrip(".,?!;:…—–-\"')]}")):] if raw else ""
        if not core:  # a standalone dash or ellipsis belongs to the previous word
            if punct and punct_of(raw):
                punct[-1] = max(punct[-1], punct_of(raw))
            continue
        norm = normalize_text(core).split()
        if not norm:
            continue
        c = case_of(core) if len(norm) == 1 else 0
        for i, w in enumerate(norm):
            words.append(w)
            punct.append(punct_of(trail) if i == len(norm) - 1 else 0)
            case.append(c if i == 0 else 0)
            surface.append(core if len(norm) == 1 else w)
    return words, punct, case, surface


class Restorer(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(name)
        h = self.encoder.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.punct = nn.Linear(h, len(PUNCT))
        self.case = nn.Linear(h, len(CASE))

    def forward(self, input_ids, attention_mask):
        x = self.drop(self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state)
        return self.punct(x), self.case(x)


def encode(tok, batch, max_len=192):
    """Tokenise pre-split words; labels on the first sub-word of each word, -100 elsewhere."""
    enc = tok([b[0] for b in batch], is_split_into_words=True, truncation=True, max_length=max_len,
              padding=True, return_tensors="pt")
    lp = torch.full(enc["input_ids"].shape, -100)
    lc = torch.full(enc["input_ids"].shape, -100)
    for i, (_, p, c, _) in enumerate(batch):
        prev = None
        for j, w in enumerate(enc.word_ids(i)):
            if w is not None and w != prev:
                lp[i, j], lc[i, j] = p[w], c[w]
            prev = w
    return enc, lp, lc


def evaluate(model, tok, data, device, bs=64):
    model.eval()
    conf = torch.zeros(len(PUNCT), len(PUNCT), dtype=torch.long)
    case_ok = case_n = 0
    with torch.no_grad():
        for i in range(0, len(data), bs):
            enc, lp, lc = encode(tok, data[i:i + bs])
            op, oc = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
            pp, pc = op.argmax(-1).cpu(), oc.argmax(-1).cpu()
            m = lp != -100
            for g, p in zip(lp[m].tolist(), pp[m].tolist()):
                conf[g, p] += 1
            case_ok += (pc[m] == lc[m]).sum().item()
            case_n += m.sum().item()
    model.train()
    out = {}
    for k in range(1, len(PUNCT)):
        tp = conf[k, k].item()
        prec = tp / max(1, conf[:, k].sum().item())
        rec = tp / max(1, conf[k, :].sum().item())
        out[PUNCT[k]] = 2 * prec * rec / max(1e-9, prec + rec)
    out["case_acc"] = case_ok / max(1, case_n)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="FacebookAI/xlm-roberta-base")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--init", type=Path, help="state dict to start from (e.g. a previous best.pt)")
    p.add_argument("--join", type=float, default=0.5,
                   help="probability of joining 2-3 sentences, so the model sees sentence boundaries")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)

    rows = [json.loads(line) for line in a.data.open(encoding="utf-8")]
    train_ex = [to_example(r["text"]) for r in rows if r["split"] == "train"]
    dev_ex = [to_example(r["text"]) for r in rows if r["split"] == "dev"]
    train_ex = [e for e in train_ex if e[0]]
    dev_ex = [e for e in dev_ex if e[0]]

    forms = defaultdict(Counter)  # mixed-case lexicon
    for words, _, case, surface in train_ex:
        for w, c, s in zip(words, case, surface):
            if c == 3:
                forms[w][s] += 1
    lexicon = {w: c.most_common(1)[0][0] for w, c in forms.items()}
    (a.out / "mixed_case.json").write_text(json.dumps(lexicon, ensure_ascii=False, indent=0))
    print(f"train {len(train_ex)} dev {len(dev_ex)} mixed-case forms {len(lexicon)}")

    def joined(data):
        out, i = [], 0
        data = data[:]
        rng.shuffle(data)
        while i < len(data):
            k = rng.choice([2, 3]) if rng.random() < a.join else 1
            parts = data[i:i + k]
            i += k
            out.append(tuple(sum((list(x[j]) for x in parts), []) for j in range(4)))
        return out

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    tok = AutoTokenizer.from_pretrained(a.model)
    model = Restorer(a.model)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location="cpu"))
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * (len(train_ex) // a.bs + 1)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    # Rarer punctuation gets more weight so the model does not settle for "no punctuation".
    loss_p = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 2.0, 3.0], device=device), ignore_index=-100)
    loss_c = nn.CrossEntropyLoss(ignore_index=-100)
    dev_fixed = joined(dev_ex)[:3000]

    step = 0
    best = -1.0
    for ep in range(a.epochs):
        data = joined(train_ex)
        for i in range(0, len(data), a.bs):
            enc, lp, lc = encode(tok, data[i:i + a.bs])
            op, oc = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
            loss = loss_p(op.reshape(-1, len(PUNCT)), lp.to(device).reshape(-1)) + \
                0.5 * loss_c(oc.reshape(-1, len(CASE)), lc.to(device).reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 200 == 0:
                print(f"{time.strftime('%H:%M:%S')} epoch {ep} step {step}/{steps} loss {loss.item():.3f}", flush=True)
                torch.save(model.state_dict(), a.out / "last.pt")
            if step % 1000 == 0 or i + a.bs >= len(data):
                m = evaluate(model, tok, dev_fixed, device)
                if device == "mps":
                    torch.mps.synchronize()
                    torch.mps.empty_cache()
                score = (m["COMMA"] + m["PERIOD"] + m["QUESTION"]) / 3
                print("dev", {k: round(v, 3) for k, v in m.items()}, f"macroF1 {score:.3f}", flush=True)
                if score > best:
                    best = score
                    torch.save(model.state_dict(), a.out / "best.pt")
                    tok.save_pretrained(a.out / "tokenizer")
                    (a.out / "config.json").write_text(json.dumps(
                        {"model": a.model, "punct": PUNCT, "case": CASE, "step": step, "dev": m}, indent=1))
    print("best macro F1", round(best, 3))


if __name__ == "__main__":
    main()
