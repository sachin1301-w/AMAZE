"""Stage 1b - dense embeddings for every record.

    python embed.py train
    python embed.py test

Lexical blocking is weakest exactly where this data is hardest: a name
transliterated into Devanagari or Kannada shares no character n-gram with its
Latin form, and a scrambled token (``Wooten`` -> ``Wheooitne``) shares almost
none.  A multilingual sentence encoder maps both to nearby vectors, so the
cosine between the two records becomes a feature the character-level features
cannot express.

The encoder is ``intfloat/multilingual-e5-small`` (MIT licence, 118M
parameters) -- comfortably inside the challenge's MIT/Apache and 8B-parameter
limits.  It is a general-purpose text encoder; no business directory, gazetteer
or entity database is consulted.

Vectors are written per (split, source) as float16, ordered exactly like the
corresponding parquet file.
"""
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import hw  # noqa: E402  (must precede numpy/torch)

import os
import sys
import time

import numpy as np

from hw import breathe
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

WORK = os.environ.get("BER_WORK", "../../../work")
DATA = os.environ.get("BER_DATA", "../../../dataset")
MODEL = os.environ.get("BER_EMB_MODEL", "intfloat/multilingual-e5-small")
MAXLEN = int(os.environ.get("BER_EMB_LEN", "48"))
BATCH = int(os.environ.get("BER_EMB_BATCH", "512"))


def build_text(name, addr):
    """What the encoder sees: the raw fields, so native scripts survive."""
    out = []
    for n, a in zip(name, addr):
        out.append("query: " + n + " ; " + a if a else "query: " + n)
    return out


@torch.no_grad()
def embed_all(texts, tok, model, dev, out_path, log_every=200):
    """Embed in length-sorted order.

    Batches are padded to their longest member, so mixing a 6-token name with a
    48-token address wastes most of the compute.  Sorting by length first makes
    every batch nearly uniform; it roughly halves the wall time here.  Results
    are written back in the original order.
    """
    n = len(texts)
    dim = model.config.hidden_size
    # Written straight to a disk-backed array: holding 5M x 384 float16 in RAM
    # costs ~4 GB, which this 16 GB machine does not have spare while the
    # tokeniser is also holding the text list.
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16,
                                    shape=(n, dim))
    order = np.argsort(np.fromiter((len(t) for t in texts), np.int32, n),
                       kind="stable")
    t0 = time.time()
    for i in range(0, n, BATCH):
        tb = time.time()
        sl = order[i:i + BATCH]
        batch = [texts[j] for j in sl]
        enc = tok(batch, padding=True, truncation=True, max_length=MAXLEN,
                  return_tensors="pt").to(dev)
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
            h = model(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
        v = (h * m).sum(1) / m.sum(1).clamp(min=1e-6)      # mean pooling
        v = torch.nn.functional.normalize(v.float(), dim=1)
        out[sl] = v.half().cpu().numpy()
        breathe(tb)
        if (i // BATCH) % log_every == 0 and i:
            el = time.time() - t0
            print(f"    {i:,}/{n:,}  {i/el:,.0f} rec/s  "
                  f"eta {(n-i)/max(1,i/el)/60:.0f} min", flush=True)
    out.flush()
    return n, dim


def main():
    split = sys.argv[1]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev} model={MODEL} maxlen={MAXLEN} batch={BATCH}",
          flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(dev).eval()
    if dev == "cuda":
        model = model.half()
    for src in (1, 2, 3):
        out = f"{WORK}/{split}_emb{src}.npy"
        if os.path.exists(out):
            print("exists", out, flush=True)
            continue
        t0 = time.time()
        df = pd.read_csv(f"{DATA}/{split}/{split}_source{src}.tsv", sep="\t",
                         dtype=str, keep_default_na=False,
                         usecols=["business_name", "business_address"])
        texts = build_text(df.business_name.values, df.business_address.values)
        del df
        print(f"  {split} s{src}: {len(texts):,} records", flush=True)
        tmp = out + ".tmp.npy"
        shape = embed_all(texts, tok, model, dev, tmp)
        del texts
        os.replace(tmp, out)
        print(f"  wrote {out} {shape} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
