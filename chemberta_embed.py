#!/usr/bin/env python3
"""Generate frozen ChemBERTa CLS embeddings for the PAMPA and Caco-2 datasets.

PeptiVerse (Nature Communications 2026) reports that ChemBERTa embeddings
outperform PeptideCLM on both PAMPA (Spearman 0.69 vs 0.59) and Caco-2
(0.80 vs 0.75), and that embedding quality dominates architecture. This
script generates the CLS-token embeddings for our two molecular endpoints
so we can test the same question on our data.

Output:
    data/chemberta/chemberta_emb_pampa_mdck.npz   keys=smiles, emb=[N,384]
    data/chemberta/chemberta_emb_caco2.npz

Model: DeepChem/ChemBERTa-77M-MLM. NOTE: the "77M" refers to the 77M-SMILES
pretraining corpus, NOT the parameter count — the model itself is a small
RoBERTa (~3.4M params, hidden 384). CLS token (last_hidden_state[:,0,:]) is
used; the checkpoint's lm_head / pooler are not part of the base AutoModel
and are irrelevant to these embeddings.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "deepchem/ChemBERTa-77M-MLM"
EMBED_DIM = 384  # NOTE: ChemBERTa-77M hidden size is 384 (not 768)

REPO = Path(__file__).resolve().parent
DATA = REPO / "data"
OUT_DIR = DATA / "chemberta"


def load_model():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    print(f"[chemberta] loaded {MODEL_NAME} in {time.time() - t0:.1f}s", flush=True)
    return tok, model


def embed_batch(tok, model, batch: list[str], device: str) -> np.ndarray:
    enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
    return out.last_hidden_state[:, 0, :].detach().cpu().numpy().astype(np.float32)


def embed_all(smiles: list[str], tok, model, batch_size: int = 32,
              device: str = "cpu") -> np.ndarray:
    out = np.zeros((len(smiles), EMBED_DIM), dtype=np.float32)
    t0 = time.time()
    for i in range(0, len(smiles), batch_size):
        batch = smiles[i:i + batch_size]
        out[i:i + len(batch)] = embed_batch(tok, model, batch, device)
        done = min(i + batch_size, len(smiles))
        if (i // batch_size) % 25 == 0:
            rate = done / max(time.time() - t0, 1e-9)
            eta = (len(smiles) - done) / max(rate, 1e-9)
            print(f"[chemberta] {done}/{len(smiles)}  ({rate:.1f}/s, ETA {eta/60:.1f} min)",
                  flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoints", default="pampa_mdck,caco2")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    tok, model = load_model()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for ep in [e.strip() for e in args.endpoints.split(",") if e.strip()]:
        csv = DATA / f"pepadmet_{ep}.csv"
        if not csv.exists():
            print(f"[chemberta] missing {csv}; skipping", flush=True)
            continue
        df = pd.read_csv(csv)
        smiles = df["smiles"].astype(str).tolist()
        print(f"[chemberta] {ep}: embedding {len(smiles)} SMILES", flush=True)
        emb = embed_all(smiles, tok, model, args.batch_size)
        out_path = OUT_DIR / f"chemberta_emb_{ep}.npz"
        np.savez_compressed(out_path, keys=np.array(smiles), emb=emb)
        print(f"[chemberta] saved {out_path} shape={emb.shape}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
