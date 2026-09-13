#!/usr/bin/env python3
"""Precompute frozen MoLFormer-XL embeddings for the molecular endpoints.

v4.2 "new weapon" for Caco-2 and PAMPA/MDCK: the frozen MoLFormer-XL
(IBM, 60M params, hidden 768) CLS token for each unique SMILES, saved as a
committed npz next to the data so retraining and cached inference never
need the model.

Model: ibm-research/MoLFormer-XL-both-10pct (trust_remote_code; the custom
modeling files are part of the public HF repo).

Outputs (data/molformer/):
    molformer_emb_caco2.npz    : keys = unique SMILES (file order), emb = [N,768] float32
    molformer_emb_pampa_mdck.npz
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from endpoint_config import ENDPOINT_BY_NAME

MOLEFORMER_MODEL = "ibm-research/MoLFormer-XL-both-10pct"
EMBED_DIM = 768

REPO = Path(__file__).resolve().parent
DATA = REPO / "data"
OUT_DIR = DATA / "molformer"

SMILES_COLUMN = "smiles"


def load_model():
    import torch
    from transformers import AutoModel, AutoTokenizer

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MOLEFORMER_MODEL, trust_remote_code=True)
    model = AutoModel.from_pretrained(MOLEFORMER_MODEL, trust_remote_code=True)
    model.eval()
    print(f"[molformer] loaded {MOLEFORMER_MODEL} in {time.time() - t0:.1f}s", flush=True)
    return tok, model


def embed_batch(tok, model, batch: list[str]) -> np.ndarray:
    import torch
    enc = tok(batch, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        out = model(**enc)
    return out.last_hidden_state[:, 0, :].detach().cpu().numpy()


def embed_all(smiles: list[str], tok, model, batch_size: int = 64) -> np.ndarray:
    out = np.zeros((len(smiles), EMBED_DIM), dtype=np.float32)
    t0 = time.time()
    for i in range(0, len(smiles), batch_size):
        batch = smiles[i : i + batch_size]
        out[i : i + len(batch)] = embed_batch(tok, model, batch)
        if (i // batch_size) % 20 == 0:
            done = min(i + batch_size, len(smiles))
            rate = done / max(time.time() - t0, 1e-9)
            eta = (len(smiles) - done) / max(rate, 1e-9)
            print(f"[molformer] {done}/{len(smiles)}  ({rate:.1f}/s, ETA {eta/60:.1f} min)", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoints", default="caco2,pampa_mdck",
                    help="batch mode: endpoint slugs to embed from prepared CSVs")
    ap.add_argument("--smiles-file", default=None,
                    help="ad-hoc mode: file with one SMILES per line")
    ap.add_argument("--out", default=None,
                    help="ad-hoc mode: output .npz path (keys + emb)")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    tok, model = load_model()

    # ---- ad-hoc mode: embed an arbitrary SMILES list (predictor fallback) ----
    if args.smiles_file:
        if not args.out:
            print("[molformer] --out is required with --smiles-file", flush=True)
            return 2
        smiles = [ln.strip() for ln in open(args.smiles_file, encoding="utf-8")
                  if ln.strip()]
        print(f"[molformer] ad-hoc: {len(smiles)} SMILES", flush=True)
        emb = embed_all(smiles, tok, model, args.batch_size)
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, keys=np.array(smiles), emb=emb)
        print(f"[molformer] ad-hoc: saved {out_path} (shape {emb.shape})", flush=True)
        return 0

    # ---- batch mode: prepared CSVs -> data/molformer/*.npz ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for slug in [s.strip() for s in args.endpoints.split(",") if s.strip()]:
        csv = DATA / f"pepadmet_{slug}.csv"
        if not csv.exists():
            print(f"[molformer] SKIP {slug}: {csv} missing", flush=True)
            continue
        name = next((e.name for e in ENDPOINT_BY_NAME.values()
                     if e.name.lower().replace(" ", "_") == slug), None)
        df = pd.read_csv(csv)
        # Same order / same universe as prepare_pepadmet_data (which drops rows
        # with NaN SMILES or target): keep rows that survive preparation.
        label_col = name  # prepared CSV writes the label under the endpoint name
        smi_col = SMILES_COLUMN if SMILES_COLUMN in df.columns else None
        if smi_col is None or label_col is None or label_col not in df.columns:
            print(f"[molformer] SKIP {slug}: missing columns "
                  f"(smiles={SMILES_COLUMN in df.columns}, label={label_col})", flush=True)
            continue
        df = df.dropna(subset=[smi_col]).dropna(subset=[label_col])
        df = df[df[smi_col].astype(str).str.strip() != ""]
        smiles = df[smi_col].astype(str).tolist()
        print(f"[molformer] {slug}: {len(smiles)} rows -> embedding", flush=True)

        emb = embed_all(smiles, tok, model, args.batch_size)
        np.savez_compressed(OUT_DIR / f"molformer_emb_{slug}.npz", keys=np.array(smiles), emb=emb)
        (OUT_DIR / f"molformer_emb_{slug}.meta.json").write_text(json.dumps(
            {"model": MOLEFORMER_MODEL, "dim": EMBED_DIM, "n": len(smiles),
             "pooling": "cls", "dtype": "float32",
             "created": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2), encoding="utf-8")
        print(f"[molformer] {slug}: saved {OUT_DIR / f'molformer_emb_{slug}.npz'} "
              f"(shape {emb.shape})", flush=True)
    print("[molformer] all done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
