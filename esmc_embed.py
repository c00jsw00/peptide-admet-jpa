#!/usr/bin/env python3
"""
esmc_embed.py
=============

ESMC-600M (Biohub, ESM Cambrian) sequence embeddings for the two
sequence-modality endpoints of the v4.1 PepADMET pipeline (Hemolysis,
Half_life).

ESMC is a protein language model: it consumes one-letter 20-AA peptide
sequences and produces a 1152-dim mean-pooled embedding per sequence.
Caco-2 / PAMPA_MDCK are molecular-modality (non-standard peptidomimetic
residue lists, ~0.2% standard AA) and are NOT embedded here - they stay on
the RDKit molecular feature path.

The Biohub `esm` package (git main) requires Python >= 3.12 and registers
the `esmc` architecture into transformers.  This script therefore runs in
the dedicated env:

    .venv-esmc/Scripts/python.exe esmc_embed.py
    .venv-esmc/Scripts/python.exe esmc_embed.py --sequences-file s.txt --out e.npz

Outputs (committed, so training is reproducible without the 3.12 env)
---------------------------------------------------------------------
    data/esmc/esmc_emb_hemolysis.npz   (emb (N,1152) float32, sequences, meta)
    data/esmc/esmc_emb_half_life.npz

Row order is the prepared-CSV row order (data/pepadmet_<slug>.csv,
'sequence' column).  All sequences in the prepared CSVs are 100% standard
20-AA, so no row is dropped.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

DEFAULTS = [
    # (endpoint slug, prepared csv, output npz)
    ('hemolysis', ROOT / 'data/pepadmet_hemolysis.csv',
     ROOT / 'data/esmc/esmc_emb_hemolysis.npz'),
    ('half_life', ROOT / 'data/pepadmet_half_life.csv',
     ROOT / 'data/esmc/esmc_emb_half_life.npz'),
]


def embed_sequences(sequences, batch=64, log_every=64):
    import torch
    import esm  # noqa: F401  -- registers the 'esmc' arch in transformers
    from transformers import AutoModel, AutoTokenizer

    t0 = time.time()
    model = AutoModel.from_pretrained('biohub/ESMC-600M')
    model.eval()
    tok = AutoTokenizer.from_pretrained('biohub/ESMC-600M')
    n_params = sum(p.numel() for p in model.parameters())
    print(f'ESMC-600M loaded in {time.time() - t0:.0f}s, '
          f'params={n_params:,}', flush=True)

    embs, done, t_last = [], 0, time.time()
    for i in range(0, len(sequences), batch):
        chunk = sequences[i:i + batch]
        b = tok(list(chunk), padding=True, truncation=True,
                max_length=512, return_tensors='pt')
        with torch.no_grad():
            out = model(**b)
        last = out.last_hidden_state
        m = b['attention_mask'].unsqueeze(-1).float()
        emb = (last * m).sum(1) / m.sum(1).clamp(min=1.0)
        embs.append(emb.cpu().numpy().astype(np.float32))
        done += len(chunk)
        if done % log_every < batch or done == len(sequences):
            el = time.time() - t_last
            print(f'  {done}/{len(sequences)}  '
                  f'{done / max(time.time() - t0, 1e-6):.1f} seq/s',
                  flush=True)
            t_last = time.time()
    return np.concatenate(embs, axis=0), time.time() - t0


def save_npz(path, emb, sequences):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        'model': 'biohub/ESMC-600M',
        'hidden_dim': int(emb.shape[1]),
        'pooling': 'mean (attention-mask weighted)',
        'n': int(emb.shape[0]),
        'created': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }
    np.savez_compressed(path, emb=emb,
                        sequences=np.array(sequences, dtype=object),
                        meta=json.dumps(meta).encode('utf-8'))
    print(f'saved {path}  shape={emb.shape}  ({time.strftime("%H:%M:%S")})')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sequences-file', type=str, default=None,
                    help='newline-separated sequence file (ad-hoc mode)')
    ap.add_argument('--out', type=str, default=None,
                    help='output npz (ad-hoc mode)')
    args = ap.parse_args()

    if args.sequences_file:
        if not args.out:
            raise SystemExit('--out required with --sequences-file')
        seqs = [ln.strip() for ln in
                Path(args.sequences_file).read_text().splitlines() if ln.strip()]
        t0 = time.time()
        emb, el = embed_sequences(seqs)
        save_npz(args.out, emb, seqs)
        print(f'ad-hoc done: {len(seqs)} seq in {el:.0f}s')
        return

    for slug, csv, out in DEFAULTS:
        df = pd.read_csv(csv)
        seqs = df['sequence'].astype(str).tolist()
        print(f'[{slug}] n={len(seqs)} embedding...')
        emb, el = embed_sequences(seqs)
        assert emb.shape == (len(seqs), int(emb.shape[1]))
        assert np.isfinite(emb).all(), f'{slug}: non-finite embeddings'
        save_npz(out, emb, seqs)
        print(f'[{slug}] done in {el:.0f}s\n')


if __name__ == '__main__':
    main()
