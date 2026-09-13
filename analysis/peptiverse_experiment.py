#!/usr/bin/env python3
"""
PeptiVerse raw-data experiment: train on the PeptiVerse dataset itself
(HuggingFace ChatterjeeLab/PeptiVerse_data) and ask whether R2 can exceed 0.7.

Dataset (permeability_pampa / permeability_caco2, __chemberta_with_embeddings
configs, downloaded to data/peptiverse/):
  * 'sequence' column is actually SMILES (cyclo/modified peptides), not AA.
  * 'label' is the log-space permeability target (logPapp / log Papp_nM).
  * 'embedding' is a PRECOMPUTED 384-dim ChemBERTa-77M-MLM vector (the exact
    embedding the paper reports as its best for PAMPA, Spearman rho=0.69).
  * The release ships only train + val (no held-out test split).
  * The PAMPA label is ALSO left-censored at -10.0000: 240 rows (3.5%) for
    PAMPA, 15 rows (2.5%) for Caco-2 -- it is NOT uncensored data.

Two splits are evaluated per endpoint:
  (a) THEIR split   : train -> val, to directly reproduce the paper's rho.
  (b) OUR split     : unique-SMILES 70/10/20 (common.split_smiles, seed 42),
                      the same leakage-controlled protocol as the pipeline,
                      for an honest generalization number + oracle ceiling.

Feature configurations (same verbatim training loop as the pipeline,
train_endpoint_model, Huber d=1.0, 3 seeds):
  E1  ChemBERTa alone          ( 384)  <- the paper's reported best
  E2  ChemBERTa + RDKit/Morgan (2649)
  E3  RDKit/Morgan alone       (2265)  <- reference

Reported per run: test/val R2 (overall + non-floor subset), Spearman rho
(overall + non-floor -- the paper's metric), RMSE, MAE.  The oracle ceiling
(non-floor perfect, floor -> global mean) is reported per endpoint to state
the mathematical upper bound on R2.

Results -> analysis/peptiverse_results.json
Usage:  python analysis/peptiverse_experiment.py [pampa|caco2|both]
"""
import json
import os
import sys
import time

import common  # re-roots CWD to repo root, adds repo root to sys.path

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from chemberta_retrain import build_feat_cache  # RDKit 217 + Morgan r2 (2265)
from train_pepadmet_model import train_endpoint_model, predict_mixed

SEEDS = (42, 123, 7)
FLOOR = -10.0000

ENDPOINTS = {
    'pampa': dict(name='PAMPA_MDCK',
                  tr='data/peptiverse/pampa_train.parquet',
                  va='data/peptiverse/pampa_val.parquet',
                  feat_cache='_pv_pampa_feat_cache.npz',
                  out='analysis/peptiverse_results.json'),
    'caco2': dict(name='Caco2',
                  tr='data/peptiverse/caco2_train.parquet',
                  va='data/peptiverse/caco2_val.parquet',
                  feat_cache='_pv_caco2_feat_cache.npz',
                  out='analysis/peptiverse_results.json'),
}


def load_ep(cfg):
    """Return (smiles, y, chem, their_tr, their_va) from the two parquets."""
    dtr = pd.read_parquet(cfg['tr'])
    dva = pd.read_parquet(cfg['va'])
    df = pd.concat([dtr, dva], ignore_index=True)
    smiles = df['sequence'].astype(str).tolist()
    y = df['label'].to_numpy(dtype=np.float32)
    chem = np.stack([np.asarray(e, dtype=np.float32) for e in df['embedding']])
    ntr = len(dtr)
    their_tr = np.arange(0, ntr, dtype=np.int64)
    their_va = np.arange(ntr, len(df), dtype=np.int64)
    if chem.shape != (len(df), 384):
        raise ValueError(f'{cfg["name"]}: embedding shape {chem.shape}')
    return smiles, y, chem, their_tr, their_va


def metrics(y_true, pred, floor=FLOOR):
    """R2 (overall + non-floor), Spearman (overall + non-floor), RMSE, MAE."""
    nf = y_true > floor + 1e-6
    return dict(
        r2_all=round(float(r2_score(y_true, pred)), 4),
        r2_nonfloor=round(float(r2_score(y_true[nf], pred[nf])), 4),
        spearman_all=round(float(spearmanr(y_true, pred)[0]), 4),
        spearman_nonfloor=round(float(spearmanr(y_true[nf], pred[nf])[0]), 4),
        rmse=round(float(np.sqrt(((y_true - pred) ** 2).mean())), 4),
        mae=round(float(np.mean(np.abs(y_true - pred))), 4),
    )


def run_split(name, X, y, tr, va, te, Xs_all, tag, seed):
    scaler = StandardScaler().fit(Xs_all[tr])
    Xs = scaler.transform(Xs_all).astype(np.float32)
    model, device, _ = train_endpoint_model(
        name, Xs, y, tr, va, epochs=80, seed=seed, hidden=(256, 128),
        regression_loss='huber')
    pred = predict_mixed(model, Xs[te])[name]
    m = metrics(y[te], pred)
    print(f'  [{tag}] seed={seed}: R2={m["r2_all"]:.4f} (nf {m["r2_nonfloor"]:.4f}) '
          f'rho={m["spearman_all"]:.4f} (nf {m["spearman_nonfloor"]:.4f}) '
          f'RMSE={m["rmse"]:.4f}', flush=True)
    return m


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else 'both'
    eps = [which] if which in ENDPOINTS else list(ENDPOINTS)
    results = {}
    for which in eps:
        cfg = ENDPOINTS[which]
        name = cfg['name']
        print(f'===== {name.upper()} (PeptiVerse raw data) =====', flush=True)
        smiles, y, chem, their_tr, their_va = load_ep(cfg)
        n = len(y)
        floor_rows = int((y <= FLOOR + 1e-6).sum())
        ss = float(((y - y.mean()) ** 2).sum())
        ssf = float(((y[y <= FLOOR + 1e-6] - y.mean()) ** 2).sum())
        ceiling = 1.0 - ssf / ss
        print(f'  N={n}  floor rows={floor_rows} ({100 * floor_rows / n:.1f}%)  '
              f'floor SS share={100 * ssf / ss:.1f}%  oracle ceiling R2={ceiling:.4f}',
              flush=True)

        # 2D features (RDKit 217 + Morgan r2 2048 = 2265), row-aligned
        if not os.path.exists(cfg['feat_cache']):
            print(f'  no {cfg["feat_cache"]} -- building (~5-10 min)...', flush=True)
            build_feat_cache(cfg['feat_cache'], smiles)
        zc = np.load(cfg['feat_cache'], allow_pickle=False)
        Xmol = np.hstack([zc['desc'], zc['morgan2']]).astype(np.float32)
        print(f'  features: 2D {Xmol.shape}, ChemBERTa {chem.shape}', flush=True)

        configs = {
            'E1_chem':        chem.astype(np.float32),
            'E2_chem_2d':     np.hstack([chem, Xmol]).astype(np.float32),
            'E3_2d':          Xmol,
        }

        # (b) OUR leakage-controlled split
        tr, va, te = common.split_smiles(smiles)
        print(f'  OUR split: train={len(tr)} val={len(va)} test={len(te)} '
              f'test floor rows={int((y[te] <= FLOOR + 1e-6).sum())}', flush=True)
        our = {}
        for cfg_name, X in configs.items():
            per_seed = []
            for seed in SEEDS:
                t0 = time.time()
                m = run_split(name, X, y, tr, va, te, X, f'{cfg_name}/ours', seed)
                m['seconds'] = round(time.time() - t0, 1)
                per_seed.append(m)
            r2s = [p['r2_all'] for p in per_seed]
            r2nf = [p['r2_nonfloor'] for p in per_seed]
            sp = [p['spearman_all'] for p in per_seed]
            our[cfg_name] = dict(
                r2_mean=round(float(np.mean(r2s)), 4),
                r2_std=round(float(np.std(r2s, ddof=1)), 4),
                r2_nonfloor_mean=round(float(np.mean(r2nf)), 4),
                spearman_mean=round(float(np.mean(sp)), 4),
                spearman_std=round(float(np.std(sp, ddof=1)), 4),
                per_seed=per_seed)
            print(f'    {cfg_name}: R2={np.mean(r2s):.4f} +/- {np.std(r2s, ddof=1):.4f} '
                  f'(non-floor {np.mean(r2nf):.4f})  '
                  f'rho={np.mean(sp):.4f} +/- {np.std(sp, ddof=1):.4f}', flush=True)

        # (a) THEIR split (train -> val) to reproduce the paper's rho
        their = {}
        for cfg_name in ('E1_chem', 'E2_chem_2d'):
            X = configs[cfg_name]
            m = run_split(name, X, y, their_tr, their_va, their_va, X,
                          f'{cfg_name}/theirs', seed=42)
            m['n_val'] = int(len(their_va))
            their[cfg_name] = m
            print(f'    THEIR split {cfg_name}: val R2={m["r2_all"]:.4f} '
                  f'(nf {m["r2_nonfloor"]:.4f})  val rho={m["spearman_all"]:.4f} '
                  f'(nf {m["spearman_nonfloor"]:.4f})  n_val={m["n_val"]}', flush=True)

        results[name] = dict(n=n, floor_rows=floor_rows,
                             floor_ss_share=round(100 * ssf / ss, 1),
                             oracle_ceiling_r2=round(ceiling, 4),
                             their_train_floor_pct=round(
                                 100 * (y[their_tr] <= FLOOR + 1e-6).mean(), 1),
                             their_val_floor_pct=round(
                                 100 * (y[their_va] <= FLOOR + 1e-6).mean(), 1),
                             our_split=our, their_split=their)
        print(flush=True)

    out = ENDPOINTS['pampa']['out']
    with open(out, 'w') as f:
        json.dump({'source': 'ChatterjeeLab/PeptiVerse_data',
                   'run': time.strftime('%Y-%m-%d %H:%M:%S'),
                   'endpoints': results}, f, indent=2)
    print(f'Wrote {out}')
    print('Done.')


if __name__ == '__main__':
    main()
