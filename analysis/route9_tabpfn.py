#!/usr/bin/env python3
"""Route 9: TabPFN v2 (foundation tabular model) on PAMPA.

Fair comparison with the committed v4.2 LightGBM baseline (0.4642):
  - SPLIT: the ONE canonical leakage-controlled split from analysis/common.py
    (common.split_smiles, seed 42, unique-SMILES 70/10/20). This is the SAME
    test set the baseline 0.4642 was measured on, so the numbers are directly
    comparable. (A prior quick sweep used a slightly different split
    int(0.7n) vs round(0.7n); this repo version corrects that.)
  - SEEDS: 42/123/7 drive TabPFN's internal feature subsampling / estimator
    ensemble only -- the data split is held fixed (same as the baseline
    protocol, which is a single canonical split).
  - DEVICE: CUDA (TabPFN runs natively on GPU; unlike KPGT/DGL which has no
    Windows CUDA build).

Feature sets:
  desc            (217 descriptors)            -- within the v2 500-feat limit
  desc+morgan     (217+2048, ignore limits)    -- v2 feature-subsamples 2265
  desc+morgan+molf(217+2048+768, ignore limits)
"""
import sys, os, time, json
os.chdir(r'C:/Users/c00jsw00/openclaw-peptide-admet')
sys.path.insert(0, 'analysis')
import common
import numpy as np, pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from scipy.stats import spearmanr
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
import torch

print('cuda:', torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else '', flush=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def make(seed, ignore=False):
    return TabPFNRegressor(
        model_path=TabPFNRegressor.create_default_for_version(ModelVersion.V2).model_path,
        n_estimators=4, device=DEVICE, random_state=seed,
        ignore_pretraining_limits=ignore,
    )

smiles, y = common.load_data()
z = np.load(common.FEAT_CACHE)
desc, morgan2, molf = z['desc'], z['morgan2'], z['molf']
N = len(smiles)

# canonical leakage-controlled split (seed 42) -- SAME as the v4.2 baseline
tr, va, te = common.split_smiles(smiles, seed=common.SEED)
print(f'rows={N} split train/val/test={len(tr)}/{len(va)}/{len(te)} '
      f'(common.split_smiles, seed {common.SEED})', flush=True)

FLOOR = common.FLOOR
SEEDS = [42, 123, 7]
feats = {
    'desc': (desc, False),
    'desc+morgan': (np.hstack([desc, morgan2]), True),
    'desc+morgan+molf': (np.hstack([desc, morgan2, molf]), True),
}

results = {}
for fname, (X, ign) in feats.items():
    r2s, r2nfs, sps, maes = [], [], [], []
    t0 = time.time()
    for seed in SEEDS:
        m = make(seed, ign)
        m.fit(X[tr], y[tr])
        pred = m.predict(X[te])
        y_te = y[te]
        nf = y_te > FLOOR + 1e-6
        r2s.append(r2_score(y_te, pred))
        r2nfs.append(r2_score(y_te[nf], pred[nf]))
        sps.append(spearmanr(y_te, pred)[0])
        maes.append(mean_absolute_error(y_te, pred))
        print(f'  {fname} seed={seed}: R2_all={r2s[-1]:.4f} R2_nf={r2nfs[-1]:.4f} '
              f'sp={sps[-1]:.4f} mae={maes[-1]:.4f}', flush=True)
    results[fname] = {
        'r2_all': round(float(np.mean(r2s)), 4),
        'r2_all_std': round(float(np.std(r2s)), 4),
        'r2_nonfloor': round(float(np.mean(r2nfs)), 4),
        'spearman': round(float(np.mean(sps)), 4),
        'mae': round(float(np.mean(maes)), 4),
        'seeds': {str(s): round(float(r2s[i]), 4) for i, s in enumerate(SEEDS)},
    }
    print(f'{fname}: R2_all={np.mean(r2s):.4f}+/-{np.std(r2s):.4f} '
          f'R2_nf={np.mean(r2nfs):.4f} ({time.time()-t0:.0f}s)', flush=True)

print('=== SUMMARY (TabPFN v2, PAMPA, common.split_smiles seed42, 3 seeds) ===')
for k, v in results.items():
    print(f'  {k:20s} R2_all={v["r2_all"]:.4f}+/-{v["r2_all_std"]:.4f} '
          f'R2_nonfloor={v["r2_nonfloor"]:.4f} sp={v["spearman"]:.4f}', flush=True)
print('  LightGBM baseline (v4.2, same split): R2_all=0.4642  R2_nonfloor=0.6317', flush=True)

out = 'analysis/route9_tabpfn_results.json'
with open(out, 'w') as f:
    json.dump({'device': str(DEVICE), 'split': f'common.split_smiles seed={common.SEED}',
               'n_train': int(len(tr)), 'n_val': int(len(va)), 'n_test': int(len(te)),
               'baseline_lightgbm': {'r2_all': 0.4642, 'r2_nonfloor': 0.6317},
               'results': results}, f, indent=2, default=float)
print('Wrote', out, flush=True)
print('DONE', flush=True)
