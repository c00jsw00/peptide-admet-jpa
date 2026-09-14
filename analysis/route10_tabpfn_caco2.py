#!/usr/bin/env python3
"""Route 10: TabPFN v2 (foundation tabular model) on Caco-2.

Mirrors analysis/route9_tabpfn.py (PAMPA) exactly, on the Caco-2 endpoint:
  - SPLIT: leakage-controlled unique-SMILES 70/10/20, seed 42, VERBATIM copy
    of train_pepadmet_model.split_molecular (same as the committed caco2
    baseline 0.3909 was measured on).
  - SEEDS: 42/123/7 drive TabPFN's internal estimator ensemble only; the
    data split is held fixed (same as the baseline protocol).
  - DEVICE: CUDA (TabPFN runs natively on GPU).

Feature sets (same 3 as PAMPA route 9):
  desc             (217 RDKit descriptors)  -- within the v2 500-feat limit
  desc+morgan      (2265, ignore_limits)    -- v2 feature-subsamples
  desc+morgan+molf (3033, ignore_limits)

Baseline reference: committed v4.2 caco2 model (R2 0.3909 all, non-floor ~?)
-> results -> analysis/route10_tabpfn_caco2.json
"""
import sys, os, time, json
os.chdir(r'C:/Users/c00jsw00/openclaw-peptide-admet')
sys.path.insert(0, 'analysis')
import common
import numpy as np, pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import spearmanr

# --- Caco-2-specific config (common.py is PAMPA; override without touching it)
CSV = 'data/pepadmet_caco2.csv'
TARGET = 'Caco2'
FEAT_CACHE = '_caco2_feat_cache.npz'
MOLF = 'data/molformer/molformer_emb_caco2.npz'
OUT = 'analysis/route10_tabpfn_caco2.json'

print('cuda check via torch...', flush=True)
import torch
print('cuda:', torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else '', flush=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

def make(seed, ignore=False):
    return TabPFNRegressor(
        model_path=TabPFNRegressor.create_default_for_version(ModelVersion.V2).model_path,
        n_estimators=4, device=DEVICE, random_state=seed,
        ignore_pretraining_limits=ignore,
    )

# --- data ---
df = pd.read_csv(CSV)
smiles = df['smiles'].astype(str).tolist()
y = df[TARGET].to_numpy(dtype=np.float64)
N = len(smiles)
z = np.load(FEAT_CACHE)
desc, morgan2 = z['desc'], z['morgan2']
molf = np.load(MOLF, allow_pickle=True)['emb'].astype(np.float64)
assert len(y) == len(desc) == len(morgan2) == len(molf) == 7429, N

# --- canonical leakage-controlled split (VERBATIM split_molecular, seed 42) ---
SEED = 42
uniq, inv = np.unique(np.asarray(smiles, dtype=object), return_inverse=True)
rng = np.random.default_rng(SEED)
perm = rng.permutation(len(uniq))
n_tr = int(round(len(uniq) * 0.70))
n_va = int(round(len(uniq) * 0.10))
tr_ids = set(perm[:n_tr].tolist())
va_ids = set(perm[n_tr:n_tr + n_va].tolist())
tr = np.array([i for i in range(N) if inv[i] in tr_ids], dtype=np.int64)
va = np.array([i for i in range(N) if inv[i] in va_ids], dtype=np.int64)
te = np.array([i for i in range(N)
               if inv[i] not in tr_ids and inv[i] not in va_ids], dtype=np.int64)
y_te = y[te]
floor_te = y_te <= common.FLOOR + 1e-6
print(f'rows={N} train/val/test={len(tr)}/{len(va)}/{len(te)} '
      f'test floor rows={int(floor_te.sum())} '
      f'(split seed {SEED}, identical to committed baseline)', flush=True)

# sanity: reproduce the committed baseline split size (1490 test rows)
assert len(te) == 1490, f'test split mismatch: {len(te)} != 1490'

# --- baseline reference (committed v4.2 caco2 MLP, loaded and evaluated here) ---
import importlib
from admet_model import MixedADMETMLP
blob = torch.load('models_v4/caco2/admet_mlp.pt', map_location='cpu', weights_only=False)
mlp = MixedADMETMLP(input_dim=blob['input_dim'], endpoints=blob['endpoints'],
                    hidden=blob.get('hidden', (256, 128)), dropout=blob.get('dropout', 0.25))
mlp.load_state_dict(blob['state_dict']); mlp.eval()
sc = torch.load('models_v4/caco2/scaler.pt', map_location='cpu', weights_only=False)
X3033 = np.hstack([desc, morgan2, molf]).astype(np.float32)
with torch.no_grad():
    p_base = mlp(torch.from_numpy(sc.transform(X3033[te])))['Caco2'].squeeze(1).numpy()
nf = ~floor_te
r2_base = r2_score(y_te, p_base)
r2_base_nf = r2_score(y_te[nf], p_base[nf])
print(f'baseline (committed v4.2 caco2 MLP): R2_all={r2_base:.4f} '
      f'R2_nonfloor={r2_base_nf:.4f}', flush=True)

# --- TabPFN runs ---
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
        'r2_nonfloor_std': round(float(np.std(r2nfs)), 4),
        'spearman': round(float(np.mean(sps)), 4),
        'mae': round(float(np.mean(maes)), 4),
        'seeds': {str(s): round(float(r2s[i]), 4) for i, s in enumerate(SEEDS)},
    }
    print(f'{fname}: R2_all={np.mean(r2s):.4f}+/-{np.std(r2s):.4f} '
          f'R2_nf={np.mean(r2nfs):.4f} ({time.time()-t0:.0f}s)', flush=True)

out = {
    'model': 'TabPFN v2 (tabpfn 8.5.0, ModelVersion.V2, n_estimators=4, CUDA)',
    'endpoint': 'Caco2',
    'split': 'unique-SMILES 70/10/20 seed=42 (VERBATIM split_molecular; test n=1490, same as committed baseline)',
    'n_rows': int(N),
    'test_floor_rows': int(floor_te.sum()),
    'baseline_v42_mlp': {'r2_all': round(float(r2_base), 4),
                         'r2_nonfloor': round(float(r2_base_nf), 4)},
    'results': results,
    'run': time.strftime('%Y-%m-%d %H:%M:%S'),
}
with open(OUT, 'w') as f:
    json.dump(out, f, indent=2)
print(f'\nWrote {OUT}')
print('=== SUMMARY (TabPFN v2, Caco-2, canonical split, 3 seeds) ===')
for k, v in results.items():
    print(f'  {k:20s} R2_all={v["r2_all"]:.4f}+/-{v["r2_all_std"]:.4f} '
          f'R2_nf={v["r2_nonfloor"]:.4f} vs baseline {r2_base:.4f} '
          f'(delta {v["r2_all"]-r2_base:+.4f})')
print('Done.')
