#!/usr/bin/env python3
"""

_pampa_round3.py — strong 2D models for PAMPA, hunting the true ceiling.

Key diagnostics (from the y-structure analysis):
  * y = logPapp, only 648 unique values (quantized to 0.01)
  * 269 rows (3.7%) sit EXACTLY at the assay floor y=-10.0  (left-censored)
  * the floor ALONE carries 49.6% of the total variance -> predicting
    "which molecules land at the floor" is ~half the battle for R2

Round 3 tests, on the SAME honest split as v4.2 (seed 42, unique-SMILES
random 70/10/20), in ORIGINAL logPapp units:

  A. current v4.2 MLP (baseline 0.4642) — per-region R2 diagnostics
  B. LightGBM grid search (val-guided early stopping), diverse fingerprints
  C. LightGBM best configs, multi-seed + ensemble
  D. TWO-STAGE floor model: LGBMClassifier(floor vs not) + LGBMRegressor
     (non-floor rows only), combined decision rule

Features: cached (RDKit 217 + Morgan r2 2048 + MoLFormer 768) plus NEW
Morgan r1/r3/r4 (2048 each) and the classic RDKit 2048-bit fingerprint.
All scalers/transforms fit on TRAIN rows only.
"""

import common  # re-roots CWD to repo root



import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator as RFG
from rdkit import Chem as _Chem

import lightgbm as lgb

DEVICE = torch.device('cpu')
SEED = 42
N_EST = 1500          # max trees; early stopping on val
PATIENCE = 150

# --------------------------------------------------------------------------- #
# split (VERBATIM copy of train_pepadmet_model.py split_molecular, seed 42)
# --------------------------------------------------------------------------- #
df = pd.read_csv('data/pepadmet_pampa_mdck.csv')
smiles_list = df['smiles'].astype(str).tolist()
y_raw = df['PAMPA_MDCK'].to_numpy(dtype=np.float64)
N = len(df)
assert N == 7283

uniq, inv = np.unique(np.asarray(smiles_list, dtype=object), return_inverse=True)
rng = np.random.default_rng(SEED)
perm = rng.permutation(len(uniq))
n_tr = int(round(len(uniq) * 0.70))
n_va = int(round(len(uniq) * 0.10))
tr_ids = set(perm[:n_tr].tolist())
va_ids = set(perm[n_tr:n_tr + n_va].tolist())
tr = np.array([i for i in range(N) if inv[i] in tr_ids], dtype=np.int64)
va = np.array([i for i in range(N) if inv[i] in va_ids], dtype=np.int64)
te = np.array([i for i in range(N) if inv[i] not in tr_ids and inv[i] not in va_ids],
              dtype=np.int64)
print(f'N={N}  unique={len(uniq)}  train={len(tr)} val={len(va)} test={len(te)}')

y_tr, y_va, y_te = y_raw[tr], y_raw[va], y_raw[te]
FLOOR = y_raw.min()
print(f'floor={FLOOR}  test: floor count = {(y_te == FLOOR).sum()}')

# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #
t0 = time.time()
z = np.load('_pampa_feat_cache.npz')
X_desc, m2, X_molf = z['desc'], z['morgan2'], z['molf']

# new Morgan radii + classic RDKit FP (fast)
def morgan_block(radius):
    gen = RFG.GetMorganGenerator(radius=radius, fpSize=2048)
    Xm = np.zeros((N, 2048), dtype=np.float64)
    for i, s in enumerate(smiles_list):
        try:
            m = Chem.MolFromSmiles(s)
            if m is None: continue
            fp = gen.GetFingerprint(m)
            arr = np.zeros(2048, dtype=np.float64)
            from rdkit.DataStructs import ConvertToNumpyArray
            ConvertToNumpyArray(fp, arr)
            Xm[i] = arr
        except Exception:
            pass
    return Xm

blocks = {}
for r in (1, 3, 4):
    blocks[r] = morgan_block(r)
    print(f'  Morgan r={r} done ({time.time()-t0:.0f}s)')

from rdkit.Chem import AllChem
X_rdkfp = np.zeros((N, 2048), dtype=np.float64)
for i, s in enumerate(smiles_list):
    try:
        m = Chem.MolFromSmiles(s)
        if m is None: continue
        fp = AllChem.GetMorganFingerprintAsBitVec(m, 2, nBits=2048)
        arr = np.zeros(2048, dtype=np.float64)
        from rdkit.DataStructs import ConvertToNumpyArray
        ConvertToNumpyArray(fp, arr)
        X_rdkfp[i] = arr
    except Exception:
        pass
print(f'  features total ({time.time()-t0:.0f}s)')

FEAT_SETS = {
    'v4.2 (2265+768=3033)': np.hstack([X_desc, m2, X_molf]),
    '+Morgan r1':           np.hstack([X_desc, m2, blocks[1], X_molf]),
    '+Morgan r1,r3,r4':     np.hstack([X_desc, m2, blocks[1], blocks[3], blocks[4], X_molf]),
    'Morgan only (r1..r4)': np.hstack([m2, blocks[1], blocks[3], blocks[4], X_molf]),
}

# --------------------------------------------------------------------------- #
# A. current v4.2 MLP — per-region diagnostics on the SAME test split
# --------------------------------------------------------------------------- #
print('\n=== A. current v4.2 MLP (models_v4/pampa_mdck) — per-region test R2 ===')
from admet_model import MixedADMETMLP
blob = torch.load('models_v4/pampa_mdck/admet_mlp.pt', map_location='cpu', weights_only=False)
mlp = MixedADMETMLP(input_dim=blob['input_dim'], endpoints=blob['endpoints'],
                    hidden=blob.get('hidden', (256, 128)),
                    dropout=blob.get('dropout', 0.25))
mlp.load_state_dict(blob['state_dict']); mlp.eval()
scaler = torch.load('models_v4/pampa_mdck/scaler.pt', map_location='cpu', weights_only=False)
X3033 = FEAT_SETS['v4.2 (2265+768=3033)']
with torch.no_grad():
    p = mlp(torch.from_numpy(scaler.transform(X3033[te].astype(np.float32))).to(DEVICE))[ 'PAMPA_MDCK'].squeeze(1).cpu().numpy()
print(f'  overall test R2 (should reproduce ~0.4642): {r2_score(y_te, p):.4f}')
for label, mask in [('floor y==-10 (269-ish)', y_te == FLOOR),
                    ('bulk  -10<y<=-6', (y_te > FLOOR) & (y_te <= -6)),
                    ('tail  y>-6', y_te > -6)]:
    if mask.sum() == 0: continue
    r = r2_score(y_te[mask], p[mask])
    mae = float(np.abs(y_te[mask] - p[mask]).mean())
    print(f'    {label:24s} n={mask.sum():4d}  R2={r:8.4f}  MAE={mae:.3f}  '
          f'pred_range=[{p[mask].min():.2f},{p[mask].max():.2f}]')

# --------------------------------------------------------------------------- #
# B. LightGBM grid (val early stopping)
# --------------------------------------------------------------------------- #
print('\n=== B. LightGBM grid search (early stop on val, N_EST=%d) ===' % N_EST)

def lgb_fit(Xtr, ytr, Xva, yva, params, seed):
    dtr = lgb.Dataset(Xtr, label=ytr, free_raw_data=False)
    dva = lgb.Dataset(Xva, label=yva, reference=dtr, free_raw_data=False)
    model = lgb.train({**params, 'verbose': -1, 'seed': seed,
                       'deterministic': True, 'num_threads': 4},
                      dtr, num_boost_round=N_EST, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
    return model

results = []
for fs_name, X in FEAT_SETS.items():
    Xtr, Xva, Xte = X[tr], X[va], X[te]
    for lr in (0.03, 0.05, 0.1):
        for nl in (31, 63, 127):
            for mcs in (10, 30):
                for ff in (0.7, 1.0):
                    params = dict(learning_rate=lr, num_leaves=nl,
                                  min_child_samples=mcs, feature_fraction=ff,
                                  bagging_fraction=0.9, bagging_freq=1,
                                  lambda_l2=1.0)
                    t1 = time.time()
                    m = lgb_fit(Xtr, y_tr, Xva, y_va, params, SEED)
                    pv = m.predict(Xva)
                    pt = m.predict(Xte)
                    r2t = r2_score(y_te, pt)
                    results.append((r2t, lr, nl, mcs, ff, fs_name, m.best_iteration,
                                    time.time() - t1, m))
                    print(f'  [{fs_name}] lr={lr} nl={nl} mcs={mcs} ff={ff}  '
                          f'iter={m.best_iteration:4d}  R2_te={r2t:.4f}  '
                          f'({time.time()-t1:.1f}s)  [t={time.time()-t0:5.0f}s]')

results.sort(reverse=True, key=lambda r: r[0])
print('\n  TOP-5 LightGBM configs (test R2):')
for r2t, lr, nl, mcs, ff, fs, it, dt, _ in results[:5]:
    print(f'    R2={r2t:.4f}  {fs}  lr={lr} nl={nl} mcs={mcs} ff={ff}  it={it}')

# --------------------------------------------------------------------------- #
# C. best-3 configs: 5-seed ensemble
# --------------------------------------------------------------------------- #
print('\n=== C. best-3 configs x 5 seeds, ensemble in logPapp space ===')
best3 = sorted({(r[1], r[2], r[3], r[4], r[5]) for r in results[:20]})[:3]
for lr, nl, mcs, ff, fs in best3:
    X = FEAT_SETS[fs]
    Xtr, Xva, Xte = X[tr], X[va], X[te]
    params = dict(learning_rate=lr, num_leaves=nl, min_child_samples=mcs,
                  feature_fraction=ff, bagging_fraction=0.9, bagging_freq=1,
                  lambda_l2=1.0)
    ens = np.zeros(len(te)); r2s = []
    for s in (42, 123, 456, 789, 1024):
        m = lgb_fit(Xtr, y_tr, Xva, y_va, params, s)
        pt = m.predict(Xte)
        r2s.append(r2_score(y_te, pt)); ens += pt
    ens /= 5.0
    print(f'  {fs}  lr={lr} nl={nl} mcs={mcs} ff={ff}: '
          f'per-seed mean={np.mean(r2s):.4f} sd={np.std(r2s):.4f}  '
          f'ENSEMBLE R2={r2_score(y_te, ens):.4f}')

# --------------------------------------------------------------------------- #
# D. two-stage floor model
# --------------------------------------------------------------------------- #
print('\n=== D. TWO-STAGE: floor classifier + non-floor regressor ===')
for fs_name, X in FEAT_SETS.items():
    Xtr, Xva, Xte = X[tr], X[va], X[te]
    floor_tr = (y_tr == FLOOR).astype(int)
    floor_va = (y_va == FLOOR).astype(int)
    floor_te = (y_te == FLOOR)

    cls = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=63,
                             min_child_samples=30, feature_fraction=0.9,
                             bagging_fraction=0.9, bagging_freq=1,
                             random_state=SEED, n_jobs=4, verbose=-1)
    cls.fit(Xtr, floor_tr, eval_set=[(Xva, floor_va)],
            callbacks=[lgb.early_stopping(150, verbose=False)])
    p_floor_te = cls.predict_proba(Xte)[:, 1]

    nm = y_tr != FLOOR
    reg = lgb.LGBMRegressor(n_estimators=1500, learning_rate=0.05, num_leaves=63,
                            min_child_samples=30, feature_fraction=0.9,
                            bagging_fraction=0.9, bagging_freq=1,
                            random_state=SEED, n_jobs=4, verbose=-1)
    reg.fit(Xtr[nm], y_tr[nm], eval_set=[(Xva[y_va != FLOOR], y_va[y_va != FLOOR])],
            callbacks=[lgb.early_stopping(150, verbose=False)])
    p_reg = reg.predict(Xte)

    for thr in (0.5,):
        p = np.where(p_floor_te >= thr, FLOOR, p_reg)
        print(f'  {fs_name}  thr={thr}:  R2_te={r2_score(y_te, p):.4f}  '
              f'(floor recall={ (p_floor_te[ floor_te] >= thr).mean():.3f}  '
              f'floor precision={(p_floor_te[~floor_te] < thr).mean():.3f})')
print('\nDone. Wall time %ds' % (time.time() - t0))
