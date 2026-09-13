#!/usr/bin/env python3
"""

_pampa_blend.py — final experiment: SOFT posterior-mean blending for the
censored floor (Bayes-optimal for y|x ~ P(f|x)·delta(-10) + (1-P(f|x))·f(y|x)).

pred' = pred - beta * P_floor * (pred + 10)

  beta=0  -> baseline regressor
  beta=1  -> full posterior mean (P-weighted blend toward -10)
  beta->inf + hard P -> two-stage (already shown to LOSE)

beta is selected on VAL (26 floor rows) ONLY, then applied to TEST.
Honest, no test-set selection.
"""

import common  # re-roots CWD to repo root



import warnings
warnings.filterwarnings('ignore')
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.calibration import calibration_curve
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, QED

SEED = 42
FLOOR = -10.0000
N_EST = 1000
PATIENCE = 100

# --------------------------------------------------------------------------- #
# data + verbatim split
# --------------------------------------------------------------------------- #
df = pd.read_csv('data/pepadmet_pampa_mdck.csv')
smiles_list = df['smiles'].astype(str).tolist()
y = df['PAMPA_MDCK'].to_numpy(dtype=np.float64)
N = len(df)
assert N == 7283
uniq, inv = np.unique(np.asarray(smiles_list, dtype=object), return_inverse=True)
rng = np.random.default_rng(SEED)
perm = rng.permutation(len(uniq))
n_tr = int(round(len(uniq) * 0.70)); n_va = int(round(len(uniq) * 0.10))
tr_ids = set(perm[:n_tr].tolist()); va_ids = set(perm[n_tr:n_tr + n_va].tolist())
tr = np.array([i for i in range(N) if inv[i] in tr_ids], dtype=np.int64)
va = np.array([i for i in range(N) if inv[i] in va_ids], dtype=np.int64)
te = np.array([i for i in range(N) if inv[i] not in tr_ids and inv[i] not in va_ids], dtype=np.int64)
y_tr, y_va, y_te = y[tr], y[va], y[te]
floor_tr = y_tr <= FLOOR + 1e-6; floor_va = y_va <= FLOOR + 1e-6; floor_te = y_te <= FLOOR + 1e-6
print(f'train={len(tr)} val={len(va)} test={len(te)}  floor: {floor_tr.sum()}/{floor_va.sum()}/{floor_te.sum()}')

# --------------------------------------------------------------------------- #
# features (same as floor script: best feats = +Morgan r1,r3,r4)
# --------------------------------------------------------------------------- #
t0 = time.time()
X = np.load('_pampa_feat_cache.npz')
desc, morgan2, molf = X['desc'], X['morgan2'], X['molf']
from rdkit.Chem import rdFingerprintGenerator as RFG
from rdkit.DataStructs import ConvertToNumpyArray
def morgan(r, nbits=2048):
    gen = RFG.GetMorganGenerator(radius=r, fpSize=nbits)
    fp = np.zeros((N, nbits), dtype=np.float64)
    for i, smi in enumerate(smiles_list):
        m = Chem.MolFromSmiles(smi)
        if m is None: continue
        arr = np.zeros(nbits, dtype=np.float64)
        ConvertToNumpyArray(gen.GetFingerprint(m), arr)
        fp[i] = arr
    return fp
r1, r3, r4 = morgan(1), morgan(3), morgan(4)
Xf = np.hstack([desc, morgan2, r1, r3, r4, molf])
print(f'features done ({time.time()-t0:.0f}s), dim={Xf.shape[1]}')

# --------------------------------------------------------------------------- #
# baseline regressor predictions (v4.2 MLP) + floor classifier probabilities
# --------------------------------------------------------------------------- #
import torch
from admet_model import MixedADMETMLP
blob = torch.load('models_v4/pampa_mdck/admet_mlp.pt', map_location='cpu', weights_only=False)
mlp = MixedADMETMLP(input_dim=blob['input_dim'], endpoints=blob['endpoints'],
                    hidden=blob.get('hidden', (256, 128)), dropout=blob.get('dropout', 0.25))
mlp.load_state_dict(blob['state_dict']); mlp.eval()
sc = torch.load('models_v4/pampa_mdck/scaler.pt', map_location='cpu', weights_only=False)
X3 = np.hstack([desc, morgan2, molf]).astype(np.float32)
with torch.no_grad():
    p_all = mlp(torch.from_numpy(sc.transform(X3)))['PAMPA_MDCK'].squeeze(1).numpy()
p_va, p_te = p_all[va], p_all[te]
print(f'baseline: val R2={r2_score(y_va, p_va):.4f}  test R2={r2_score(y_te, p_te):.4f}')

clf = lgb.LGBMClassifier(objective='binary', n_estimators=N_EST, learning_rate=0.03,
                         num_leaves=31, min_child_samples=10, subsample=0.8,
                         subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
                         random_state=SEED, n_jobs=1, verbose=-1)
clf.fit(Xf[tr], floor_tr.astype(int),
        eval_set=[(Xf[va], floor_va.astype(int))],
        callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
Pva = clf.predict_proba(Xf[va])[:, 1]
Pte = clf.predict_proba(Xf[te])[:, 1]
print(f'floor classifier: AUC_val={roc_auc_score(floor_va.astype(int), Pva):.4f}  '
      f'AUC_te={roc_auc_score(floor_te.astype(int), Pte):.4f}')
print(f'  P_floor: test floor rows median={np.median(Pte[floor_te]):.3f} (min {Pte[floor_te].min():.3f}), '
      f'non-floor median={np.median(Pte[~floor_te]):.3f} (max {Pte[~floor_te].max():.3f})')

# --------------------------------------------------------------------------- #
# beta sweep on VAL, then apply to TEST
# --------------------------------------------------------------------------- #
def blend(p, P, beta):
    return p - beta * P * (p + 10.0)

print('\n=== beta sweep (selected on VAL) ===')
print('  beta    R2_val   R2_te(peek)')
best_beta, best_val = 0.0, r2_score(y_va, p_va)
betas = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
for b in betas:
    r2v = r2_score(y_va, blend(p_va, Pva, b))
    r2t = r2_score(y_te, blend(p_te, Pte, b))
    tag = ''
    if r2v > best_val:
        best_val, best_beta = r2v, b
        tag = '  <-- val-best'
    print(f'  {b:<6.2f} {r2v:<9.4f} {r2t:<10.4f}{tag}')

print(f'\nVAL-SELECTED beta = {best_beta:.2f}  (val R2 = {best_val:.4f})')
print(f'HONEST test R2 at selected beta = {r2_score(y_te, blend(p_te, Pte, best_beta)):.4f}   '
      f'(baseline 0.4642)')

# also report the full val curve at the selected beta by region
pb_va = blend(p_va, Pva, best_beta)
pb_te = blend(p_te, Pte, best_beta)
print(f'\nselected-beta by region (test):')
for name, m in (('floor', floor_te), ('non-floor', ~floor_te)):
    print(f'  {name:<9} n={m.sum():>4}  R2={r2_score(y_te[m], pb_te[m]):.4f}  '
          f'pred median={np.median(pb_te[m]):.3f}')

# --------------------------------------------------------------------------- #
# bonus: is P_floor well calibrated? (reliability on test, informational)
# --------------------------------------------------------------------------- #
fr, meanp = calibration_curve(floor_te.astype(int), Pte, n_bins=5, strategy='quantile')
print('\n  calibration (test, quantile bins):  frac_positive vs mean_P')
for f, mp in zip(fr, meanp):
    print(f'    {f:.2f}  {mp:.2f}')

# --------------------------------------------------------------------------- #
# descriptor separability (section 4 of floor script, fixed)
# --------------------------------------------------------------------------- #
print('\n=== physchem descriptor separability (train, floor vs non-floor) ===')
mols = [Chem.MolFromSmiles(s) for s in smiles_list]
cols = {
    'MW': [Descriptors.MolWt(m) if m else 0 for m in mols],
    'LogP': [Descriptors.MolLogP(m) if m else 0 for m in mols],
    'TPSA': [Descriptors.TPSA(m) if m else 0 for m in mols],
    'HBA': [rdMolDescriptors.CalcNumHBA(m) if m else 0 for m in mols],
    'HBD': [rdMolDescriptors.CalcNumHBD(m) if m else 0 for m in mols],
    'RotBonds': [rdMolDescriptors.CalcNumRotatableBonds(m) if m else 0 for m in mols],
    'frSp3': [rdMolDescriptors.CalcFractionCSP3(m) if m else 0 for m in mols],
    'MR': [Descriptors.MolMR(m) if m else 0 for m in mols],
    'QED': [QED.qed(m) if m else 0 for m in mols],
}
Xdesc = np.column_stack(list(cols.values())).astype(np.float64)
clf0 = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31,
                          min_child_samples=20, random_state=SEED, n_jobs=1, verbose=-1)
clf0.fit(Xdesc[tr], floor_tr.astype(int),
         eval_set=[(Xdesc[va], floor_va.astype(int))],
         callbacks=[lgb.early_stopping(50, verbose=False)])
auc0 = roc_auc_score(floor_va.astype(int), clf0.predict_proba(Xdesc[va])[:, 1])
imp = dict(zip(list(cols.keys()), clf0.feature_importances_))
print(f'  9-descriptor floor classifier AUC_val = {auc0:.4f}')
print('  median: floor vs non-floor (train)')
for k in cols:
    idx = list(cols.keys()).index(k)
    print(f'    {k:<9} floor={np.median(Xdesc[tr][floor_tr, idx]):10.2f}   '
          f'non-floor={np.median(Xdesc[tr][~floor_tr, idx]):10.2f}')

print('\nDone.')
