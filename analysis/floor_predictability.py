#!/usr/bin/env python3
"""

_pampa_floor.py — IS THE CENSORING FLOOR PREDICTABLE FROM STRUCTURE?

Decisive experiment for the PAMPA -> 0.7 question.

Facts from the ceiling analysis (same split, seed 42):
  * current v4.2 MLP: overall R2=0.4642, non-floor R2=0.6317
  * 47/1457 test rows are censored at FLOOR=-10.0000
  * floor rows contribute 64% of total ss_res (MAE 3.43 there)
  * oracle (floor identifiable + current regressor) -> R2 ~= 0.807
  * perfect non-floor + floor->mean -> only 0.5387

So the ONLY path to R2 >= 0.7 is a floor classifier (or a regressor that
learns to emit -10). This script measures the best achievable floor
classifier (LightGBM, tuned) and the two-stage R2 it unlocks.
"""

import common  # re-roots CWD to repo root



import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score, roc_auc_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, QED

SEED = 42
FLOOR = -10.0000
FLOOR_TOL = 1e-6
N_EST = 1000
PATIENCE = 100

# --------------------------------------------------------------------------- #
# data + verbatim split (identical to train_pepadmet_model.py / round3)
# --------------------------------------------------------------------------- #
df = pd.read_csv('data/pepadmet_pampa_mdck.csv')
smiles_list = df['smiles'].astype(str).tolist()
y = df['PAMPA_MDCK'].to_numpy(dtype=np.float64)
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
y_tr, y_va, y_te = y[tr], y[va], y[te]
floor_tr = y_tr <= FLOOR + FLOOR_TOL
floor_va = y_va <= FLOOR + FLOOR_TOL
floor_te = y_te <= FLOOR + FLOOR_TOL
print(f'N={N} train={len(tr)} val={len(va)} test={len(te)}')
print(f'floor: train={floor_tr.sum()} ({floor_tr.mean()*100:.2f}%)  '
      f'val={floor_va.sum()} ({floor_va.mean()*100:.2f}%)  test={floor_te.sum()} ({floor_te.mean()*100:.2f}%)')

# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #
print('\n=== features ===')
X = np.load('_pampa_feat_cache.npz')
desc, morgan2, molf = X['desc'], X['morgan2'], X['molf']

def morgan(r, nbits=2048):
    from rdkit.Chem import rdFingerprintGenerator as RFG
    from rdkit.DataStructs import ConvertToNumpyArray
    gen = RFG.GetMorganGenerator(radius=r, fpSize=nbits)
    fp = np.zeros((N, nbits), dtype=np.float64)
    for i, smi in enumerate(smiles_list):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        arr = np.zeros(nbits, dtype=np.float64)
        ConvertToNumpyArray(gen.GetFingerprint(m), arr)
        fp[i] = arr
    return fp

r1 = morgan(1); r3 = morgan(3); r4 = morgan(4)

FEATS = {
    'v4.2 (2265+768)': np.hstack([desc, morgan2, molf]),
    '+Morgan r1,r3,r4': np.hstack([desc, morgan2, r1, r3, r4, molf]),
}

# --------------------------------------------------------------------------- #
# 0) regressor-predictions -> floor AUC (is the model's ranking already
#    floor-aware?)
# --------------------------------------------------------------------------- #
import torch
from admet_model import MixedADMETMLP
blob = torch.load('models_v4/pampa_mdck/admet_mlp.pt', map_location='cpu', weights_only=False)
mlp = MixedADMETMLP(input_dim=blob['input_dim'], endpoints=blob['endpoints'],
                    hidden=blob.get('hidden', (256, 128)), dropout=blob.get('dropout', 0.25))
mlp.load_state_dict(blob['state_dict']); mlp.eval()
sc = torch.load('models_v4/pampa_mdck/scaler.pt', map_location='cpu', weights_only=False)
with torch.no_grad():
    p_mlp = mlp(torch.from_numpy(sc.transform(FEATS['v4.2 (2265+768)'][te].astype(np.float32))))['PAMPA_MDCK'].squeeze(1).numpy()

auc_by_pred = roc_auc_score(floor_te.astype(int), -p_mlp)  # lower pred should = floor
print(f'\n0) current MLP: AUC of (pred -> floor) = {auc_by_pred:.4f}  (0.5 = no rank signal)')
# where do the 47 test floor rows sit among all test rows, ranked by pred?
order = np.argsort(p_mlp)
ranks = np.empty(len(order), dtype=int); ranks[order] = np.arange(len(order))
floor_ranks = ranks[floor_te]
print(f'    floor rows rank positions (0=lowest pred): mean={floor_ranks.mean():.0f}, '
      f'median={np.median(floor_ranks):.0f}, in bottom-47: {(floor_ranks < 47).sum()}/47, '
      f'in bottom-94: {(floor_ranks < 94).sum()}/47, in bottom-141: {(floor_ranks < 141).sum()}/47')
nf_p = p_mlp[~floor_te]; fp_ = p_mlp[floor_te]
print(f'    pred range: floor {fp_.min():.2f}..{fp_.max():.2f} (median {np.median(fp_):.2f})  |  non-floor {nf_p.min():.2f}..{nf_p.max():.2f}')

# --------------------------------------------------------------------------- #
# 1) floor classifier grid (LightGBM), AUC + best threshold + two-stage R2
# --------------------------------------------------------------------------- #
print(f'\n1) floor classifier grid (N_EST={N_EST}, early-stop val)')
clf_results = []
for fname, Xf in FEATS.items():
    for lr in (0.03, 0.05, 0.1):
        for nl in (31, 63, 127):
            for mcs in (10, 30):
                clf = lgb.LGBMClassifier(
                    objective='binary', n_estimators=N_EST, learning_rate=lr,
                    num_leaves=nl, min_child_samples=mcs,
                    subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                    reg_lambda=1.0, random_state=SEED, n_jobs=1, verbose=-1)
                clf.fit(Xf[tr], floor_tr.astype(int),
                        eval_set=[(Xf[va], floor_va.astype(int))],
                        callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
                pva = clf.predict_proba(Xf[va])[:, 1]
                auc_va = roc_auc_score(floor_va.astype(int), pva)
                pte = clf.predict_proba(Xf[te])[:, 1]
                auc_te = roc_auc_score(floor_te.astype(int), pte)

                # best threshold on VAL (max F1-ish: maximize TPR - FPR balance via PR)
                prec, rec, th = precision_recall_curve(floor_va.astype(int), pva)
                f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
                thr = th[int(np.argmax(f1[:-1]))] if len(th) > 0 else 0.5
                pred_floor = pte > thr
                tp = (pred_floor & floor_te).sum(); fp_2 = (pred_floor & ~floor_te).sum()
                fn = (~pred_floor & floor_te).sum()
                prec_t = tp / max(tp + fp_2, 1); rec_t = tp / max(tp + fn, 1)
                clf_results.append(dict(feats=fname, lr=lr, nl=nl, mcs=mcs,
                                        auc_va=auc_va, auc_te=auc_te, thr=thr,
                                        prec=prec_t, rec=rec_t, n_flag=int(pred_floor.sum()),
                                        score=auc_va, pte=pte))

clf_results.sort(key=lambda r: -r['score'])
print(f'  {"feats":<20} {"lr":>5} {"nl":>4} {"mcs":>4}  AUC_val  AUC_te   thr   prec  rec  n_flag')
for r in clf_results[:8]:
    print(f'  {r["feats"]:<20} {r["lr"]:>5} {r["nl"]:>4} {r["mcs"]:>4}  {r["auc_va"]:.4f} {r["auc_te"]:.4f}  {r["thr"]:.3f}  {r["prec"]:.3f} {r["rec"]:.3f}  {r["n_flag"]:>4}')

# --------------------------------------------------------------------------- #
# 2) two-stage R2 with the best classifier (val-tuned threshold)
#    stage-2 = current v4.2 MLP predictions (p_mlp) on non-floor rows
# --------------------------------------------------------------------------- #
print('\n2) two-stage: best floor classifier + current MLP regressor')
best = clf_results[0]
pte_c = best['pte']
p2 = p_mlp.copy()
flag = pte_c > best['thr']
p2[flag] = FLOOR
r2_two = r2_score(y_te, p2)
print(f'  classifier: {best["feats"]} lr={best["lr"]} nl={best["nl"]} mcs={best["mcs"]} '
      f'AUC_te={best["auc_te"]:.4f} thr={best["thr"]:.3f} prec={best["prec"]:.3f} rec={best["rec"]:.3f}')
print(f'  two-stage R2 = {r2_two:.4f}   (baseline MLP = {r2_score(y_te, p_mlp):.4f})')

# sweep the threshold to show the precision/recall -> R2 tradeoff
print('  threshold sweep (test):')
for q in (0.50, 0.60, 0.70, 0.80, 0.90, 0.95):
    th_q = np.quantile(pte_c, 1 - q / 100.0)
    pf = pte_c > th_q
    p2q = p_mlp.copy(); p2q[pf] = FLOOR
    tp = (pf & floor_te).sum(); fpq = (pf & ~floor_te).sum()
    print(f'    top-{q:.0f}% flagged (thr={th_q:.3f}, n={pf.sum()}): '
          f'R2={r2_score(y_te, p2q):.4f}  floor-TP={tp} FP={fpq}  '
          f'prec={tp/max(tp+fpq,1):.3f} rec={tp/max((pf&floor_te).sum()+(~pf&floor_te).sum(),1):.3f}')

# --------------------------------------------------------------------------- #
# 3) oracle floor curve: R2 as a function of how many of the 47 true floor
#    rows are correctly flagged (and the FP cost at each step)
# --------------------------------------------------------------------------- #
print('\n3) oracle sweep: flag k of the 47 true floor rows (no FP) -> R2')
floor_pos = np.where(floor_te)[0]
for k in (0, 10, 20, 30, 40, 47):
    p_or = p_mlp.copy()
    p_or[floor_pos[:k]] = FLOOR
    print(f'    k={k:>2}: R2={r2_score(y_te, p_or):.4f}')
print('    (with the CURRENT regressor on non-floor rows; oracle = k=47)')

# --------------------------------------------------------------------------- #
# 4) descriptor separability: how well do simple physchem descriptors alone
#    separate floor vs non-floor? (cheap sanity check of the signal)
# --------------------------------------------------------------------------- #
print('\n4) physchem-descriptor separability (train AUC, floor vs non-floor)')
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
importances = dict(zip(list(cols.keys()), clf0.feature_importances_))
print(f'    9-descriptor floor classifier AUC_val = {auc0:.4f}  (importance: '
      + ', '.join(f'{k}={v:.0f}' for k, v in sorted(importances.items(), key=lambda kv: -kv[1])[:5]) + ')')
# floor vs non-floor medians on key descriptors
print('    median descriptor: floor vs non-floor (train)')
for k in ('MW', 'LogP', 'TPSA', 'HBA', 'HBD', 'RotBonds', 'frSp3'):
    idx = list(cols.keys()).index(k)
    print(f'      {k:<9} floor={np.median(Xdesc[tr][floor_tr, idx]):10.2f}   '
          f'non-floor={np.median(Xdesc[tr][~floor_tr, idx]):10.2f}')

print('\nDone.')
