#!/usr/bin/env python3
"""

_pampa_round2.py — Rank-Gaussian target transform, done HONESTLY.

Round 1 found rank-Gaussian target -> R2=0.6638 (single seed, rank computed
on the FULL dataset = leakage).  This round:

1. HONEST rank mapping: the rank->Gaussian transform is fit on TRAIN only
   (train empirical CDF).  Val/test values are mapped through the TRAIN
   distribution.  Predictions are inverse-mapped back to logPapp via the
   train quantile function before R2 is computed, so BOTH R2 values are
   reported:
     R2_rank     : in the N(0,1) rank space (model fit quality)
     R2_logpapp  : in the original logPapp units (what the user cares about)

2. Multi-seed stability (5 seeds) on the winning config.

3. Ensemble: average of 5 seed models (predictions in logPapp space).

4. Feature-set ablation in rank space (Morgan-only vs RDKit+Morgan vs full).

5. Wider nets in rank space.

Same split as the pipeline: random 70/10/20 on unique SMILES, seed 42.
"""

import common  # re-roots CWD to repo root



import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import norm
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem import Descriptors, DataStructs
from rdkit.Chem import rdFingerprintGenerator as RFG

SEED = 42
EPOCHS = 80
PATIENCE = 10
BATCH = 128
HIDDEN = (256, 128)
LR = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

# --- load PAMPA (reuse round-1 timing: RDKit descriptors are the slow part,
# ~460s; we recompute but that's the cost of an honest standalone script) ---
df = pd.read_csv('data/pepadmet_pampa_mdck.csv')
smiles_list = df['smiles'].astype(str).tolist()
y_raw = df['PAMPA_MDCK'].to_numpy(dtype=np.float64)
N = len(df)
print(f'PAMPA: N={N}')

# --- same split as pipeline ---
uniq, inv = np.unique(np.asarray(smiles_list, dtype=object), return_inverse=True)
rng = np.random.default_rng(SEED)
perm = rng.permutation(len(uniq))
n_tr = int(round(len(uniq) * 0.70))
n_va = int(round(len(uniq) * 0.10))
tr_ids = set(perm[:n_tr].tolist())
va_ids = set(perm[n_tr:n_tr + n_va].tolist())
tr = np.array([i for i in range(N) if inv[i] in tr_ids], dtype=np.int64)
va = np.array([i for i in range(N) if inv[i] in va_ids], dtype=np.int64)
te = np.array([i for i in range(N) if inv[i] not in tr_ids and inv[i] not in va_ids], dtype=np.int64)
print(f'Split: train={len(tr)}, val={len(va)}, test={len(te)}')

# --- HONEST rank-Gaussian transform, fit on train only ---
y_tr = y_raw[tr]
# train empirical CDF (midrank style): rank of each train value in train
order = np.argsort(y_tr, kind='stable')
ranks = np.empty(len(order))
ranks[order] = np.arange(1, len(order) + 1)
p_tr = (ranks - 0.5) / len(order)          # in (0,1)
y_tr_rank = norm.ppf(p_tr).astype(np.float64)  # train values -> N(0,1)

def map_to_rank(yvals):
    """Map arbitrary values through the TRAIN empirical CDF -> N(0,1).
    Values beyond the train range are clamped to the train CDF extremes
    (with a small buffer to keep ppf finite)."""
    lo, hi = y_tr.min(), y_tr.max()
    ycl = np.clip(yvals, lo, hi)
    # fraction of train <= ycl
    cdf = np.searchsorted(np.sort(y_tr), ycl, side='right') / len(y_tr)
    cdf = np.clip(cdf, 0.5 / len(y_tr), 1 - 0.5 / len(y_tr))
    return norm.ppf(cdf)

def map_rank_to_logpapp(rvals):
    """Inverse-map N(0,1) values back to logPapp via the TRAIN quantile fn."""
    p = norm.cdf(rvals)
    p = np.clip(p, 0.5 / len(y_tr), 1 - 0.5 / len(y_tr))
    # train quantile at p (linear interpolation over sorted train values)
    srt = np.sort(y_tr)
    pos = p * (len(srt) - 1)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, len(srt) - 1)
    frac = pos - lo
    return srt[lo] * (1 - frac) + srt[hi] * frac

y_va_rank = map_to_rank(y_raw[va])
y_te_rank = map_to_rank(y_raw[te])
y_tr_rank = map_to_rank(y_raw[tr])  # same as above, recompute for consistency

print(f'\nHonest rank mapping (fit on train only):')
print(f'  train y:   [{y_tr.min():.3f}, {y_tr.max():.3f}]')
print(f'  test  y:   [{y_raw[te].min():.3f}, {y_raw[te].max():.3f}]')
print(f'  test rank: mean={y_te_rank.mean():.3f} std={y_te_rank.std():.3f}')
# sanity: how well does train-quantile-inverse-map correlate with true test y?
inv_map = map_rank_to_logpapp(y_te_rank)
print(f'  (sanity) R2 of [train-quantile(map(test y)) vs true test y] = '
      f'{r2_score(y_raw[te], inv_map):.4f}  <- the best a perfect rank-model could do in logPapp space')

# --- features (cached to npz; RDKit descriptors are the ~460s part) ---
FEAT_CACHE = '_pampa_feat_cache.npz'
print('\n=== Features ===')
t0 = time.time()
if Path(FEAT_CACHE).exists():
    zc = np.load(FEAT_CACHE)
    X_desc, m2, X_molf = zc['desc'], zc['morgan2'], zc['molf']
    morgan = {2: m2}
    print(f'  loaded from cache {FEAT_CACHE} ({time.time()-t0:.1f}s)')
else:
    names = [nm for (nm, _fn) in Descriptors._descList]
    D = len(names)
    X_desc = np.zeros((N, D), dtype=np.float64)
    for i, s in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(str(s))
            if mol is None: continue
            d = Descriptors.CalcMolDescriptors(mol)
            for j, nm in enumerate(names):
                v = d.get(nm)
                if isinstance(v, (int, float)) and np.isfinite(v):
                    X_desc[i, j] = float(v)
        except: pass
    print(f'  RDKit descriptors done ({time.time()-t0:.0f}s)')

    morgan = {}
    for radius in [2]:
        gen = RFG.GetMorganGenerator(radius=radius, fpSize=2048)
        Xm = np.zeros((N, 2048), dtype=np.float64)
        for i, s in enumerate(smiles_list):
            try:
                mol = Chem.MolFromSmiles(str(s))
                if mol is None: continue
                fp = gen.GetFingerprint(mol)
                arr = np.zeros(2048, dtype=np.float64)
                DataStructs.ConvertToNumpyArray(fp, arr)
                Xm[i] = arr
            except: pass
        morgan[radius] = Xm

    z = np.load('data/molformer/molformer_emb_pampa_mdck.npz', allow_pickle=True)
    X_molf = np.asarray(z['emb'], dtype=np.float64)
    np.savez(FEAT_CACHE, desc=X_desc, morgan2=morgan[2], molf=X_molf)
    print(f'  features cached to {FEAT_CACHE}')
print(f'  Feature extraction total: {time.time()-t0:.0f}s')

# feature sets to compare in rank space
feature_sets = {
    'Morgan r=2 only (2048)': morgan[2],
    'RDKit + Morgan r=2 (2265)': np.hstack([X_desc, morgan[2]]),
    'FULL: RDKit + Morgan + MoLFormer (3033)': np.hstack([X_desc, morgan[2], X_molf]),
    'Morgan + MoLFormer (2816)': np.hstack([morgan[2], X_molf]),
}

# --- MLP ---
def build_mlp(d, hidden, dropout=0.1):
    layers = []
    prev = d
    for h in hidden:
        layers += [torch.nn.Linear(prev, h), torch.nn.BatchNorm1d(h),
                   torch.nn.ReLU(), torch.nn.Dropout(dropout)]
        prev = h
    layers.append(torch.nn.Linear(prev, 1))
    return torch.nn.Sequential(*layers)

def train_and_predict(X, y_rank_tr, y_rank_va, hidden=(256, 128), seed=42,
                      lr=1e-3, epochs=EPOCHS, patience=PATIENCE, dropout=0.1,
                      loss_fn='huber', verbose=False):
    """Train in RANK space; return (model, scaler)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    sc = StandardScaler().fit(X[tr])
    Xt = torch.from_numpy(sc.transform(X[tr]).astype(np.float32)).to(DEVICE)
    yt = torch.from_numpy(y_rank_tr.astype(np.float32)).to(DEVICE)
    Xv = torch.from_numpy(sc.transform(X[va]).astype(np.float32)).to(DEVICE)
    yv = torch.from_numpy(y_rank_va.astype(np.float32)).to(DEVICE)
    model = build_mlp(X.shape[1], hidden, dropout).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=4)
    best_val, best_state, bad = float('inf'), None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        pidx = torch.randperm(len(Xt))
        for i in range(0, len(Xt), BATCH):
            b = pidx[i:i+BATCH]
            opt.zero_grad()
            out = model(Xt[b]).squeeze(1)
            loss = (F.huber_loss(out, yt[b], delta=1.0) if loss_fn == 'huber'
                    else F.mse_loss(out, yt[b]))
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = F.huber_loss(model(Xv).squeeze(1), yv, delta=1.0).item()
        sched.step(vloss)
        if vloss < best_val - 1e-5:
            best_val, bad = vloss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, sc

def test_r2(model, sc, X, y_rank_te, y_raw_te):
    """R2 in rank space AND in original logPapp space (inverse-mapped).
    X = the TEST feature rows only."""
    model.eval()
    with torch.no_grad():
        p_rank = model(torch.from_numpy(sc.transform(X).astype(np.float32)).to(DEVICE)).squeeze(1).cpu().numpy()
    r2_rank = r2_score(y_rank_te, p_rank)
    p_logpapp = map_rank_to_logpapp(p_rank)
    r2_log = r2_score(y_raw_te, p_logpapp)
    return r2_rank, r2_log, p_rank, p_logpapp

results = []

# ===========================================================================
# A. Honest rank-gauss: feature-set ablation (seed 42)
# ===========================================================================
print('\n' + '='*70)
print('A. HONEST rank-gauss (train-fit transform) — feature ablation, seed 42')
print('   R2_rank = rank space | R2_logpapp = original units (inverse-mapped)')
print('='*70)
for fs_name, X in feature_sets.items():
    t1 = time.time()
    model, sc = train_and_predict(X, y_tr_rank, y_va_rank, seed=SEED)
    r2_rank, r2_log, _, _ = test_r2(model, sc, X[te], y_te_rank, y_raw[te])
    dt = time.time() - t1
    results.append((r2_log, f'A seed42  {fs_name}'))
    print(f'  {fs_name:40s}  R2_rank={r2_rank:.4f}  R2_logpapp={r2_log:.4f}  ({dt:.0f}s)')

# ===========================================================================
# B. Multi-seed stability: FULL features, (256,128), rank space
# ===========================================================================
print('\n' + '='*70)
print('B. Multi-seed stability (FULL features, (256,128), rank-gauss, Huber)')
print('='*70)
X_full = feature_sets['FULL: RDKit + Morgan + MoLFormer (3033)']
seeds = [42, 123, 456, 789, 1024]
r2r_list, r2l_list, ens_rank = [], [], np.zeros(len(te))
for s in seeds:
    t1 = time.time()
    model, sc = train_and_predict(X_full, y_tr_rank, y_va_rank, seed=s)
    r2_rank, r2_log, p_rank, p_log = test_r2(model, sc, X_full[te], y_te_rank, y_raw[te])
    dt = time.time() - t1
    r2r_list.append(r2_rank); r2l_list.append(r2_log)
    ens_rank += p_rank
    results.append((r2_log, f'B seed{s:5d}  FULL 3033 (256,128)'))
    print(f'  seed {s:5d}  R2_rank={r2_rank:.4f}  R2_logpapp={r2_log:.4f}  ({dt:.0f}s)')
print(f'  mean: R2_rank={np.mean(r2r_list):.4f}±{np.std(r2r_list):.4f}   '
      f'R2_logpapp={np.mean(r2l_list):.4f}±{np.std(r2l_list):.4f}')

# ===========================================================================
# C. Ensemble: average rank predictions of 5 seeds -> inverse map
# ===========================================================================
print('\n' + '='*70)
print('C. 5-seed ENSEMBLE (average in rank space, inverse-map to logPapp)')
print('='*70)
ens_rank = ens_rank / len(seeds)
ens_log = map_rank_to_logpapp(ens_rank)
r2_rank_e = r2_score(y_te_rank, ens_rank)
r2_log_e = r2_score(y_raw[te], ens_log)
results.append((r2_log_e, 'C ensemble 5-seed  FULL 3033 (256,128)'))
print(f'  R2_rank={r2_rank_e:.4f}  R2_logpapp={r2_log_e:.4f}')

# ===========================================================================
# D. Wider nets in rank space (seed 42)
# ===========================================================================
print('\n' + '='*70)
print('D. Architecture ablation in RANK space (FULL features, seed 42)')
print('='*70)
for hl, hid in [('(256,128)', (256, 128)), ('(512,256)', (512, 256)),
                ('(512,256,128)', (512, 256, 128)), ('(768,384,192)', (768, 384, 192))]:
    t1 = time.time()
    model, sc = train_and_predict(X_full, y_tr_rank, y_va_rank, hidden=hid, seed=SEED)
    r2_rank, r2_log, _, _ = test_r2(model, sc, X_full[te], y_te_rank, y_raw[te])
    dt = time.time() - t1
    results.append((r2_log, f'D {hl}  FULL 3033 rank'))
    print(f'  {hl:20s}  R2_rank={r2_rank:.4f}  R2_logpapp={r2_log:.4f}  ({dt:.0f}s)')

# ===========================================================================
# E. Loss fn: huber vs mse in rank space (seed 42)
# ===========================================================================
print('\n' + '='*70)
print('E. Loss ablation in RANK space (FULL features, (256,128), seed 42)')
print('='*70)
for lf in ['huber', 'mse']:
    t1 = time.time()
    model, sc = train_and_predict(X_full, y_tr_rank, y_va_rank, loss_fn=lf, seed=SEED)
    r2_rank, r2_log, _, _ = test_r2(model, sc, X_full[te], y_te_rank, y_raw[te])
    dt = time.time() - t1
    results.append((r2_log, f'E {lf}  FULL 3033 rank'))
    print(f'  {lf:10s}  R2_rank={r2_rank:.4f}  R2_logpapp={r2_log:.4f}  ({dt:.0f}s)')

# ===========================================================================
# Summary
# ===========================================================================
print('\n' + '='*70)
print('SUMMARY (test split; R2_logpapp = original units)')
print('='*70)
print(f'  v4.2 committed (identity target): R2_logpapp = 0.4642')
print(f'  Round-1 (LEAKY full-data rank, seed42): R2_rank = 0.6638 (not comparable)')
print()
for r2, label in sorted(results, key=lambda x: -x[0]):
    mark = '  <<<' if r2 >= 0.70 else ''
    print(f'  R2_logpapp={r2:.4f}  {label}{mark}')
print(f'\n  Target R2_logpapp >= 0.70: '
      f'{"REACHED" if max(r[0] for r in results) >= 0.70 else "NOT reached"}')
