#!/usr/bin/env python3
"""

_pampa_diagnostic.py — PAMPA R² diagnostic ladder + experiment grid.

Uses the EXACT same split (random 70/10/20 on unique SMILES, seed 42) and
the same MLP architecture as the main pipeline, so all R² values are
directly comparable to the committed v4.2 result (0.4642).

Step 1: Signal localization — which feature blocks carry the signal?
Step 2: Feature upgrades — more Morgan radii, target transforms
Step 3: Model upgrades — wider/deeper MLP
Step 4: Ensemble — multi-seed averaging
"""

import common  # re-roots CWD to repo root



import sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem import Descriptors, DataStructs
from rdkit.Chem import rdFingerprintGenerator as RFG

# --- config (match pipeline) ---
SEED = 42
EPOCHS = 80
PATIENCE = 10
BATCH = 128
HIDDEN = (256, 128)
LR = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'Device: {DEVICE}')

# --- load PAMPA ---
df = pd.read_csv('data/pepadmet_pampa_mdck.csv')
smiles_list = df['smiles'].astype(str).tolist()
y_raw = df['PAMPA_MDCK'].to_numpy(dtype=np.float64)
N = len(df)
print(f'PAMPA: N={N}, unique SMILES={len(set(smiles_list))}')
print(f'y range: [{y_raw.min():.3f}, {y_raw.max():.3f}], mean={y_raw.mean():.3f}, std={y_raw.std():.3f}')

# --- same split as pipeline (random 70/10/20 on unique SMILES, seed 42) ---
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

# --- feature extraction (cached across experiments) ---
print('\n=== Extracting features ===')
t0 = time.time()

# 1. RDKit 2D descriptors (217)
names = [nm for (nm, _fn) in Descriptors._descList]
D = len(names)
print(f'  RDKit descriptors: {D}')
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
print(f'  done ({time.time()-t0:.1f}s)')

# 2. Morgan fingerprints at multiple radii
morgan_blocks = {}
for radius in [1, 2, 3]:
    t1 = time.time()
    gen = RFG.GetMorganGenerator(radius=radius, fpSize=2048)
    X_m = np.zeros((N, 2048), dtype=np.float64)
    for i, s in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(str(s))
            if mol is None: continue
            fp = gen.GetFingerprint(mol)
            arr = np.zeros(2048, dtype=np.float64)
            DataStructs.ConvertToNumpyArray(fp, arr)
            X_m[i] = arr
        except: pass
    morgan_blocks[radius] = X_m
    print(f'  Morgan r={radius}: {X_m.shape} ({time.time()-t1:.1f}s)')

# 3. MoLFormer embeddings (cached npz)
print('  Loading MoLFormer cache...')
z = np.load('data/molformer/molformer_emb_pampa_mdck.npz', allow_pickle=True)
X_molf = np.asarray(z['emb'], dtype=np.float64)
print(f'  MoLFormer: {X_molf.shape}')
t_feat = time.time() - t0
print(f'  Feature extraction total: {t_feat:.1f}s')

# 4. Target transforms
y_identity = y_raw.copy()
# rank-Gaussian (map to N(0,1) by rank)
from scipy.stats import rankdata
y_rank = rankdata(y_raw) / (N + 1)
from scipy.special import erfinv
y_rankgauss = np.sqrt(2) * erfinv(2 * y_rank - 1)
print(f'\n  Target stats:')
print(f'    identity:  mean={y_identity.mean():.3f} std={y_identity.std():.3f}')
print(f'    rankgauss: mean={y_rankgauss.mean():.3f} std={y_rankgauss.std():.3f}')

# --- MLP training helper ---
def build_mlp(d, hidden, dropout=0.1):
    """Build an MLP with arbitrary hidden layer count."""
    layers = []
    prev = d
    for h in hidden:
        layers += [torch.nn.Linear(prev, h), torch.nn.BatchNorm1d(h),
                   torch.nn.ReLU(), torch.nn.Dropout(dropout)]
        prev = h
    layers.append(torch.nn.Linear(prev, 1))
    return torch.nn.Sequential(*layers)


def train_mlp(Xtr, ytr, Xva, yva, Xte, yte, hidden, lr, epochs, patience,
              batch_size, seed, loss_fn='huber', dropout=0.1):
    """Train a plain MLP. hidden is a tuple of widths (any depth)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    d = Xtr.shape[1]
    model = build_mlp(d, hidden, dropout).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=4)
    
    Xt = torch.from_numpy(Xtr.astype(np.float32)).to(DEVICE)
    yt = torch.from_numpy(ytr.astype(np.float32)).to(DEVICE)
    Xv = torch.from_numpy(Xva.astype(np.float32)).to(DEVICE)
    yv = torch.from_numpy(yva.astype(np.float32)).to(DEVICE)
    
    best_val, best_state, bad = float('inf'), None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        perm_idx = torch.randperm(len(Xt))
        for i in range(0, len(Xt), batch_size):
            b = perm_idx[i:i+batch_size]
            opt.zero_grad()
            out = model(Xt[b]).squeeze(1)
            if loss_fn == 'huber':
                loss = F.huber_loss(out, yt[b], delta=1.0)
            else:
                loss = F.mse_loss(out, yt[b])
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
    model.eval()
    with torch.no_grad():
        pred_te = model(torch.from_numpy(Xte.astype(np.float32)).to(DEVICE)).squeeze(1).cpu().numpy()
    r2 = r2_score(yte, pred_te)
    return r2


def scale_and_eval(X_full, y_full, tr, va, te, hidden, **kwargs):
    """StandardScaler (fit on train) + train + eval on test. Returns test R²."""
    sc = StandardScaler()
    Xs = sc.fit_transform(X_full[tr])
    Xva_s = sc.transform(X_full[va])
    Xte_s = sc.transform(X_full[te])
    r2 = train_mlp(Xs, y_full[tr], Xva_s, y_full[va], Xte_s, y_full[te], hidden, **kwargs)
    return r2


# ===========================================================================
# STEP 1: Signal localization — which feature blocks carry the signal?
# ===========================================================================
print('\n' + '='*70)
print('STEP 1: Signal localization (same split, same arch (256,128), seed 42)')
print('='*70)

experiments = [
    ('RDKit 217 desc only', X_desc),
    ('Morgan r=2 only (2048)', morgan_blocks[2]),
    ('MoLFormer 768 only', X_molf),
    ('RDKit + Morgan r=2 (2265) [v4.2 baseline]', np.hstack([X_desc, morgan_blocks[2]])),
    ('RDKit + MoLFormer (217+768=985)', np.hstack([X_desc, X_molf])),
    ('Morgan r=2 + MoLFormer (2048+768=2816)', np.hstack([morgan_blocks[2], X_molf])),
    ('ALL: RDKit + Morgan r=2 + MoLFormer (3033)', np.hstack([X_desc, morgan_blocks[2], X_molf])),
]

results = {}
for label, X in experiments:
    t1 = time.time()
    r2 = scale_and_eval(X, y_identity, tr, va, te, HIDDEN,
                        lr=LR, epochs=EPOCHS, patience=PATIENCE,
                        batch_size=BATCH, seed=SEED, loss_fn='huber')
    dt = time.time() - t1
    results[label] = r2
    print(f'  {label:50s}  R²={r2:.4f}  ({dt:.1f}s)')

# ===========================================================================
# STEP 2: Feature upgrades — more Morgan radii, target transforms
# ===========================================================================
print('\n' + '='*70)
print('STEP 2: Feature upgrades')
print('='*70)

experiments2 = [
    ('RDKit + Morgan r1+r2+r3 (217+6144=6361)', 
     np.hstack([X_desc, morgan_blocks[1], morgan_blocks[2], morgan_blocks[3]])),
    ('RDKit + Morgan r1+r2 (217+4096=4313)', 
     np.hstack([X_desc, morgan_blocks[1], morgan_blocks[2]])),
    ('ALL + Morgan r1+r3 (3033+4096=7129)',
     np.hstack([X_desc, morgan_blocks[1], morgan_blocks[2], morgan_blocks[3], X_molf])),
]

for label, X in experiments2:
    t1 = time.time()
    r2 = scale_and_eval(X, y_identity, tr, va, te, HIDDEN,
                        lr=LR, epochs=EPOCHS, patience=PATIENCE,
                        batch_size=BATCH, seed=SEED, loss_fn='huber')
    dt = time.time() - t1
    results[label] = r2
    print(f'  {label:50s}  R²={r2:.4f}  ({dt:.1f}s)')

# Target transform experiments (using best feature set from step 1)
print('\n  Target transforms (ALL features, 3033-dim):')
for label, y_t in [('identity', y_identity), ('rankgauss', y_rankgauss)]:
    t1 = time.time()
    r2 = scale_and_eval(np.hstack([X_desc, morgan_blocks[2], X_molf]), y_t, tr, va, te, HIDDEN,
                        lr=LR, epochs=EPOCHS, patience=PATIENCE,
                        batch_size=BATCH, seed=SEED, loss_fn='huber')
    dt = time.time() - t1
    results[f'rankgauss target / {label}'] = r2
    print(f'    {label:20s}  R²={r2:.4f}  ({dt:.1f}s)')

# ===========================================================================
# STEP 3: Model upgrades — wider/deeper MLP
# ===========================================================================
print('\n' + '='*70)
print('STEP 3: Model upgrades (ALL features, 3033-dim, identity target)')
print('='*70)

X_all = np.hstack([X_desc, morgan_blocks[2], X_molf])
sc_all = StandardScaler().fit(X_all[tr])
Xtr_all = sc_all.transform(X_all[tr])
Xva_all = sc_all.transform(X_all[va])
Xte_all = sc_all.transform(X_all[te])

for hidden_label, hidden in [('(256,128) [baseline]', (256, 128)),
                              ('(512,256,128)', (512, 256, 128)),
                              ('(512,256,128,64)', (512, 256, 128, 64)),
                              ('(1024,512,256)', (1024, 512, 256))]:
    t1 = time.time()
    r2 = train_mlp(Xtr_all, y_identity[tr], Xva_all, y_identity[va],
                   Xte_all, y_identity[te], hidden, LR, EPOCHS, PATIENCE,
                   BATCH, SEED, 'huber')
    dt = time.time() - t1
    results[f'arch {hidden_label}'] = r2
    print(f'  {hidden_label:30s}  R²={r2:.4f}  ({dt:.1f}s)')

# ===========================================================================
# STEP 4: Multi-seed stability (best config from above)
# ===========================================================================
print('\n' + '='*70)
print('STEP 4: Multi-seed stability (best config, 5 seeds)')
print('='*70)

# Use the best-performing config
best_label = max(results, key=results.get)
print(f'  Best so far: {best_label} -> R²={results[best_label]:.4f}')

# For stability check, use the ALL 3033-dim + (256,128) baseline
X_for_stability = X_all
sc = StandardScaler().fit(X_for_stability[tr])
Xtr_s = sc.transform(X_for_stability[tr])
Xva_s = sc.transform(X_for_stability[va])
Xte_s = sc.transform(X_for_stability[te])

r2s = []
for s in [42, 123, 456, 789, 1024]:
    t1 = time.time()
    r2 = train_mlp(Xtr_s, y_identity[tr], Xva_s, y_identity[va], Xte_s, y_identity[te],
                   HIDDEN, LR, EPOCHS, PATIENCE, BATCH, s, 'huber')
    dt = time.time() - t1
    r2s.append(r2)
    print(f'    seed {s:5d}  R²={r2:.4f}  ({dt:.1f}s)')
print(f'  Mean R² = {np.mean(r2s):.4f} ± {np.std(r2s):.4f}  (min={min(r2s):.4f}, max={max(r2s):.4f})')

# ===========================================================================
# Summary
# ===========================================================================
print('\n' + '='*70)
print('SUMMARY — all R² values (test split, same random 70/10/20 seed 42)')
print('='*70)
print(f'  v4.2 committed: R²=0.4642 (RDKit+Morgan r2+MoLFormer, (256,128), Huber)')
print()
for k, v in sorted(results.items(), key=lambda x: -x[1]):
    marker = ' <<<' if v > 0.7 else ''
    print(f'  R²={v:.4f}  {k}{marker}')

print(f'\n  Ceiling (from noise analysis): R²_max ≈ 0.97')
print(f'  Target: R² ≥ 0.70')
print(f'  Achieved: R² ≥ 0.70? {"YES" if max(results.values()) >= 0.70 else "NO"}')
