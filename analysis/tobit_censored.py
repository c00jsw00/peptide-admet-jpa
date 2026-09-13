#!/usr/bin/env python3
"""

_pampa_tobit.py — Tobit (censored) regression for PAMPA: the statistically
CORRECT model for left-censored data (floor at y*=-10.0000).

Model: y* = f(x) + eps, eps ~ N(0, sigma^2); observed y = max(y*, c), c=-10.

Log-likelihood per row:
  y > c : -log sigma - (y-f)^2/(2 sigma^2) - 0.5 log(2 pi)
  y == c: log Phi((c-f)/sigma)

Prediction for honest R2 = E[y|x]:
  E[y|x] = f*Phi((f-c)/sigma) + sigma*phi((f-c)/sigma) + c*Phi((c-f)/sigma)

This is the only model family that uses the censored rows' likelihood
properly (vs. blend/two-stage post-hoc fixes). If Tobit also lands at
~0.46-0.50, the 0.7 question is definitively closed.
"""

import common  # re-roots CWD to repo root



import warnings
warnings.filterwarnings('ignore')
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

SEED = 42
C = -10.0000  # censoring floor
EPOCHS = 300
PATIENCE = 60
BATCH = 128

# --------------------------------------------------------------------------- #
# data + verbatim split (identical to pipeline)
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
print(f'train={len(tr)} val={len(va)} test={len(te)}')
print(f'censored: train={(y[tr] <= C + 1e-6).sum()} val={(y[va] <= C + 1e-6).sum()} test={(y[te] <= C + 1e-6).sum()}')

# --------------------------------------------------------------------------- #
# features: v4.2 set (2265 + 768), same scaler convention
# --------------------------------------------------------------------------- #
t0 = time.time()
X = np.load('_pampa_feat_cache.npz')
X_all = np.hstack([X['desc'], X['morgan2'], X['molf']])
from sklearn.preprocessing import StandardScaler
sc = StandardScaler().fit(X_all[tr])
Xtr = torch.tensor(sc.transform(X_all[tr]), dtype=torch.float32)
Xva = torch.tensor(sc.transform(X_all[va]), dtype=torch.float32)
Xte = torch.tensor(sc.transform(X_all[te]), dtype=torch.float32)
ytr = torch.tensor(y[tr], dtype=torch.float32)
yva = torch.tensor(y[va], dtype=torch.float32)
yte = torch.tensor(y[te], dtype=torch.float32)
print(f'features ({time.time()-t0:.0f}s), dim={X_all.shape[1]}')

# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
torch.manual_seed(SEED)
np.random.seed(SEED)
net = nn.Sequential(
    nn.Linear(X_all.shape[1], 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.25),
    nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.25),
    nn.Linear(128, 1),
)
log_sigma = nn.Parameter(torch.tensor(-0.2))  # sigma ~ 0.82
opt = torch.optim.Adam(list(net.parameters()) + [log_sigma], lr=1e-3, weight_decay=1e-5)

def _log_phi_cdf(x):
    """log Phi(x), numerically stable AND NaN-safe gradients.

    Root cause of the earlier NaN: float32 erf(-4.24) rounds to exactly -1.0,
    so 0.5*(1+erf) == 0 and log(0) == -inf with an inf gradient. Fix: the
    direct branch is ALWAYS evaluated in float64 (where erf has no such
    underflow until x ~ -11), then cast back. For x < -6 the two-term Mills
    expansion (error <= 3/x^4 ~ 2.3e-3 at x=-6, negligible: P there ~ 1e-9).
    Clamps ensure the unused branch is never evaluated at a singular
    argument (0*inf / log(0) grad hazards)."""
    x64 = x.double()
    log2pi = 0.9189385332046727
    xc = x64.clamp(min=-6.0)
    direct = torch.log(0.5 * (1.0 + torch.erf(xc / 1.4142135623730951)))
    xa = x64.clamp(max=-6.0)
    # log Mills ratio: log(1/|x| - 1/|x|^3) = log(1/|x|) + log1p(-1/x^2)
    mills = -torch.log(-xa) + torch.log1p(-1.0 / (xa * xa))
    asym = -0.5 * xa * xa - log2pi + mills
    return torch.where(x64 < -6.0, asym, direct)

def nll_tobit(f, y_obs, sigma):
    alpha = (f - C) / sigma
    uncens = y_obs > C + 1e-6
    ll_u = -0.5 * ((y_obs - f) / sigma) ** 2 - torch.log(sigma)
    # censored rows: P(y* < C) = Phi((C - f)/sigma)  <-- sign matters
    ll_c = _log_phi_cdf((C - f) / sigma)
    return -torch.where(uncens, ll_u, ll_c).mean()

def pred_mean(f, sigma):
    """E[y|x] under Tobit: f*Phi((f-c)/s) + s*phi((f-c)/s) + c*Phi((c-f)/s)"""
    a = (f - C) / sigma
    Phi = 0.5 * (1.0 + torch.erf(a / np.sqrt(2.0)))
    phi = torch.exp(-0.5 * a ** 2) / np.sqrt(2 * np.pi)
    return f * Phi + sigma * phi + C * (1.0 - Phi)

best_va, best_state, patience, patience_left = float('inf'), None, PATIENCE, PATIENCE
best_ep, best_sigma = 0, 0.8
hist = []
t0 = time.time()
for ep in range(EPOCHS):
    net.train()
    idx = torch.randperm(len(ytr))
    tot = 0.0
    for i in range(0, len(idx), BATCH):
        b = idx[i:i + BATCH]
        f = net(Xtr[b]).squeeze(1)
        sigma = torch.exp(log_sigma)
        loss = nll_tobit(f, ytr[b], sigma)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item() * len(b)
    # val
    net.eval()
    with torch.no_grad():
        fva = net(Xva).squeeze(1)
        sigma = torch.exp(log_sigma)
        lval = nll_tobit(fva, yva, sigma).item()
        pva = pred_mean(fva, sigma).numpy()
        r2va = 1 - np.sum((y[va] - pva) ** 2) / np.sum((y[va] - y[va].mean()) ** 2)
    hist.append((ep, lval, r2va, sigma.item()))
    if (ep + 1) % 25 == 0 or ep < 3:
        print(f'  ep={ep+1:<3} nll_val={lval:.4f}  R2_val={r2va:.4f}  sigma={sigma.item():.3f}  [t={time.time()-t0:.0f}s]')
    if lval < best_va - 1e-4:
        best_va = lval
        best_state = {k: v.clone() for k, v in net.state_dict().items()}
        best_sigma = sigma.item()
        best_ep = ep + 1
        patience_left = PATIENCE
    else:
        patience_left -= 1
        if patience_left <= 0:
            print(f'  early stop at ep={ep+1} (best ep={best_ep}, nll={best_va:.4f})')
            break

net.load_state_dict(best_state)
sigma = torch.tensor(best_sigma)
net.eval()
with torch.no_grad():
    pte = pred_mean(net(Xte).squeeze(1), sigma).numpy()
    pva = pred_mean(net(Xva).squeeze(1), sigma).numpy()

from sklearn.metrics import r2_score
r2_te = r2_score(y[te], pte)
r2_va = r2_score(y[va], pva)
print(f'\n=== TOBIT RESULT ===')
print(f'best ep={best_ep}  sigma={best_sigma:.4f}  nll_val={best_va:.4f}')
print(f'VAL  R2 = {r2_va:.4f}')
print(f'TEST R2 = {r2_te:.4f}   (baseline v4.2 = 0.4642)')
print(f'  by region: censored n={(y[te] <= C + 1e-6).sum()} R2={r2_score(y[te][y[te] <= C + 1e-6], pte[y[te] <= C + 1e-6]):.4f}')
print(f'             uncensored n={(y[te] > C + 1e-6).sum()} R2={r2_score(y[te][y[te] > C + 1e-6], pte[y[te] > C + 1e-6]):.4f}')
# raw f vs conditional-mean prediction
with torch.no_grad():
    fte = net(Xte).squeeze(1).numpy()
print(f'  raw f(x) test R2 (no censoring correction) = {r2_score(y[te], fte):.4f}')

# multi-seed (3 seeds) for stability
print('\n=== multi-seed (3) ===')
r2s = []
for s in (0, 1, 2):
    torch.manual_seed(1000 + s)
    net2 = nn.Sequential(
        nn.Linear(X_all.shape[1], 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.25),
        nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.25),
        nn.Linear(128, 1))
    ls2 = nn.Parameter(torch.tensor(-0.2))
    opt2 = torch.optim.Adam(list(net2.parameters()) + [ls2], lr=1e-3, weight_decay=1e-5)
    bva, best2 = float('inf'), (0, 0.8, None)
    for ep in range(EPOCHS):
        net2.train()
        idx = torch.randperm(len(ytr))
        for i in range(0, len(idx), BATCH):
            b = idx[i:i + BATCH]
            f = net2(Xtr[b]).squeeze(1)
            sig = torch.exp(ls2)
            loss = nll_tobit(f, ytr[b], sig)
            opt2.zero_grad(); loss.backward(); opt2.step()
        net2.eval()
        with torch.no_grad():
            fv = net2(Xva).squeeze(1)
            sig = torch.exp(ls2)
            lv = nll_tobit(fv, yva, sig).item()
        if lv < bva:
            bva = lv
            best2 = (ep, sig.item(), {k: v.clone() for k, v in net2.state_dict().items()})
        if ep - best2[0] > PATIENCE:
            break
    ep_b, sig_b, st_b = best2
    net2.load_state_dict(st_b)
    net2.eval()
    with torch.no_grad():
        p2 = pred_mean(net2(Xte).squeeze(1), torch.tensor(sig_b)).numpy()
    r2s.append(r2_score(y[te], p2))
    print(f'  seed={s}: ep={ep_b} sigma={sig_b:.3f} test R2={r2s[-1]:.4f}')
print(f'  mean={np.mean(r2s):.4f} sd={np.std(r2s):.4f}')
print('\nDone.')
