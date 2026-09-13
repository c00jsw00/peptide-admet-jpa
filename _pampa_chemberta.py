#!/usr/bin/env python3
"""
_pampa_chemberta.py — ChemBERTa vs MoLFormer for PAMPA/Caco-2, same split.

Motivated by PeptiVerse (Nat Commun 2026): embedding choice dominates
architecture; ChemBERTa (PubChem-trained) beat peptide-specific CLMs on
PAMPA (Spearman rho 0.69 vs 0.59) and Caco-2 (0.80 vs 0.75).

Uses the EXACT pipeline split (split_molecular verbatim, seed 42) and the
same MLP/Huber training as train_pepadmet_model.py, so numbers are directly
comparable to the committed v4.2 metrics (PAMPA 0.4642, Caco-2 0.3909).

For each endpoint we compare, on the same test set:
  A. v4.2 baseline:  RDKit+Morgan (2265) + MoLFormer (768)        = 3033
  B. ChemBERTa swap: RDKit+Morgan (2265) + ChemBERTa (384)        = 2649
  C. Both concat:    RDKit+Morgan (2265) + MoLFormer + ChemBERTa  = 3417

Reports test R2 AND Spearman rho (rho comparable to PeptiVerse).
"""
import common  # CWD -> repo root, sys.path
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from feature_extractor import molecule_features

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

FLOOR = -10.0 + 1e-9


def split_molecular(n, keys, seed=42):
    """VERBATIM copy of train_pepadmet_model.split_molecular (random 70/10/20
    on UNIQUE SMILES, duplicates grouped, seed 42)."""
    uniq, inv = np.unique(np.asarray(keys, dtype=object), return_inverse=True)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    n_tr = int(round(len(uniq) * 0.70))
    n_va = int(round(len(uniq) * 0.10))
    tr_ids = set(perm[:n_tr].tolist())
    va_ids = set(perm[n_tr:n_tr + n_va].tolist())
    tr = np.array([i for i in range(n) if inv[i] in tr_ids], dtype=np.int64)
    va = np.array([i for i in range(n) if inv[i] in va_ids], dtype=np.int64)
    te = np.array([i for i in range(n) if inv[i] not in tr_ids and inv[i] not in va_ids],
                  dtype=np.int64)
    return tr, va, te


def build_mlp(d, hidden=(256, 128), dropout=0.1):
    layers, prev = [], d
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


def train_mlp(Xtr, ytr, Xva, yva, Xte, yte, hidden=(256, 128), lr=1e-3, epochs=300,
              patience=40, batch=256, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    m = build_mlp(Xtr.shape[1], hidden)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.HuberLoss(delta=1.0)
    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr).unsqueeze(1)
    Xv, yv = torch.tensor(Xva), torch.tensor(yva).unsqueeze(1)
    best, best_st, wait = -9e9, None, 0
    for ep in range(epochs):
        m.train()
        idx = torch.randperm(len(Xt))
        for s in range(0, len(Xt), batch):
            b = idx[s:s+batch]
            opt.zero_grad()
            loss = crit(m(Xt[b]), yt[b]); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            vloss = crit(m(Xv), yv).item()
        if vloss < best - 1e-5:
            best, best_st, wait = vloss, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    m.load_state_dict(best_st)
    m.eval()
    with torch.no_grad():
        pte = m(torch.tensor(Xte)).squeeze(1).numpy()
    return pte


def main():
    import pandas as pd
    for slug, label, base_r2 in [('pampa_mdck', 'PAMPA_MDCK', 0.4642),
                                  ('caco2', 'Caco-2', 0.3909)]:
        print(f'\n{"="*64}\n{slug.upper()}  (v4.2 baseline test R2 = {base_r2})\n{"="*64}')
        df = pd.read_csv(f'data/pepadmet_{slug}.csv')
        y = df[label].to_numpy(dtype=np.float64)
        keys = df['smiles'].astype(str).tolist()
        n = len(df)
        print(f'rows={n}')

        # --- RDKit+Morgan (2265) — cached via common cache if present ---
        import time; t0 = time.time()
        Xr = molecule_features(keys).astype(np.float32)
        print(f'RDKit+Morgan {Xr.shape}  ({time.time()-t0:.0f}s)')

        zmol = np.load('data/molformer/molformer_emb_%s.npz' % slug)
        emb_mol = np.asarray(zmol['emb'], dtype=np.float32)
        emb_mol_keys = np.asarray(zmol['keys'], dtype=object)
        assert (emb_mol_keys == np.asarray(keys, dtype=object)).all(), 'MoLFormer order drift'

        zcb = np.load('data/chemberta/chemberta_emb_%s.npz' % slug)
        emb_cb = np.asarray(zcb['emb'], dtype=np.float32)
        emb_cb_keys = np.asarray(zcb['keys'], dtype=object)
        assert (emb_cb_keys == np.asarray(keys, dtype=object)).all(), 'ChemBERTa order drift'

        tr, va, te = split_molecular(n, keys, y, seed=42)
        print(f'split: tr={len(tr)} va={len(va)} te={len(te)}')

        for name, X in [
            ('A. v4.2 MoLFormer   3033', np.hstack([Xr, emb_mol]).astype(np.float32)),
            ('B. ChemBERTa swap  2649', np.hstack([Xr, emb_cb]).astype(np.float32)),
            ('C. Both concat     3417', np.hstack([Xr, emb_mol, emb_cb]).astype(np.float32)),
        ]:
            sc = StandardScaler().fit(X[tr])
            pte = train_mlp(sc.transform(X[tr]), y[tr], sc.transform(X[va]), y[va],
                            sc.transform(X[te]), y[te])
            r2 = r2_score(y[te], pte)
            rho, _ = spearmanr(y[te], pte)
            print(f'  {name}:  R2={r2:.4f}   Spearman rho={rho:.4f}')


if __name__ == '__main__':
    main()
