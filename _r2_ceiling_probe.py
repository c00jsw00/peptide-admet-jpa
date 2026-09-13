#!/usr/bin/env python3
"""Diagnostic: how high can R2 go on Half_life / Caco2 / PAMPA_MDCK?

For each endpoint we load the SAME prepared data + SAME leakage-controlled
split as the shipped trainer, then fit a battery of models and report
train / val / test R2.  If a much larger model still plateaus near the
shipped R2 on test while reaching ~1.0 on train, the ceiling is data
noise / generalization, not model capacity.  Nothing is shipped or
overwritten — this is a read-only probe.
"""
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge

sys.path.insert(0, '.')
from endpoint_config import ENDPOINT_BY_NAME, ESMC_DIM, esmc_cache_path
from feature_extractor import molecule_features, sequence_features
from train_pepadmet_model import (load_endpoint, split_sequence,
                                  split_molecular, load_esmc_embeddings,
                                  train_endpoint_model, PREPARED_CSV, SLUGS)
from admet_model import MixedADMETMLP, predict_mixed

ENDPOINTS = ['Half_life', 'Caco2', 'PAMPA_MDCK']


def wide_mlp(Xtr, ytr, Xv, yv, seed=42, epochs=300, hidden=(1024, 512, 256)):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = nn.Sequential(
        nn.Linear(Xtr.shape[1], hidden[0]), nn.BatchNorm1d(hidden[0]), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(hidden[0], hidden[1]), nn.BatchNorm1d(hidden[1]), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(hidden[1], hidden[2]), nn.BatchNorm1d(hidden[2]), nn.ReLU(),
        nn.Linear(hidden[2], 1))
    X, y = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    Xv_t = torch.from_numpy(Xv)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=1e-4)
    best, best_state, bad = float('inf'), None, 0
    for ep in range(1, epochs + 1):
        m.train()
        perm = torch.randperm(len(X))
        tot = 0.0
        for i in range(0, len(X), 256):
            b = perm[i:i + 256]
            opt.zero_grad()
            loss = nn.functional.mse_loss(m(X[b]).squeeze(1), y[b])
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        with torch.no_grad():
            m.eval()
            vloss = nn.functional.mse_loss(m(Xv_t).squeeze(1),
                                           torch.from_numpy(yv)).item()
        if vloss < best - 1e-6:
            best, bad = vloss, 0
            best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= 15:
                break
    if best_state:
        m.load_state_dict(best_state)
    m.eval()
    return m


def mlp_r2(m, Xs, y):
    with torch.no_grad():
        p = m(torch.from_numpy(Xs)).squeeze(1).numpy()
    return r2_score(y, p)


def main():
    report = {}
    for name in ENDPOINTS:
        t0 = time.time()
        ep = ENDPOINT_BY_NAME[name]
        X, y, keys, key_label, n = load_endpoint(name, PREPARED_CSV[name])
        if getattr(ep, 'esmc', False):
            emb = load_esmc_embeddings(name, keys, n)
            X = np.hstack([X, emb]).astype(np.float32)

        if ep.modality == 'sequence':
            tr, va, te, audit = split_sequence(n, keys, y, seed=42)
        else:
            tr, va, te, audit = split_molecular(n, keys, seed=42)
        print(f'== {name}  n={n}  D={X.shape[1]}  '
              f'train={len(tr)} val={len(va)} test={len(te)}')

        sc = StandardScaler().fit(X[tr])
        Xs = sc.transform(X).astype(np.float32)
        yv = y.astype(np.float64)

        rows = {}
        # A) shipped architecture (reproduces metrics.json)
        model, device, _ = train_endpoint_model(name, Xs, y, tr, va, epochs=80, seed=42)
        pred = predict_mixed(model, Xs)[name]
        rows['A shipped MLP (256,128)'] = {
            'train': r2_score(yv[tr], pred[tr]),
            'val': r2_score(yv[va], pred[va]),
            'test': r2_score(yv[te], pred[te]),
        }
        print(f'   A shipped      train {rows["A shipped MLP (256,128)"]["train"]:.4f} '
              f'val {rows["A shipped MLP (256,128)"]["val"]:.4f} '
              f'test {rows["A shipped MLP (256,128)"]["test"]:.4f}  '
              f'({time.time()-t0:.0f}s)')

        # B) wide MLP (1024,512,256), 300 ep, early stop on val
        wb = wide_mlp(Xs[tr], y[tr], Xs[va], yv[va])
        rows['B wide MLP (1024,512,256)'] = {
            'train': mlp_r2(wb, Xs[tr], yv[tr]),
            'val': mlp_r2(wb, Xs[va], yv[va]),
            'test': mlp_r2(wb, Xs[te], yv[te]),
        }
        print(f'   B wide-MLP     train {rows["B wide MLP (1024,512,256)"]["train"]:.4f} '
              f'val {rows["B wide MLP (1024,512,256)"]["val"]:.4f} '
              f'test {rows["B wide MLP (1024,512,256)"]["test"]:.4f}  '
              f'({time.time()-t0:.0f}s)')

        # C) HistGBDT
        gb = HistGradientBoostingRegressor(
            max_iter=3000, learning_rate=0.05, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=42, early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=50).fit(Xs[tr], yv[tr])
        rows['C HistGBDT (3000 it)'] = {
            'train': r2_score(yv[tr], gb.predict(Xs[tr])),
            'val': r2_score(yv[va], gb.predict(Xs[va])),
            'test': r2_score(yv[te], gb.predict(Xs[te])),
        }
        print(f'   C HistGBDT     train {rows["C HistGBDT (3000 it)"]["train"]:.4f} '
              f'val {rows["C HistGBDT (3000 it)"]["val"]:.4f} '
              f'test {rows["C HistGBDT (3000 it)"]["test"]:.4f}  '
              f'({time.time()-t0:.0f}s)')

        # D) KernelRidge (RBF) — strong over-param probe; O(n^2), fine for n<=9k
        try:
            kr = KernelRidge(alpha=1e-6, kernel='rbf', gamma='scale',
                             kernel_params={'gamma': 0.1 / Xs.shape[1]}).fit(Xs[tr], yv[tr])
            rows['D KernelRidge RBF'] = {
                'train': r2_score(yv[tr], kr.predict(Xs[tr])),
                'val': r2_score(yv[va], kr.predict(Xs[va])),
                'test': r2_score(yv[te], kr.predict(Xs[te])),
            }
            print(f'   D KernelRidge  train {rows["D KernelRidge RBF"]["train"]:.4f} '
                  f'val {rows["D KernelRidge RBF"]["val"]:.4f} '
                  f'test {rows["D KernelRidge RBF"]["test"]:.4f}  '
                  f'({time.time()-t0:.0f}s)')
        except Exception as e:
            print(f'   D KernelRidge  failed: {e}')

        # noise floor estimate: duplicate-family analysis for sequence endpoint
        if ep.modality == 'sequence':
            print(f'   y var={yv.var():.4f}  rmse(A,test)={np.sqrt(mean_squared_error(yv[te], pred[te])):.4f}')
        report[name] = rows
    print('\n===== SUMMARY =====')
    for name, rows in report.items():
        print(f'{name}:')
        for k, v in rows.items():
            print(f'  {k:32s} train={v["train"]:.4f} val={v["val"]:.4f} test={v["test"]:.4f}')


if __name__ == '__main__':
    main()
