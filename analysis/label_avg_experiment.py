#!/usr/bin/env python3
"""
pepADMET label-averaging A/B (route 8): does averaging replicate labels
explain the higher PAMPA/Caco-2 R2 the pepADMET platform (JCIM 2026,
10.1021/acs.jcim.5c02518) reports?

Their Methods (preprocessing step 1): "If the same molecule corresponded to
multiple experimental values, their arithmetic mean was used" (plus
InChIKey-based duplicate removal). We replicate ONLY that label handling and
keep everything else identical to our pipeline: same features, same
leakage-controlled unique-SMILES 70/10/20 split, same MLP + Huber trainer.

A = raw rows (one row per measurement, the committed v4.2 protocol)
B = label-averaged (one row per unique SMILES, y = arithmetic mean of all
    replicate measurements for that SMILES; floor rows enter the mean as-is,
    exactly as pepADMET describes -- no re-censoring)

Endpoints:
  pampa  data/pepadmet_pampa_mdck.csv  PAMPA_MDCK  feats desc+morgan2+molf (3033)
         A @ seed 42 should reproduce the committed baseline R2 0.4642
  caco2  data/pepadmet_caco2.csv       Caco2       feats desc+morgan2 (2265)

Usage: .venv/Scripts/python.exe analysis/label_avg_experiment.py [pampa|caco2|both]
"""
import common  # re-roots CWD to repo root, adds repo root to sys.path

import json
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from train_pepadmet_model import train_endpoint_model, predict_mixed

SEEDS = (42, 123, 7)
FLOOR = -10.0000
EPS = 1e-6

ENDPOINTS = {
    'pampa': dict(name='PAMPA_MDCK', csv='data/pepadmet_pampa_mdck.csv',
                  target='PAMPA_MDCK', cache='_pampa_feat_cache.npz',
                  feats=('desc', 'morgan2', 'molf')),
    'caco2': dict(name='Caco2', csv='data/pepadmet_caco2.csv', target='Caco2',
                  cache='_caco2_feat_cache.npz', feats=('desc', 'morgan2')),
}


def metrics(y_true, pred, floor=FLOOR):
    nf = y_true > floor + EPS
    return dict(
        r2_all=round(float(r2_score(y_true, pred)), 4),
        r2_nonfloor=round(float(r2_score(y_true[nf], pred[nf])), 4),
        spearman_all=round(float(spearmanr(y_true, pred)[0]), 4),
        spearman_nonfloor=round(float(spearmanr(y_true[nf], pred[nf])[0]), 4),
        rmse=round(float(np.sqrt(((y_true - pred) ** 2).mean())), 4),
        mae=round(float(np.mean(np.abs(y_true - pred))), 4),
        n_floor_test=int((y_true <= floor + EPS).sum()),
    )


def run(name, X, y, tr, va, te, seed):
    scaler = StandardScaler().fit(X[tr])
    Xs = scaler.transform(X).astype(np.float32)
    model, _, _ = train_endpoint_model(
        name, Xs, y, tr, va, epochs=80, seed=seed, hidden=(256, 128),
        regression_loss='huber')
    p = predict_mixed(model, Xs[te])[name]
    return metrics(y[te], p)


def main():
    sel = (sys.argv[1] if len(sys.argv) > 1 else 'both')
    eps = ['pampa', 'caco2'] if sel == 'both' else [sel]
    results = {}
    for ep in eps:
        cfg = ENDPOINTS[ep]
        df = pd.read_csv(cfg['csv'])
        smiles = df['smiles'].astype(str).tolist()
        y = df[cfg['target']].to_numpy(dtype=np.float32)
        ok = ~np.isnan(y)
        if not ok.all():
            print(f'{cfg["name"]}: dropping {int((~ok).sum())} NaN-y rows')
            smiles = [s for s, k in zip(smiles, ok) if k]
            y = y[ok]
        z = np.load(cfg['cache'], allow_pickle=False)
        X = np.concatenate([z[f] for f in cfg['feats']], axis=1).astype(np.float32)
        assert X.shape[0] == len(smiles), (X.shape, len(smiles))

        uniq, inv = np.unique(np.asarray(smiles, dtype=object), return_inverse=True)
        n_uniq = len(uniq)

        print(f'\n=== {cfg["name"]}  rows={len(smiles)}  unique_SMILES={n_uniq} '
              f'X={X.shape} ===', flush=True)

        # ---- A: raw rows, group-level split (committed v4.2 protocol) ----
        a = {}
        for seed in SEEDS:
            tr_ids = _split_ids(uniq, inv, seed)
            tr = np.array([i for i in range(len(smiles)) if inv[i] in tr_ids[0]], dtype=np.int64)
            va = np.array([i for i in range(len(smiles)) if inv[i] in tr_ids[1]], dtype=np.int64)
            te = np.array([i for i in range(len(smiles)) if inv[i] in tr_ids[2]], dtype=np.int64)
            if seed == SEEDS[0]:
                print(f'  split: tr={len(tr)} va={len(va)} te={len(te)}', flush=True)
            a[seed] = run(cfg['name'], X, y, tr, va, te, seed)
            print(f'  A(raw)  seed={seed}  R2={a[seed]["r2_all"]}  '
                  f'rho={a[seed]["spearman_all"]}', flush=True)

        # ---- B: label-averaged, one row per unique SMILES ----
        # features are SMILES-derived -> identical within a group; take first row
        first_idx = np.array([np.where(inv == j)[0][0] for j in range(n_uniq)])
        X_b = X[first_idx]
        # sanity: features really constant within groups (check 20 random groups)
        rng = np.random.default_rng(0)
        for j in rng.integers(0, n_uniq, 20):
            rows = np.where(inv == j)[0]
            if len(rows) > 1:
                assert np.array_equal(X[rows[0]], X[rows[1]]), \
                    f'feature mismatch within SMILES group {j}'
        # group-level label stats
        y_floor = y <= FLOOR + EPS
        g_floor_cnt = np.bincount(inv, weights=y_floor.astype(float), minlength=n_uniq)
        g_cnt = np.bincount(inv, minlength=n_uniq)
        g_sum = np.bincount(inv, weights=y, minlength=n_uniq)
        y_b = (g_sum / g_cnt).astype(np.float32)
        mixed = int(((g_floor_cnt > 0) & (g_floor_cnt < g_cnt)).sum())
        lifted = int(((y_b > FLOOR + EPS) & (g_floor_cnt > 0)).sum())
        print(f'  B: unique rows={n_uniq}  (from {len(smiles)}; '
              f'avg {len(smiles)/n_uniq:.2f} replicates/SMILES)', flush=True)
        print(f'  B floor handling: {int(g_floor_cnt.sum())} floor rows '
              f'across {int((g_floor_cnt > 0).sum())} SMILES; '
              f'{mixed} SMILES mixed floor/non-floor; '
              f'{lifted} SMILES lifted above floor by the mean', flush=True)

        b = {}
        for seed in SEEDS:
            tr_ids, va_ids, te_ids = _split_ids(uniq, None, seed)
            # In B-space one row == one unique SMILES, so group id == row id.
            tr = np.array(sorted(tr_ids), dtype=np.int64)
            va = np.array(sorted(va_ids), dtype=np.int64)
            te = np.array(sorted(te_ids), dtype=np.int64)
            b[seed] = run(cfg['name'], X_b, y_b, tr, va, te, seed)
            print(f'  B(avg)  seed={seed}  R2={b[seed]["r2_all"]}  '
                  f'rho={b[seed]["spearman_all"]}', flush=True)

        def agg(d):
            keys = ('r2_all', 'r2_nonfloor', 'spearman_all', 'spearman_nonfloor', 'rmse', 'mae')
            out = {}
            for k in keys:
                vals = [d[s][k] for s in SEEDS]
                out[k] = dict(mean=round(float(np.mean(vals)), 4),
                              std=round(float(np.std(vals, ddof=0)), 4),
                              per_seed=list(vals))
            return out

        results[ep] = dict(
            endpoint=cfg['name'], rows=len(smiles), unique_smiles=n_uniq,
            avg_replicates=round(len(smiles) / n_uniq, 2),
            mixed_floor_smiles=mixed, floor_lifted_smiles=lifted,
            A_raw=agg(a), B_label_avg=agg(b),
            delta_r2_all=round(float(np.mean([b[s]['r2_all'] - a[s]['r2_all'] for s in SEEDS])), 4),
        )
        print(f'  DELTA R2(B-A) mean over seeds = {results[ep]["delta_r2_all"]:+}', flush=True)

    with open('analysis/label_avg_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('\nWrote analysis/label_avg_results.json')
    print('Done.')


def _split_ids(uniq, inv, seed, row_mode=True):
    """Same assignment as common.split_smiles (seed -> permutation of uniq)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    n_tr = int(round(len(uniq) * 0.70))
    n_va = int(round(len(uniq) * 0.10))
    tr_ids = set(perm[:n_tr].tolist())
    va_ids = set(perm[n_tr:n_tr + n_va].tolist())
    return tr_ids, va_ids, set(range(len(uniq))) - tr_ids - va_ids


if __name__ == '__main__':
    import sys
    main()
