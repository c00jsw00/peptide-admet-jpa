#!/usr/bin/env python3
"""
ChemBERTa embedding experiment (c3): does PeptiVerse's #1 PAMPA embedding
beat our frozen MoLFormer-XL on the molecular endpoints?

PeptiVerse (Nat. Commun. 2026, s41467-026-74167-w) reports embedding choice
dominates architecture, with ChemBERTa beating PeptideCLM on PAMPA
(Spearman 0.69 vs 0.59). We test the analogous question on OUR data:
same pipeline (verbatim split seed 42, verbatim training loop imported from
train_pepadmet_model.train_endpoint_model, Huber d=1.0, Adam lr=1e-3 wd=1e-5,
ReduceLROnPlateau, early-stop patience 10, hidden 256/128, dropout 0.25,
batch 128, max 80 epochs) across feature configurations and 3 seeds.

Configurations (molecule = 2265-dim RDKit 2D + Morgan r2, as in the pipeline):
  A  mol + MoLFormer-XL  (3033)  <- committed v4.2 configuration (re-run)
  B  mol + ChemBERTa     (2649)  <- the new weapon under test
  C  mol + MoLFormer + ChemBERTa (3417)
  D  ChemBERTa alone     ( 384)  <- reference (cf. round1: MoLFormer alone 0.3991)

Seeds: 42 (pipeline default), 123, 7.  The honest baseline is A's own
re-run (re-training noise is ~+/-0.01 R2, see round1: 0.4505 vs committed
0.4642), not the committed number.

Endpoints (both have a left-censoring floor at y = -10.0000 in logPapp):
  pampa  data/pepadmet_pampa_mdck.csv  target PAMPA_MDCK  (7283 rows)
  caco2  data/pepadmet_caco2.csv       target Caco2       (7429 rows)
The prepared CSVs store targets already in model space (logPapp), so y is
used verbatim — matching load_endpoint exactly (identity transform).

Reported per run: test R2 (overall), R2 on the non-floor subset (the fair
model-quality number given the censored floor), RMSE, MAE.
Results -> analysis/chemberta_results[_caco2].json

Usage:  python analysis/chemberta_retrain.py [pampa|caco2]   (default pampa)
"""
import json
import sys
import time

import common  # re-roots CWD to repo root, adds repo root to sys.path

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

import feature_extractor  # noqa: E402
from train_pepadmet_model import train_endpoint_model, predict_mixed  # noqa: E402

SEEDS = (42, 123, 7)
FLOOR = -10.0000  # left-censoring floor of the assays (logPapp), both endpoints

ENDPOINTS = {
    'pampa': dict(
        name='PAMPA_MDCK',
        csv='data/pepadmet_pampa_mdck.csv',
        target='PAMPA_MDCK',
        feat_cache=common.FEAT_CACHE,  # _pampa_feat_cache.npz (already built)
        molf='data/molformer/molformer_emb_pampa_mdck.npz',
        chem='data/chemberta/chemberta_emb_pampa_mdck.npz',
        out='analysis/chemberta_results.json',
    ),
    'caco2': dict(
        name='Caco2',
        csv='data/pepadmet_caco2.csv',
        target='Caco2',
        feat_cache='_caco2_feat_cache.npz',
        molf='data/molformer/molformer_emb_caco2.npz',
        chem='data/chemberta/chemberta_emb_caco2.npz',
        out='analysis/chemberta_results_caco2.json',
    ),
}


def build_feat_cache(path, smiles):
    """RDKit 2D descriptors (217) + Morgan r=2 (2048), same code as round2."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, DataStructs, AllChem as RFG
    names = [nm for (nm, _fn) in Descriptors._descList]
    D = len(names)
    N = len(smiles)
    X_desc = np.zeros((N, D), dtype=np.float64)
    t0 = time.time()
    for i, s in enumerate(smiles):
        try:
            mol = Chem.MolFromSmiles(str(s))
        except Exception:
            continue
        if mol is None:
            continue
        try:
            d = Descriptors.CalcMolDescriptors(mol)
            for j, nm in enumerate(names):
                v = d.get(nm)
                if isinstance(v, (int, float)) and np.isfinite(v):
                    X_desc[i, j] = float(v)
        except Exception:
            pass
        if (i + 1) % 1000 == 0:
            print(f'    desc {i + 1}/{N} ({time.time() - t0:.0f}s)', flush=True)
    gen = RFG.GetMorganGenerator(radius=2, fpSize=2048)
    m2 = np.zeros((N, 2048), dtype=np.float64)
    for i, s in enumerate(smiles):
        try:
            mol = Chem.MolFromSmiles(str(s))
            if mol is None:
                continue
            fp = gen.GetFingerprint(mol)
            DataStructs.ConvertToNumpyArray(fp, m2[i])
        except Exception:
            pass
    np.savez(path, desc=X_desc, morgan2=m2)
    print(f'  feature cache built -> {path} in {time.time() - t0:.0f}s')


def load_emb(path, n, tag):
    z = np.load(path, allow_pickle=True)
    emb, keys = np.asarray(z['emb'], dtype=np.float32), np.asarray(z['keys'], dtype=object)
    if emb.shape != (n, 384 if 'chemberta' in path else 768):
        raise ValueError(f'{tag}: shape {emb.shape}')
    if not np.isfinite(emb).all():
        raise ValueError(f'{tag}: non-finite values')
    return emb


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else 'pampa'
    cfg_ep = ENDPOINTS[which]
    name = cfg_ep['name']

    df = pd.read_csv(cfg_ep['csv'])
    smiles = df['smiles'].astype(str).tolist()
    y = df[cfg_ep['target']].to_numpy(dtype=np.float32)  # pipeline trains float32
    n = len(y)
    tr, va, te = common.split_smiles(smiles)
    print(f'[{name}] N={n}  train={len(tr)}  val={len(va)}  test={len(te)}  '
          f'test floor rows={int((y[te] <= FLOOR + 1e-6).sum())}', flush=True)

    # --- 2D features (RDKit 217 + Morgan r2 2048 = 2265), row-aligned with CSV ---
    t0 = time.time()
    if not __import__('os').path.exists(cfg_ep['feat_cache']):
        print(f'  no {cfg_ep["feat_cache"]} — building (~5-10 min)...')
        build_feat_cache(cfg_ep['feat_cache'], smiles)
    zc = np.load(cfg_ep['feat_cache'], allow_pickle=False)
    Xmol = np.hstack([zc['desc'], zc['morgan2']]).astype(np.float32)
    # verify against a fresh computation on a sample.  Known tolerance: the two
    # Ipc columns are ~1e23-magnitude and RDKit is not bitwise-deterministic
    # there (rel ~1e-2); after StandardScaler this is a ~0.01-unit perturbation
    # on 2 of 2265 dims — negligible at the +/-0.01 R2 noise level.
    sample = smiles[:64]
    Xf = feature_extractor.molecule_features(sample)
    d_diff = np.abs(zc['desc'][:64] - Xf[:, :217])
    m_diff = np.abs(zc['morgan2'][:64] - Xf[:, 217:2265])
    desc_bad = int((d_diff.max(axis=0) > 1e-8).sum())
    morgan_bad = int((m_diff.max(axis=0) > 1e-8).sum())
    assert morgan_bad == 0, f'morgan cache mismatch ({morgan_bad} cols)'
    assert desc_bad <= 5, f'descriptor cache mismatch ({desc_bad} cols)'
    if desc_bad:
        bad_cols = np.where(d_diff.max(axis=0) > 1e-8)[0]
        print(f'  NOTE: {desc_bad} desc cols differ at float noise level: '
              f'{[feature_extractor.descriptor_names()[j] for j in bad_cols]} '
              f'(Ipc ~1e23, rel ~1e-2; negligible after scaling)')
    print(f'feature cache ready: {Xmol.shape} in {time.time() - t0:.0f}s', flush=True)

    # --- LM embeddings (frozen, generated in CSV row order; verify order) ---
    molf = load_emb(cfg_ep['molf'], n, 'molformer')
    chem = load_emb(cfg_ep['chem'], n, 'chemberta')
    for path, emb, tag in ((cfg_ep['molf'], molf, 'molformer'),
                           (cfg_ep['chem'], chem, 'chemberta')):
        z = np.load(path, allow_pickle=True)
        assert np.array_equal(np.asarray(z['keys'], dtype=object),
                              np.asarray(smiles, dtype=object)), f'{tag} row-order mismatch'
    print(f'embeddings ready: molformer {molf.shape}, chemberta {chem.shape}', flush=True)

    configs = {
        'A_mol_molf':    np.hstack([Xmol, molf]).astype(np.float32),
        'B_mol_chem':    np.hstack([Xmol, chem]).astype(np.float32),
        'C_mol_molf_chem': np.hstack([Xmol, molf, chem]).astype(np.float32),
        'D_chem':        chem.astype(np.float32),
    }

    results = []
    for cfg_name, X in configs.items():
        for seed in SEEDS:
            t0 = time.time()
            scaler = StandardScaler().fit(X[tr])
            Xs = scaler.transform(X).astype(np.float32)
            model, device, _ = train_endpoint_model(
                name, Xs, y, tr, va, epochs=80, seed=seed, hidden=(256, 128),
                regression_loss='huber')
            pred = predict_mixed(model, Xs[te])[name]
            r2_all = float(r2_score(y[te], pred))
            nf = y[te] > FLOOR + 1e-6
            r2_nf = float(r2_score(y[te][nf], pred[nf]))
            rec = {'config': cfg_name, 'seed': seed, 'input_dim': int(X.shape[1]),
                   'test_r2': round(r2_all, 4), 'test_r2_nonfloor': round(r2_nf, 4),
                   'rmse': round(float(np.sqrt(((y[te] - pred) ** 2).mean())), 4),
                   'mae': round(float(np.mean(np.abs(y[te] - pred))), 4),
                   'seconds': round(time.time() - t0, 1)}
            results.append(rec)
            print(f'{cfg_name}  seed={seed}:  test R2={r2_all:.4f}  '
                  f'non-floor R2={r2_nf:.4f}  RMSE={rec["rmse"]}  '
                  f'MAE={rec["mae"]}  ({rec["seconds"]}s)', flush=True)

    out = cfg_ep['out']
    with open(out, 'w') as f:
        json.dump({'endpoint': name, 'seeds': list(SEEDS),
                   'configs': list(configs.keys()),
                   'run': time.strftime('%Y-%m-%d %H:%M:%S'), 'results': results},
                  f, indent=2)
    print(f'\nWrote {out}')
    print('Done.')


if __name__ == '__main__':
    main()
