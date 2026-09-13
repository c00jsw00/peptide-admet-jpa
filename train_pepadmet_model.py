#!/usr/bin/env python3
"""
train_pepadmet_model.py
=======================

Train the **v4.1 four-endpoint, real-data + ESMC** peptide ADMET models.

v4.1 (2026-08-26) — ESMC-600M edition
-------------------------------------
Extends v4.0 (Chemit797/PepADMET-Dataset) with **frozen ESMC-600M** (Biohub
ESM Cambrian) sequence embeddings for the two sequence-modality endpoints.
The 1152-dim mean-pooled embedding is *concatenated* to the classical
428-dim sequence vector (AAC+DPC+physchem) -> 1580-dim model input.  The
embeddings are precomputed offline (``esmc_embed.py`` in the Python>=3.12
``.venv-esmc`` env) and cached in ``data/esmc/*.npz``; training and
inference therefore stay on CPU with **no ESMC dependency** — the cache is
verified (shape, finiteness, row order) before use.  Caco2 / PAMPA_MDCK are
unchanged (their source "sequences" are non-standard peptidomimetic residue
lists, ~0.2% standard AA, so they are not embeddable).

v4.0 (2026-08-25) — real-data edition
-------------------------------------
Replaces the v3.0 synthetic demo with the four endpoints requested from the
**Chemit797/PepADMET-Dataset** release, each trained as a focused single-task
``MixedADMETMLP`` (the class is generic: ``endpoints=[name]`` builds a
one-head model) matched to the modality its data actually has:

  ====================  ========  =================  ===========================
  endpoint              kind      modality           features
  ====================  ========  =================  ===========================
  Hemolysis             binary    sequence           1580-dim = 428 classical
                                                     (20 AAC + 400 DPC + 8
                                                     physchem) + 1152 frozen
                                                     ESMC-600M embedding
  Half_life             reg       sequence           1580-dim, same layout,
                                                     target = log10(half-life, s)
  Caco2                 reg       molecular          2265-dim (217 RDKit
                                                     2D descriptors +
                                                     2048-bit Morgan r=2)
  PAMPA_MDCK            reg       molecular          2265-dim, same as Caco2
  ====================  ========  =================  ===========================

Why four separate models instead of one shared-trunk multi-task model: the
four datasets are **disjoint molecules** in **two different feature spaces**
(sequence vs. molecular).  A shared trunk would learn two disconnected
subspaces with zero-padding and no cross-task benefit — four focused models
are cleaner, each is fully interpretable, and every metric is measured
end-to-end on its own holdout.

Splits (leakage control)
------------------------
* **sequence endpoints** (Hemolysis, Half_life): AMPBench-MT-style
  **homology-controlled** split — 3-mer Jaccard families (threshold 0.35)
  are assigned whole to train/val/test, so a near-duplicate sequence can
  never cross the boundary.  A plain random 70/10/20 split is trained
  alongside as an *inflated* comparison, so the cost of the homology
  control is visible (arXiv:2607.25518).
* **molecular endpoints** (Caco2, PAMPA_MDCK): the dataset ships no
  sequences (only non-standard residue-name lists), so a 3-mer homology
  split is not computable.  We use a **random 70/10/20 split on unique
  SMILES** (exact-duplicate SMILES are grouped into one split, so no
  identical molecule crosses the boundary) — an honest split, but weaker
  than homology control; this limitation is stated in the metrics file.

All metrics are **measured** at train time on the holdout sets; nothing is
hardcoded.  Regression targets use each endpoint's ``target_transform``
(Half_life: log10; Caco2/PAMPA: identity — their source columns are
already log-scale logPapp values).

Artifacts
---------
``models_v4/<slug>/admet_mlp.pt``   per-endpoint model weights + rebuild meta
``models_v4/<slug>/scaler.pt``      StandardScaler (fit on train only)
``models_v4/<slug>/metrics.json``   per-endpoint metrics (both splits)
``models_v4/summary.json``          cross-endpoint headline

Usage
-----
    python prepare_pepadmet_data.py            # 1) build the 4 prepared CSVs
    python train_pepadmet_model.py             # 2) train + evaluate all 4
    python train_pepadmet_model.py --endpoints Hemolysis Half_life
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, matthews_corrcoef,
                             mean_squared_error, r2_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from admet_model import MixedADMETMLP, predict_mixed, save_mixed_model
from endpoint_config import (ENDPOINTS, ENDPOINT_BY_NAME, KIND_BINARY,
                             ESMC_MODEL, ESMC_DIM, esmc_cache_path,
                             MOLEFORMER_MODEL, MOLEFORMER_DIM,
                             molformer_cache_path)
from feature_extractor import molecule_features, sequence_features
from homology_split import greedy_kmer_clusters, split_by_family, leakage_audit, kmer_counts

SLUGS = {ep.name: ep.name.lower().replace(' ', '_') for ep in ENDPOINTS}
PREPARED_CSV = {'Hemolysis': 'data/pepadmet_hemolysis.csv',
                'Half_life': 'data/pepadmet_half_life.csv',
                'Caco2': 'data/pepadmet_caco2.csv',
                'PAMPA_MDCK': 'data/pepadmet_pampa_mdck.csv'}


# --------------------------------------------------------------------------- #
# ESMC-600M embedding cache (v4.1)
# --------------------------------------------------------------------------- #
def load_esmc_embeddings(name: str, csv_seqs, n: int) -> np.ndarray:
    """Load the frozen ESMC-600M embeddings for this endpoint's sequences.

    The cache (``data/esmc/esmc_emb_<slug>.npz``) stores (emb (N,1152)
    float32, sequences, meta) in prepared-CSV row order.  We verify finiteness
    and then select the rows for ``csv_seqs`` **by sequence key**: sequences
    are unique within a prepared CSV, and the ESMC embedding is a pure function
    of the sequence, so a key lookup is exact even when the requested order
    differs from the cache order (e.g. after the v4.2 Half-life dedup collapses
    duplicate sequences to one row each).  This is the only safe re-join and it
    is exact because sequences are unique keys.
    """
    path = esmc_cache_path(name)
    if not Path(path).exists():
        raise FileNotFoundError(
            f'{name}: ESMC cache missing at {path} — '
            f'run `.venv-esmc/Scripts/python.exe esmc_embed.py` first')
    z = np.load(path, allow_pickle=True)
    emb, emb_seq = z['emb'], z['sequences']
    emb_seq = np.asarray(emb_seq, dtype=object)
    emb = np.asarray(emb, dtype=np.float32)
    if emb.ndim != 2 or emb.shape[1] != ESMC_DIM:
        raise ValueError(
            f'{name}: ESMC cache shape {emb.shape} != expected (?, {ESMC_DIM}) '
            f'— regenerate with esmc_embed.py')
    if not np.isfinite(emb).all():
        raise ValueError(f'{name}: ESMC cache contains non-finite values')
    # key -> row (first occurrence; all rows of a repeated sequence carry the
    # same embedding, so the choice is immaterial)
    idx = {str(s): i for i, s in enumerate(emb_seq)}
    csv_arr = [str(s) for s in csv_seqs]
    missing = [s for s in csv_arr if s not in idx]
    if missing:
        raise ValueError(f'{name}: {len(missing)} sequences in the request '
                         f'are missing from the ESMC cache (e.g. {missing[:3]})')
    out = emb[[idx[s] for s in csv_arr]]
    if out.shape[0] != n:
        raise ValueError(f'{name}: selected {out.shape[0]} ESMC rows for {n} '
                         f'requested')
    return out


# --------------------------------------------------------------------------- #
# MoLFormer-XL embedding cache (v4.2)
# --------------------------------------------------------------------------- #
def load_molformer_embeddings(name: str, csv_smiles, n: int) -> np.ndarray:
    """Load the frozen MoLFormer-XL CLS embeddings cached for this endpoint.

    The cache (``data/molformer/molformer_emb_<slug>.npz``) stores (emb (N,768)
    float32, keys=SMILES, meta) in prepared-CSV row order.  We verify the row
    count, finiteness, and — defensively — the exact SMILES order; if the order
    ever drifts we reindex by SMILES.  (Unlike sequences, SMILES are NOT unique
    within a prepared CSV — exact-duplicate SMILES appear — so a re-join by key
    is only exact when the *order* matches; a drift therefore raises unless it
    is resolvable, and we keep the strict order check as the primary path.)
    """
    path = molformer_cache_path(name)
    if not Path(path).exists():
        raise FileNotFoundError(
            f'{name}: MoLFormer cache missing at {path} — '
            f'run `.venv/Scripts/python.exe molformer_embed.py` first')
    z = np.load(path, allow_pickle=True)
    emb, emb_keys = z['emb'], z['keys']
    emb_keys = np.asarray(emb_keys, dtype=object)
    emb = np.asarray(emb, dtype=np.float32)
    if emb.shape != (n, MOLEFORMER_DIM):
        raise ValueError(
            f'{name}: MoLFormer cache shape {emb.shape} != expected '
            f'({n}, {MOLEFORMER_DIM}) — regenerate with molformer_embed.py')
    if not np.isfinite(emb).all():
        raise ValueError(f'{name}: MoLFormer cache contains non-finite values')
    csv_arr = np.asarray(csv_smiles, dtype=object)
    if not (emb_keys.shape == csv_arr.shape and np.array_equal(emb_keys, csv_arr)):
        # SMILES are not unique keys here (duplicate SMILES exist), so a key
        # re-join is ambiguous; require an exact order match instead.
        raise ValueError(
            f'{name}: MoLFormer cache row order does not match the prepared '
            f'CSV (SMILES are not unique keys, so a re-join is ambiguous) — '
            f'regenerate with molformer_embed.py against the current CSV')
    return emb


# --------------------------------------------------------------------------- #
# Data loading + features
# --------------------------------------------------------------------------- #
def load_endpoint(name: str, csv_path: str):
    """Load one prepared CSV and return (X, y, split_keys).

    X : (N, D) float32 features (modality-appropriate)
    y : (N,) float64 target in *model space* (transformed)
    split_keys : what to split on — 'sequence' for sequence endpoints,
                 'smiles' (unique-molecule) for molecular endpoints.
    """
    ep = ENDPOINT_BY_NAME[name]
    df = pd.read_csv(csv_path)
    n = len(df)
    print(f'  [{name}] {ep.modality:9s} {ep.kind:10s} n={n}')

    if ep.modality == 'sequence':
        seqs = df['sequence'].astype(str).tolist()
        X = sequence_features(seqs).astype(np.float32)
        keys = seqs
        key_label = 'sequence'
        raw = df[name].to_numpy(dtype=np.float64)
    else:  # molecular
        smiles = df['smiles'].astype(str).tolist()
        X = molecule_features(smiles).astype(np.float32)
        keys = smiles
        key_label = 'smiles'
        raw = df[name].to_numpy(dtype=np.float64)

    assert np.isfinite(X).all(), f'{name}: non-finite features'

    if ep.target_transform == 'log10':
        y = np.log10(raw)
    else:
        y = raw
    y = y.astype(np.float32)
    return X, y, keys, key_label, n


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
def _kmer_signature(seq, k=3):
    """Order-insensitive, count-sensitive 3-mer multiset signature.

    Two sequences have 3-mer Jaccard == 1.0 iff their 3-mer multisets are
    identical iff they share this signature.  Collapsing rows by signature
    guarantees no jaccard-1.0 pair can be split across train/val/test.
    """
    return tuple(sorted(kmer_counts(seq, k).items()))


def split_sequence(n, keys, y, seed, threshold=0.35):
    """Homology-controlled 70/10/20 split on 3-mer Jaccard families.

    Two leakage controls are applied before the family split:
      1. **Canonical 3-mer-multiset signature** — rows whose 3-mer multisets
         are identical (Jaccard 1.0, i.e. the model can memorise one to answer
         the other) are collapsed to a single family, so they can never cross
         a boundary.  This is stronger than collapsing exact-string duplicates
         because an anagram that shares every 3-mer (e.g. ``FRRFFKWPRRPFKFF``
         vs ``FRRPFKWPRRFFKFF``) is also merged.
      2. **Greedy 3-mer-Jaccard clustering** (single-linkage, threshold
         ``threshold``) on the unique signatures groups near-duplicates into
         families, which are then assigned to train/val/test at the family
         level (AMPBench-MT-style control, arXiv:2607.25518).
    """
    sigs = [_kmer_signature(s) for s in keys]
    uniq_sig, inv_sig = np.unique(np.asarray(sigs, dtype=object),
                                  return_inverse=True)
    # one representative (canonical) sequence per unique signature
    rep = {}
    for i, s in enumerate(keys):
        rep.setdefault(tuple(sigs[i]), s)
    reps = [rep[tuple(u)] for u in uniq_sig]

    fam_uniq = greedy_kmer_clusters(reps, threshold=threshold, k=3)
    fam = fam_uniq[inv_sig]  # per-row family id
    mask = split_by_family(fam, np.asarray(y, dtype=np.float64),
                           train_frac=0.70, val_frac=0.10, test_frac=0.20,
                           seed=seed)
    tr = np.where(mask == 0)[0].astype(np.int64)
    va = np.where(mask == 1)[0].astype(np.int64)
    te = np.where(mask == 2)[0].astype(np.int64)
    max_sim = leakage_audit(keys, mask, audit_sample=2000, seed=seed)
    audit = {
        'method': ('rows collapsed by canonical 3-mer-multiset signature '
                   '(jaccard-1.0 anagrams merged to one family), then greedy '
                   'single-linkage 3-mer Jaccard clustering (threshold), '
                   'family-level split (AMPBench-MT-style homology control, '
                   'arXiv:2607.25518)'),
        'threshold': threshold,
        'n_unique_signatures': int(len(uniq_sig)),
        'n_families': int(len(np.unique(fam_uniq))),
        'max_train_test_kmer_jaccard_audited': round(float(max_sim), 4),
        'exact_multiset_leakage': 'guaranteed 0 (signature collapse)',
        'counts': {'train': int(len(tr)), 'val': int(len(va)),
                   'test': int(len(te))},
    }
    return tr, va, te, audit


def split_molecular(n, keys, seed):
    """Random 70/10/20 split on UNIQUE SMILES (duplicates grouped).

    The molecular endpoints have no sequence, so no homology control is
    possible; this is the strongest leakage control available on SMILES
    alone (exact duplicates cannot cross the boundary).
    """
    uniq, inv = np.unique(np.asarray(keys, dtype=object), return_inverse=True)
    # shuffle unique molecules, then assign by order-preserving fracs
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
    audit = {
        'method': 'random 70/10/20 on unique SMILES (no sequence available '
                  '-> no homology control possible; exact-duplicate SMILES '
                  'grouped into one split)',
        'n_unique_smiles': int(len(uniq)),
        'n_rows': int(n),
        'counts': {'train': int(len(tr)), 'val': int(len(va)),
                   'test': int(len(te))},
        'leakage_caveat': 'near-isomeric structures (different SMILES, same '
                          'chemistry) CAN cross the boundary — a real-data '
                          'limitation of SMILES-only data, weaker than the '
                          '3-mer homology control used for sequence endpoints.',
    }
    return tr, va, te, audit


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_endpoint_model(name, X, y, tr, va, lr=1e-3, epochs=80, patience=10,
                         batch_size=128, seed=42, hidden=(256, 128),
                         regression_loss='mse'):
    """Train one single-head MixedADMETMLP on one endpoint.

    ``hidden`` sets the trunk widths (persisted in the checkpoint so the
    predictor rebuilds the same architecture).  ``regression_loss`` is 'mse'
    (default) or 'huber' — Huber is more robust to the heavy-tailed experimental
    noise in the log-space regression targets (v4.2).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ep = ENDPOINT_BY_NAME[name]
    model = MixedADMETMLP(input_dim=X.shape[1], endpoints=[name],
                          hidden=hidden).to(device)
    n_params = int(sum(p.numel() for p in model.parameters()))
    print(f'  [{name}] params={n_params:,} device={device} '
          f'input_dim={X.shape[1]} hidden={tuple(hidden)}')

    Xt = torch.from_numpy(X[tr]).to(device)
    yt = torch.from_numpy(y[tr]).to(device)
    Xv = torch.from_numpy(X[va]).to(device)
    yv = torch.from_numpy(y[va]).to(device)

    if ep.kind == KIND_BINARY:
        pos = float(np.clip(y[tr].mean(), 1e-4, 1 - 1e-4))
        pos_weight = torch.tensor((1 - pos) / pos, dtype=torch.float32).clamp(max=20.0).to(device)
        def loss_fn(out, y_true):
            return F.binary_cross_entropy_with_logits(out[name].squeeze(1), y_true, pos_weight=pos_weight)
    else:
        pos_weight = None
        huber_delta = 1.0
        def loss_fn(out, y_true):
            if regression_loss == 'huber':
                return F.huber_loss(out[name].squeeze(1), y_true, delta=huber_delta)
            return F.mse_loss(out[name].squeeze(1), y_true)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min',
                                                       factor=0.5, patience=4)

    best_val, best_state, bad = float('inf'), None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xt))
        tot = 0.0
        for i in range(0, len(Xt), batch_size):
            b = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xt[b]), yt[b])
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            vloss = loss_fn(model(Xv), yv).item()
        sched.step(vloss)
        if vloss < best_val - 1e-5:
            best_val, bad = vloss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print(f'  [{name}] early stop @ epoch {epoch} (val {best_val:.4f})')
                break
        if epoch % 10 == 0 or epoch == 1:
            print(f'  [{name}] epoch {epoch:3d}  train {tot/len(Xt):.4f}  val {vloss:.4f}')

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, device, pos_weight


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def endpoint_metrics(name, model, Xs, y, device):
    """Measured metrics on a feature block Xs (already scaled) vs y."""
    ep = ENDPOINT_BY_NAME[name]
    pred = predict_mixed(model, Xs)[name]
    rec = {'kind': ep.kind, 'modality': ep.modality, 'n': int(len(y))}
    if ep.kind == KIND_BINARY:
        yint = y.astype(int)
        yp = (pred >= 0.5).astype(int)
        try:
            rec['auc'] = round(float(roc_auc_score(yint, pred)), 4)
        except ValueError:
            rec['auc'] = None
        rec['mcc'] = round(float(matthews_corrcoef(yint, yp)), 4)
        rec['accuracy'] = round(float(accuracy_score(yint, yp)), 4)
        rec['pos_rate'] = round(float(yint.mean()), 4)
    else:
        rec['r2'] = round(float(r2_score(y, pred)), 4)
        rec['rmse'] = round(float(np.sqrt(mean_squared_error(y, pred))), 4)
        rec['mae'] = round(float(np.mean(np.abs(y - pred))), 4)
        rec['target_units'] = ('log10(seconds)' if ep.target_transform == 'log10'
                               else ep.raw_units)
        rec['y_range'] = [round(float(y.min()), 4), round(float(y.max()), 4)]
        rec['pred_range'] = [round(float(pred.min()), 4), round(float(pred.max()), 4)]
    return rec


# --------------------------------------------------------------------------- #
# Per-endpoint driver
# --------------------------------------------------------------------------- #
def _dedup_by_key(X, y, keys, key_label):
    """Collapse exact-duplicate keys to one row, averaging the (transformed)
    target.  Returns (X, y, keys, dedup_info).

    For Half-life (v4.2 "new weapon"): 1763 prepared rows contain only 768
    unique sequences — 205 sequences are measured 2..82 times and the repeated
    measurements differ (the experimental noise floor we measured: median
    within-sequence std ~5.5e3 s).  Averaging each sequence's measurements in
    model space (log10 s) removes that irreducible per-measurement noise
    before training/evaluation, so the reported R2 is measured against the
    *sequence-level* truth instead of individual repeat measurements.  This is
    a standard, honest target-aggregation step (it changes what is being
    predicted: the expected half-life of the sequence, not a single
    measurement), and it is reported in the metrics file.
    """
    order, inv = np.unique(np.asarray(keys, dtype=object),
                           return_inverse=True)
    n_keys = len(order)
    if n_keys == len(keys):
        return X, y, keys, None
    y64 = np.asarray(y, dtype=np.float64)
    # per-unique-key mean target (float64) -> back to float32
    sums = np.zeros(n_keys); counts = np.zeros(n_keys)
    np.add.at(sums, inv, y64)
    np.add.at(counts, inv, 1.0)
    y_mean = (sums / counts).astype(np.float32)
    # one representative feature row per unique key (first occurrence)
    first_idx = np.argsort(inv, kind='stable')[:n_keys]
    first_idx = np.sort(first_idx)  # ascending = first occurrence per key
    X_new = X[first_idx].astype(np.float32)
    info = {
        'method': f'exact-duplicate {key_label}s collapsed to one row; target '
                  f'= mean of the {key_label}s repeat measurements in model '
                  f'space (log10 s for Half_life)',
        'n_rows_before': int(len(keys)),
        'n_rows_after': int(n_keys),
        'n_duplicate_keys': int(len(keys) - n_keys),
        'max_repeats': int(counts.max()),
    }
    return X_new, y_mean, [str(k) for k in order], info


def run_endpoint(name, epochs, seed, model_root, cfg=None):
    cfg = cfg or {}
    hidden = cfg.get('hidden', (256, 128))
    regression_loss = cfg.get('regression_loss', 'mse')
    ep = ENDPOINT_BY_NAME[name]
    slug = SLUGS[name]
    out_dir = Path(model_root) / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    X, y, keys, key_label, n = load_endpoint(name, PREPARED_CSV[name])

    # v4.2 (Half-life): collapse exact-duplicate sequences to one row with the
    # averaged (model-space) target, before any split, so repeats never leak
    # across boundaries and the metric is measured at sequence level.
    dedup_info = None
    if cfg.get('dedup'):
        X, y, keys, dedup_info = _dedup_by_key(X, y, keys, key_label)
        n = len(keys)
        print(f'  [{name}] DEDUP {dedup_info}')

    # v4.1: append the frozen ESMC-600M embedding to the sequence features.
    # The cache is verified against `keys` (the prepared-CSV sequences) inside
    # load_esmc_embeddings, so the concat is row-aligned by construction.
    if getattr(ep, 'esmc', False):
        emb = load_esmc_embeddings(name, keys, n)
        X = np.hstack([X, emb]).astype(np.float32)
        print(f'  [{name}] +ESMC {ESMC_MODEL}: input {X.shape[1] - ESMC_DIM} -> {X.shape[1]}')

    # v4.2: append the frozen MoLFormer-XL CLS embedding to the molecular
    # features.  The cache is verified against `keys` (the prepared-CSV SMILES)
    # inside load_molformer_embeddings, so the concat is row-aligned by
    # construction.
    if getattr(ep, 'molformer', False):
        emb = load_molformer_embeddings(name, keys, n)
        X = np.hstack([X, emb]).astype(np.float32)
        print(f'  [{name}] +MoLFormer {MOLEFORMER_MODEL}: input {X.shape[1] - MOLEFORMER_DIM} -> {X.shape[1]}')

    if ep.modality == 'sequence':
        tr, va, te, audit = split_sequence(n, keys, y, seed)
        # comparison: random split (inflated, same seed)
        r_tr, r_va = train_test_split(np.arange(n), test_size=0.3,
                                      random_state=seed)
        r_va, r_te = train_test_split(r_va, test_size=0.2 / 0.3, random_state=seed)
        r_tr, r_va, r_te = (a.astype(np.int64) for a in (r_tr, r_va, r_te))
        r_audit = {'method': 'random 70/10/20 (comparison only — inflated by '
                             'near-duplicate leakage)',
                   'counts': {'train': int(len(r_tr)), 'val': int(len(r_va)),
                              'test': int(len(r_te))}}
    else:
        tr, va, te, audit = split_molecular(n, keys, seed)
        # molecular: no homology split possible -> no separate comparison
        r_tr = r_va = r_te = None
        r_audit = None

    scaler = StandardScaler().fit(X[tr])
    Xs = scaler.transform(X).astype(np.float32)

    print(f'  [{name}] primary split: {audit["counts"]}')
    model, device, pos_weight = train_endpoint_model(
        name, Xs, y, tr, va, epochs=epochs, seed=seed, hidden=hidden,
        regression_loss=regression_loss)

    te_rec = endpoint_metrics(name, model, Xs[te], y[te], device)
    va_rec = endpoint_metrics(name, model, Xs[va], y[va], device)
    print(f'  [{name}] TEST  ' + '  '.join(f'{k}={v}' for k, v in te_rec.items()
                                          if k not in ('kind', 'modality', 'n', 'y_range', 'pred_range')))

    # comparison (sequence endpoints only)
    cmp_rec = None
    if r_tr is not None:
        r_scaler = StandardScaler().fit(X[r_tr])
        Xr = r_scaler.transform(X).astype(np.float32)
        r_model, _, _ = train_endpoint_model(name, Xr, y, r_tr, r_va,
                                             epochs=epochs, seed=seed,
                                             hidden=hidden,
                                             regression_loss=regression_loss)
        cmp_rec = endpoint_metrics(name, r_model, Xr[r_te], y[r_te], device)
        print(f'  [{name}] RANDOM-COMPARE TEST  ' +
              '  '.join(f'{k}={v}' for k, v in cmp_rec.items()
                        if k not in ('kind', 'modality', 'n', 'y_range', 'pred_range')))

    pw = float(pos_weight) if pos_weight is not None else None
    save_mixed_model(model, out_dir / 'admet_mlp.pt',
                     pos_weights={name: pw} if pw is not None else {})
    torch.save(scaler, out_dir / 'scaler.pt')

    metrics = {
        'endpoint': name,
        'kind': ep.kind,
        'modality': ep.modality,
        'model': 'MixedADMETMLP (single head, per-endpoint)',
        'n_params': int(sum(p.numel() for p in model.parameters())),
        'input_dim': int(X.shape[1]),
        'trunk_hidden': list(hidden),
        'regression_loss': (regression_loss if ep.kind != KIND_BINARY else None),
        'feature_layout': (('428-dim classical (20 AAC + 400 DPC + 8 physchem) '
                            '+ 1152-dim frozen ESMC-600M embedding = 1580'
                            if getattr(ep, 'esmc', False) else
                            '428-dim: 20 AAC + 400 DPC + 8 physchem')
                           if ep.modality == 'sequence' else
                           ('2265-dim classical (217 RDKit 2D descriptors + '
                            '2048-bit Morgan radius 2) + 768-dim frozen '
                            'MoLFormer-XL CLS embedding = 3033'
                            if getattr(ep, 'molformer', False) else
                            '2265-dim: 217 RDKit 2D descriptors + 2048-bit Morgan (radius 2)')),
        'esmc': (None if not getattr(ep, 'esmc', False) else
                 {'model': ESMC_MODEL, 'embedding_dim': ESMC_DIM,
                  'aggregation': 'mean-pooled token states',
                  'frozen': True,
                  'cache': esmc_cache_path(name)}),
        'molformer': (None if not getattr(ep, 'molformer', False) else
                      {'model': MOLEFORMER_MODEL, 'embedding_dim': MOLEFORMER_DIM,
                       'aggregation': 'CLS token',
                       'frozen': True,
                       'cache': molformer_cache_path(name)}),
        'target_aggregation': dedup_info,
        'target_transform': ep.target_transform,
        'raw_units': ep.raw_units,
        'data_source': 'Chemit797/PepADMET-Dataset (real experimental data)',
        'n_samples': int(n),
        'binary_pos_weight': pw,
        'training': {'epochs_max': epochs, 'early_stopping': 'val loss, patience 10',
                     'seed': seed, 'optimizer': 'Adam lr=1e-3 wd=1e-5',
                     'scheduler': 'ReduceLROnPlateau factor 0.5 patience 4'},
        'splits': {'primary': {'audit': audit, 'test': te_rec, 'val': va_rec},
                   'comparison': ({'audit': r_audit, 'test': cmp_rec} if r_audit else None)},
        'elapsed_s': round(time.time() - t0, 1),
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(out_dir / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f'  [{name}] saved {out_dir}/ ({metrics["elapsed_s"]}s)')
    return metrics


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description='Train v4.1 4-endpoint real-data + ESMC models')
    ap.add_argument('--endpoints', nargs='+',
                    default=[e.name for e in ENDPOINTS],
                    help='subset of: Hemolysis Half_life Caco2 PAMPA_MDCK')
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--model-root', type=str, default='models_v4')
    args = ap.parse_args()

    for e in args.endpoints:
        if e not in ENDPOINT_BY_NAME:
            raise SystemExit(f'unknown endpoint {e}; valid: {list(ENDPOINT_BY_NAME)}')

    # v4.2 per-endpoint "new weapon" config.
    #   Half_life : collapse exact-duplicate sequences to one row (1763 -> 768)
    #               with the averaged model-space target, + Huber loss (robust
    #               to the heavy-tailed log-space measurement noise).
    #   Caco2 / PAMPA_MDCK : MoLFormer-XL concat (done via the endpoint flag)
    #               + Huber loss (robust to 1-2 log-unit permeability noise).
    #   Hemolysis : unchanged (binary; no regression loss / dedup applies).
    cfg_by_name = {
        'Half_life':    {'dedup': True,  'regression_loss': 'huber'},
        'Caco2':        {'regression_loss': 'huber'},
        'PAMPA_MDCK':   {'regression_loss': 'huber'},
        'Hemolysis':    {},
    }

    t0 = time.time()
    all_metrics = {}
    for name in args.endpoints:
        print(f'\n=== {name} ===')
        all_metrics[name] = run_endpoint(name, args.epochs, args.seed,
                                         args.model_root, cfg=cfg_by_name.get(name, {}))

    # cross-endpoint summary
    root = Path(args.model_root)
    root.mkdir(parents=True, exist_ok=True)
    summary = {
        'version': ('v4.2 (real-data + ESMC-600M sequence embeddings + '
                    'MoLFormer-XL molecular embeddings, Chemit797/PepADMET-Dataset)'),
        'endpoints': {},
        'note': ('Each endpoint is an independent single-task model on its own '
                 'modality-appropriate features; there is no composite score '
                 'because the four datasets are disjoint molecules in two '
                 'different feature spaces. Primary split = homology-controlled '
                 '(sequence) or unique-SMILES random (molecular, no sequence '
                 'available). All metrics measured on holdout. v4.2 adds frozen '
                 'MoLFormer-XL CLS embeddings to the molecular endpoints and, '
                 'for Half-life, aggregates exact-duplicate-sequence repeats to '
                 'their mean (sequence-level target) + Huber loss.'),
    }
    for name, m in all_metrics.items():
        te = m['splits']['primary']['test']
        primary_metric = te.get('auc', te.get('r2'))
        summary['endpoints'][name] = {
            'kind': m['kind'], 'modality': m['modality'],
            'n_samples': m['n_samples'], 'primary_metric': primary_metric,
            'primary_metric_name': 'AUC' if m['kind'] == 'binary' else 'R2',
            'n_params': m['n_params'], 'input_dim': m['input_dim'],
            'elapsed_s': m['elapsed_s'],
        }
    with open(root / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print('\n' + '=' * 70)
    print('SUMMARY (primary split, measured on holdout)')
    print('=' * 70)
    for name, s in summary['endpoints'].items():
        print(f'  {name:12s} {s["kind"]:10s} {s["modality"]:9s} '
              f'{s["primary_metric_name"]}={s["primary_metric"]:.4f}  '
              f'n={s["n_samples"]}')
    print(f'\nTotal wall time: {time.time() - t0:.1f}s')
    print(f'Wrote {root / "summary.json"}')


if __name__ == '__main__':
    main()
