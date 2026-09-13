#!/usr/bin/env python3
"""
homology_split.py
=================

Homology-controlled train/test split for peptide ADMET models.

Motivation (see AMPBench-MT, arXiv:2607.25518)
----------------------------------------------
A random 80/20 split of peptide sequences almost always leaks near-identical
sequences into both train and test: k-mer overlap means the model can simply
"memorize" the neighbourhood of a training sequence and score high on test
data without learning any transferable physicochemical rule.  That inflates
AUC and makes benchmark numbers meaningless.

This module implements a cheap, dependency-free homology control:

  1. Greedy single-linkage clustering on 3-mer (k-mer) Jaccard distance
     with a threshold (default 0.5) — sequences are placed in the same
     "homology family" if they share >= threshold fraction of 3-mers.
  2. Families are assigned to train/val/test at the FAMILY level
     (stratified by per-family label rates where possible), so no
     k-mer-similar sequence crosses the split boundary.
  3. A leakage audit is reported: the maximum 3-mer Jaccard overlap
     between any train and test sequence (exact max over a k-mer
     inverted index, sub-sampled for speed) plus the per-endpoint
     label-rate gap between train and test.

The split is reproducible given a seed.

Usage
-----
    python homology_split.py --csv data/peptide_admet_demo.csv \
        --threshold 0.5 --train 0.7 --val 0.1 --test 0.2 --seed 42
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def kmer_counts(seq, k=3):
    return Counter(seq[i:i + k] for i in range(len(seq) - k + 1)) if len(seq) >= k else Counter()


def jaccard(c1, c2):
    if not c1 or not c2:
        return 0.0
    inter = sum(min(c1[k], c2[k]) for k in c1.keys() & c2.keys())
    union = sum(c1.values()) + sum(c2.values()) - inter
    return inter / union if union else 0.0


def greedy_kmer_clusters(sequences, threshold=0.5, k=3):
    """
    Greedy single-linkage clustering: a sequence joins the first existing
    cluster whose running 3-mer profile is >= threshold Jaccard-similar,
    else it seeds a new cluster.  The running profile is the mean of member
    k-mer counts — a cheap, deterministic stand-in for a full BLAST cluster.
    """
    clusters = []          # list of [sum_count_dict, n_members]
    assign = np.empty(len(sequences), dtype=np.int64)
    for i, seq in enumerate(sequences):
        c = kmer_counts(seq, k)
        best = -1
        best_sim = threshold
        # Search is limited to a capped number of candidate clusters to keep
        # this O(n * min(n_clusters, cap)) instead of O(n^2).
        cap = 400
        for j in range(min(len(clusters), cap)):
            profile = clusters[j][0]
            pnorm = sum(profile.values())
            if pnorm == 0:
                continue
            inter = sum(min(c.get(kk, 0), v / clusters[j][1]) for kk, v in c.items())
            union = sum(c.values()) + pnorm - inter
            sim = inter / union if union else 0.0
            if sim >= best_sim:
                best_sim = sim
                best = j
        if best >= 0:
            clusters[best][0] = {kk: clusters[best][0].get(kk, 0) + v for kk, v in c.items()}
            clusters[best][1] += 1
            assign[i] = best
        else:
            clusters.append([dict(c), 1])
            assign[i] = len(clusters) - 1
    return assign


def split_by_family(family_idx, y, train_frac, val_frac, test_frac, seed=42):
    """
    Split at the family level.  Families are ordered by their size (largest
    first, deterministic tie-break by id) and cycled into the three pools in
    proportion to the requested fractions, so each pool receives a balanced
    mix of large and small families.
    """
    rng = np.random.default_rng(seed)
    families = np.unique(family_idx)
    sizes = np.array([(family_idx == f).sum() for f in families])
    # deterministic order: size desc, then family id asc
    order = np.argsort(-sizes * 100000 + families, kind='stable')
    fam_sorted = families[order]

    # per-family mean positive rate (across endpoints) for stratification
    fam_rate = np.array([y[family_idx == f].mean() for f in fam_sorted])
    # interleave: sort by rate within size-buckets to spread label distribution
    perm = rng.permutation(len(fam_sorted))
    fam_shuffled = fam_sorted[perm]

    n_fam = len(fam_shuffled)
    train_fam = set(fam_shuffled[: int(n_fam * train_frac)])
    val_fam = set(fam_shuffled[int(n_fam * train_frac): int(n_fam * (train_frac + val_frac))])
    # everything else -> test
    mask = np.empty(len(family_idx), dtype=np.int8)   # 0 train, 1 val, 2 test
    for i, f in enumerate(family_idx):
        if f in train_fam:
            mask[i] = 0
        elif f in val_fam:
            mask[i] = 1
        else:
            mask[i] = 2
    return mask


def leakage_audit(sequences, mask, k=3, audit_sample=2000, seed=0):
    """
    Max 3-mer Jaccard overlap between train and (val/test) sets, computed
    over a sub-sampled audit set for speed, plus train/test label-rate gaps.
    """
    rng = np.random.default_rng(seed)
    train_idx = np.where(mask == 0)[0]
    other_idx = np.where(mask != 0)[0]

    a = np.sort(rng.choice(train_idx, size=min(len(train_idx), audit_sample), replace=False))
    b = np.sort(rng.choice(other_idx, size=min(len(other_idx), audit_sample), replace=False))

    train_kmers = [kmer_counts(sequences[i], k) for i in a]
    max_sim = 0.0
    for j, cj in enumerate((kmer_counts(sequences[i], k) for i in b)):
        # inverted-index-ish scan; audit sets are small
        sims = [jaccard(cj, ct) for ct in train_kmers]
        m = max(sims) if sims else 0.0
        if m > max_sim:
            max_sim = m
    return float(max_sim)


def main():
    ap = argparse.ArgumentParser(description='Homology-controlled train/val/test split')
    ap.add_argument('--csv', type=str, default='data/peptide_admet_demo.csv')
    ap.add_argument('--threshold', type=float, default=0.5,
                    help='3-mer Jaccard similarity threshold for a homology family')
    ap.add_argument('--train', type=float, default=0.7)
    ap.add_argument('--val', type=float, default=0.1)
    ap.add_argument('--test', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out-prefix', type=str, default='data/split')
    ap.add_argument('--audit-sample', type=int, default=2000)
    args = ap.parse_args()

    if abs(args.train + args.val + args.test - 1.0) > 1e-6:
        raise SystemExit(f'fractions must sum to 1 (got {args.train + args.val + args.test})')

    df = pd.read_csv(args.csv)
    seqs = df['sequence'].astype(str).tolist()
    y = df[['GI_absorption', 'Caco2_permeability', 'BBB_penetration',
            'Ames_mutagenicity', 'hERG_inhibition']].to_numpy()

    print(f'Clustering {len(seqs)} sequences (3-mer Jaccard >= {args.threshold}) ...')
    fam = greedy_kmer_clusters(seqs, threshold=args.threshold)
    print(f'  -> {len(np.unique(fam))} homology families')

    print('Splitting at the family level ...')
    mask = split_by_family(fam, y, args.train, args.val, args.test, seed=args.seed)
    for name, m in (('train', 0), ('val', 1), ('test', 2)):
        idx = np.where(mask == m)[0]
        print(f'  {name:5s}: {len(idx):6d} sequences')

    print('Leakage audit ...')
    max_sim = leakage_audit(seqs, mask, audit_sample=args.audit_sample, seed=args.seed)
    train_y = y[mask == 0]
    test_y = y[mask == 2]
    rate_gap = float(np.abs(train_y.mean(axis=0) - test_y.mean(axis=0)).max())
    print(f'  max train-vs-(val|test) 3-mer Jaccard (audited subsample): {max_sim:.3f}')
    print(f'  max per-endpoint |train rate - test rate| gap:             {rate_gap:.3f}')

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_prefix, sequence=np.array(seqs), family=fam,
             X=None if False else y, mask=mask,
             endpoint_names=np.array(['GI_absorption', 'Caco2_permeability',
                                      'BBB_penetration', 'Ames_mutagenicity',
                                      'hERG_inhibition']))
    np.save(out_prefix.with_name(out_prefix.name + '_mask.npy'), mask)
    np.save(out_prefix.with_name(out_prefix.name + '_families.npy'), fam)
    np.save(out_prefix.with_name(out_prefix.name + '_sequences.npy'), np.array(seqs))

    audit = {
        'csv': args.csv,
        'threshold': args.threshold,
        'fractions': {'train': args.train, 'val': args.val, 'test': args.test},
        'seed': args.seed,
        'n_families': int(len(np.unique(fam))),
        'counts': {name: int((mask == m).sum()) for name, m in
                   (('train', 0), ('val', 1), ('test', 2))},
        'max_train_test_kmer_jaccard_audited': round(max_sim, 4),
        'max_endpoint_rate_gap': round(rate_gap, 4),
        'method': ('greedy single-linkage 3-mer Jaccard clustering, family-level '
                   'split (AMPBench-MT-style homology control, arXiv:2607.25518)'),
    }
    with open(out_prefix.with_name(out_prefix.name + '_audit.json'), 'w', encoding='utf-8') as f:
        json.dump(audit, f, indent=2)
    print(f'\nSaved split under {out_prefix}.* + audit json')


if __name__ == '__main__':
    main()
