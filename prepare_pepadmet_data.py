#!/usr/bin/env python3
"""
prepare_pepadmet_data.py
========================

Load the four cleaned endpoint tables from the **Chemit797/PepADMET-Dataset**
release (``整理/`` sub-directory), validate + clean each endpoint's rows
according to its feature modality, and write one prepared per-endpoint CSV
plus a data provenance JSON.

Per-endpoint cleaning rules (see endpoint_config.py for the mapping):

  * sequence modality (Hemolysis, Half-life)
      - the raw sequence column must be a one-letter string over the 20
        standard amino acids (upper-cased, whitespace stripped);
      - length within [SEQUENCE_MIN_LEN, SEQUENCE_MAX_LEN];
      - label must be present (Hemolysis 0/1; Half-life finite seconds).
  * molecular modality (Caco2, PAMPA_MDCK)
      - the raw SMILES must parse to a non-null RDKit molecule;
      - label must be present (finite logPapp value).

Rows failing any rule are DROPPED and counted in the meta JSON — never
zero-filled, never imputed.  The prepared CSVs are the sole training input
(the synthetic generator path is gone in v4.0).

Usage:
    python prepare_pepadmet_data.py [--dataset-root DIR] [--out-dir data]
                                    [--smoke N]

    --dataset-root  path to a local clone of Chemit797/PepADMET-Dataset
                    (default: ../PepADMET-Dataset)
    --out-dir       where to write prepared CSVs + pepadmet_data.meta.json
                    (default: data)
    --smoke N       process only the first N rows of each table (dev only)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from endpoint_config import (
    ENDPOINTS, ENDPOINT_NAMES,
    MODALITY_SEQUENCE, MODALITY_MOLECULAR,
    SEQUENCE_MIN_LEN, SEQUENCE_MAX_LEN, VALID_AA,
)
from feature_extractor import parseable_smiles_mask

DEFAULT_DATASET_ROOT = Path('PepADMET-Dataset')  # resolved relative to repo


def clean_sequence(s) -> Optional[str]:
    """Return a canonical 20-AA sequence string, or None if invalid."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    t = str(s).strip().upper().replace(' ', '')
    if not t:
        return None
    if any(ch not in VALID_AA for ch in t):
        return None
    if not (SEQUENCE_MIN_LEN <= len(t) <= SEQUENCE_MAX_LEN):
        return None
    return t


def _label_stats(label: pd.Series) -> Dict:
    """Compact numeric label stats for the meta JSON."""
    v = label.astype(float)
    out = {'n': int(len(v)), 'n_null': int(v.isna().sum()),
           'min': float(np.nanmin(v)), 'median': float(np.nanmedian(v)),
           'max': float(np.nanmax(v))}
    return out


def prepare_endpoint(ep, df: pd.DataFrame, dataset_root: Path) -> Dict:
    """Clean one endpoint's table; return (prepared_df, stats_dict)."""
    src = dataset_root / ep.source_file
    if not src.exists():
        raise FileNotFoundError(f'endpoint {ep.name}: source not found: {src}')
    raw_n = len(df)
    stats = {'endpoint': ep.name, 'kind': ep.kind, 'modality': ep.modality,
             'source_file': str(ep.source_file), 'n_raw': raw_n}

    if ep.modality == MODALITY_SEQUENCE:
        seqs = [clean_sequence(x) for x in df[ep.seq_column].tolist()]
        seq_series = pd.Series(seqs)
        seq_ok = seq_series.notna().to_numpy()
        label = df[ep.source_column].to_numpy(dtype='float64')
        lab_ok = ~np.isnan(label)
        if ep.kind == 'binary':
            lab_ok &= np.isin(label, [0.0, 1.0])
        keep = seq_ok & lab_ok
        seq_out = seq_series.to_numpy()[keep]
        lab_out = label[keep]
        stats['n_valid_sequence'] = int(seq_ok.sum())
        stats['n_with_label'] = int(lab_ok.sum())
        lab_series = pd.Series(lab_out)
        if ep.kind == 'binary':
            stats['label_distribution'] = {int(k): int(v)
                                           for k, v in lab_series.value_counts().sort_index().items()}
        else:
            stats['label_stats_raw_units'] = _label_stats(lab_series)
            stats['target_transform'] = ep.target_transform
        prepared = pd.DataFrame({'endpoint': ep.name, 'kind': ep.kind,
                                 'modality': ep.modality, 'sequence': seq_out,
                                 ep.name: lab_out})
    else:  # molecular
        smiles_raw = df[ep.smiles_column].tolist()
        ok = parseable_smiles_mask(smiles_raw)
        kept_idx = [i for i, k in enumerate(ok) if k]
        smiles_out = pd.Series(smiles_raw)[ok].astype(str).reset_index(drop=True)
        label = df[ep.source_column].iloc[kept_idx].reset_index(drop=True).astype(float)
        lab_ok = label.notna().to_numpy()
        kept2 = [i for i, k in enumerate(lab_ok) if k]
        smiles_out = smiles_out.iloc[kept2].reset_index(drop=True)
        label = label[lab_ok].reset_index(drop=True)
        stats['n_parseable_smiles'] = int(ok.sum())
        stats['n_with_label'] = int(lab_ok.sum())
        stats['label_stats_raw_units'] = _label_stats(label)
        stats['target_transform'] = ep.target_transform
        prepared = pd.DataFrame({'endpoint': ep.name, 'kind': ep.kind,
                                 'modality': ep.modality, 'smiles': smiles_out,
                                 ep.name: label})

    stats['n_final'] = int(len(prepared))
    stats['n_dropped'] = raw_n - len(prepared)
    return prepared, stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset-root', type=Path, default=None,
                    help='local clone of Chemit797/PepADMET-Dataset '
                         '(default: repo_root/../PepADMET-Dataset)')
    ap.add_argument('--out-dir', type=Path, default=Path('data'))
    ap.add_argument('--smoke', type=int, default=None,
                    help='only first N rows per table (dev only)')
    args = ap.parse_args(argv)

    repo_root = Path(__file__).resolve().parent
    dataset_root = (repo_root / args.dataset_root) if args.dataset_root \
        else repo_root.parent / DEFAULT_DATASET_ROOT
    if not dataset_root.exists():
        print(f'ERROR: dataset root not found: {dataset_root}', file=sys.stderr)
        return 2
    args_out = args.out_dir
    if not args_out.is_absolute():
        args_out = repo_root / args_out
    args_out.mkdir(parents=True, exist_ok=True)

    meta = {
        'pipeline_version': 'v4.0-real-data',
        'dataset': 'Chemit797/PepADMET-Dataset (整理/ cleaned tables)',
        'dataset_root': str(dataset_root),
        'smoke': args.smoke,
        'endpoints': {},
        'files': {},
    }

    for ep in ENDPOINTS:
        df = pd.read_csv(dataset_root / ep.source_file)
        if args.smoke is not None:
            df = df.head(args.smoke)
        prepared, stats = prepare_endpoint(ep, df, dataset_root)
        slug = ep.name.lower().replace(' ', '_')
        out_name = f'pepadmet_{slug}.csv'
        out_path = args_out / out_name
        prepared.to_csv(out_path, index=False)
        meta['endpoints'][ep.name] = stats
        meta['files'][ep.name] = str(out_path)
        print(f'  {ep.name:12s} {ep.modality:10s} {ep.kind:11s} '
              f'raw={stats["n_raw"]:6d} final={stats["n_final"]:6d} '
              f'dropped={stats["n_dropped"]:5d}')

    meta_path = args_out / 'pepadmet_data.meta.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f'\nwrote {len(ENDPOINTS)} prepared CSVs + {meta_path.name}')
    print('total rows:', sum(s['n_final'] for s in meta['endpoints'].values()))
    return 0


if __name__ == '__main__':
    sys.exit(main())
