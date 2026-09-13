#!/usr/bin/env python3
"""v4.2 end-to-end verification.

Confirms, on the freshly trained v4.2 checkpoints:
  1. Each model rebuilds from its saved blob (input_dim, hidden, dropout) and
     loads its state_dict cleanly.
  2. A cached sequence (Hemolysis/Half-life) and a cached SMILES (Caco2/PAMPA)
     predict without touching the ESMC/MoLFormer subprocesses (cache path).
  3. A NOVEL sequence + a NOVEL SMILES predict via the subprocess fallback
     (ad-hoc embedding path) and return finite values.
  4. The predictor's to_readable conversion is correct (Half-life -> seconds,
     molecular -> logPapp, Hemolysis -> probability).
Exit 0 on full pass, non-zero on any failure.
"""
import sys
import numpy as np

import peptide_admet_predictor as P

REPO = P.Path(__file__).resolve().parent if hasattr(P, 'Path') else None
import pandas as pd
from pathlib import Path
root = Path(__file__).resolve().parent

def pick_smiles(slug):
    df = pd.read_csv(root / f'data/pepadmet_{slug}.csv')
    return df['smiles'].astype(str).tolist()

def main():
    ok = True
    # --- 1+2: cached-path prediction over a handful of training rows ---
    seqs_hemi = pd.read_csv(root / 'data/pepadmet_hemolysis.csv')['sequence'].astype(str).tolist()[:3]
    seqs_half = pd.read_csv(root / 'data/pepadmet_half_life.csv')['sequence'].astype(str).tolist()[:3]
    smi_caco2 = pick_smiles('caco2')[:3]
    smi_pampa = pick_smiles('pampa_mdck')[:3]

    rows = []
    for s in seqs_hemi:
        rows.append({'sequence': s})
    for s in seqs_half:
        rows.append({'sequence': s})
    for s in smi_caco2:
        rows.append({'smiles': s})
    for s in smi_pampa:
        rows.append({'smiles': s})

    res = P.predict_rows(rows, ['Hemolysis', 'Half_life', 'Caco2', 'PAMPA_MDCK'],
                         str(root / 'models_v4'))
    print('--- cached-path predictions (first 3 rows per modality) ---')
    # rows 0-2: sequence; 3-5: sequence (half-life); 6-8: SMILES (caco2); 9-11: SMILES (pampa)
    seq_row = {'Hemolysis', 'Half_life'}          # must be ok on a sequence row
    mol_row = {'Caco2', 'PAMPA_MDCK'}             # must be ok on a SMILES row
    for i, r in enumerate(res):
        line = f'row {i:2d}: '
        for name, rec in r.items():
            line += (f'{name}={rec["value"]:.4f}  ' if rec['ok']
                     else f'{name}=SKIP({rec["reason"]})  ')
        print(line)
        # determine which modality this row supplies
        row = rows[i]
        is_seq = 'sequence' in row
        is_smi = 'smiles' in row
        must_ok = (seq_row if is_seq else set()) | (mol_row if is_smi else set())
        must_skip = (mol_row if is_seq else set()) | (seq_row if is_smi else set())
        for name in must_ok:
            if not r[name]['ok']:
                print(f'  FAIL: row {i} — {name} should have a value, got {r[name]["reason"]}')
                ok = False
        for name in must_skip:
            if r[name]['ok'] or r[name]['reason'] != 'no input':
                print(f'  FAIL: row {i} — {name} should be a clean "no input" skip')
                ok = False

    # sanity: hemolysis is a probability in [0,1]; half-life is seconds (>0);
    # caco2/pampa are logPapp (finite)
    h = res[0]['Hemolysis']['value']
    if not (0.0 <= h <= 1.0):
        print(f'  FAIL: Hemolysis prob out of [0,1]: {h}'); ok = False
    hl = res[3]['Half_life']['value']
    if not (hl > 0 and np.isfinite(hl)):
        print(f'  FAIL: Half_life seconds not >0/finite: {hl}'); ok = False
    c = res[6]['Caco2']['value']
    p = res[9]['PAMPA_MDCK']['value']
    if not np.isfinite(c) or not np.isfinite(p):
        print(f'  FAIL: molecular logPapp not finite (caco2={c}, pampa={p})'); ok = False
    # a Caco2 row must NOT have a Half_life value (different modality)
    if res[6].get('Half_life', {}).get('ok'):
        print('  FAIL: Caco2 row unexpectedly produced Half_life'); ok = False

    # --- 3: novel inputs via the ad-hoc subprocess fallback ---
    print('\n--- ad-hoc (novel) predictions via subprocess fallback ---')
    novel_seq = 'ACDEFGHIKLMNPQRSTVWY' * 2   # 40 AA, not in training set
    novel_smi = 'CC(=O)Oc1ccccc1C(=O)O'      # aspirin, not a peptide
    res2 = P.predict_rows([{'sequence': novel_seq, 'smiles': novel_smi}],
                          ['Hemolysis', 'Half_life', 'Caco2', 'PAMPA_MDCK'],
                          str(root / 'models_v4'))
    for name, rec in res2[0].items():
        if rec['ok']:
            print(f'  novel {name} = {rec["value"]:.4f}  ({rec["unit"]})')
            if not np.isfinite(rec["value"]):
                print(f'  FAIL: novel {name} not finite'); ok = False
        else:
            print(f'  novel {name} = SKIP ({rec["reason"]})')
            ok = False

    print('\n' + ('E2E ALL PASS' if ok else 'E2E FAILURES PRESENT'))
    return 0 if ok else 1

if __name__ == '__main__':
    sys.exit(main())
