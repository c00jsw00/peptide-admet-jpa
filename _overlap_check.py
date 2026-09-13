"""Quantify SMILES overlap between our PAMPA (pepADMET) and PeptiVerse PAMPA
raw data (HF ChatterjeeLab/PeptiVerse_data, CycPeptMPDB-sourced)."""
import pandas as pd

ours = pd.read_csv('data/pepadmet_pampa_mdck.csv')
sm_o = ours['smiles'].astype(str)
y_o = ours['PAMPA_MDCK']
dtr = pd.read_parquet('data/peptiverse/pampa_train.parquet')
dva = pd.read_parquet('data/peptiverse/pampa_val.parquet')
pv = pd.concat([dtr, dva], ignore_index=True)
sm_p = pv['sequence'].astype(str)
y_p = pv['label']

so, sp = set(sm_o), set(sm_p)
inter = so & sp
print(f'ours pepADMET: n={len(sm_o)} unique={len(so)}')
print(f'peptiverse   : n={len(sm_p)} unique={len(sp)}')
print(f'intersection (exact SMILES string): {len(inter)} '
      f'= {100*len(inter)/len(so):.1f}% of ours, {100*len(inter)/len(sp):.1f}% of PV')
print(f'PV rows NOT in ours: {len(sp)-len(inter)} unique SMILES')

# label agreement on the intersection (same molecule, same assay value?)
m_o = dict(zip(sm_o, y_o))
m_p = dict(zip(sm_p, y_p))
common = [s for s in inter]
diffs = [abs(m_o[s]-m_p[s]) for s in common]
import statistics
print(f'label diff on intersection: max={max(diffs):.4f} '
      f'median={statistics.median(diffs):.4f}')
print(f'  exact label matches: {sum(1 for d in diffs if d<1e-4)} / {len(diffs)}')

# floor comparison
FLOOR = -10.0
fo = (y_o <= FLOOR + 1e-6).sum(); fp = (y_p <= FLOOR + 1e-6).sum()
print(f'floor rows: ours {fo}/{len(y_o)} ({100*fo/len(y_o):.1f}%), '
      f'PV {fp}/{len(y_p)} ({100*fp/len(y_p):.1f}%)')
