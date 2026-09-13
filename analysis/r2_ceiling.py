#!/usr/bin/env python3
"""
R2 ceiling decomposition for the PAMPA endpoint (the '0.54' number).

Recomputes, reproducibly:
  1) variance share of the three target regions (floor / bulk / tail)
  2) the 'uncensored perfect, censored -> mean' R2 ceiling
  3) the oracle curve (flag k of the true test-floor rows -> R2) using
     the committed v4.2 MLP predictions on the non-floor rows

All on the verbatim pipeline split (seed 42, unique-SMILES 70/10/20).
No training involved: loads models_v4/pampa_mdck directly, so it is
fast (<2 min) and needs only the committed artifacts.
"""
import common  # re-roots CWD to repo root

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
from sklearn.metrics import r2_score
from admet_model import MixedADMETMLP

smiles_list, y = common.load_data()
tr, va, te = common.split_smiles(smiles_list)
y_te = y[te]
floor_te = y_te <= common.FLOOR + 1e-6

# --- committed v4.2 MLP predictions on the test set ---
# v4.2 input = molecule_features (2265) + frozen MoLFormer CLS (768) = 3033,
# built exactly as train_pepadmet_model.py does (row-aligned via the cache).
import feature_extractor  # noqa: E402
import endpoint_config  # noqa: E402
from pathlib import Path  # noqa: E402
X_mol = feature_extractor.molecule_features(smiles_list)
z = np.load(endpoint_config.molformer_cache_path('PAMPA_MDCK'))
emb = np.asarray(z['emb'], dtype=np.float32)
assert emb.shape == (len(smiles_list), 768), emb.shape
X3033 = np.hstack([X_mol, emb]).astype(np.float32)
blob = torch.load('models_v4/pampa_mdck/admet_mlp.pt', map_location='cpu', weights_only=False)
mlp = MixedADMETMLP(input_dim=blob['input_dim'], endpoints=blob['endpoints'],
                    hidden=blob.get('hidden', (256, 128)), dropout=blob.get('dropout', 0.25))
mlp.load_state_dict(blob['state_dict']); mlp.eval()
sc = torch.load('models_v4/pampa_mdck/scaler.pt', map_location='cpu', weights_only=False)
with torch.no_grad():
    p_te = mlp(torch.from_numpy(sc.transform(X3033[te])))['PAMPA_MDCK'].squeeze(1).numpy()

print(f'N={len(y)}  test={len(te)}  test floor rows={floor_te.sum()}')
print(f'overall test R2 (committed model) = {r2_score(y_te, p_te):.4f}')

# --- 1) variance shares (full dataset): contribution to the GLOBAL SS ---
# (NOT within-region variance — the floor rows are near-constant at -10, so
#  their within-region variance is ~0; what matters for R2 is how far the
#  region sits from the global mean, i.e. sum of (y_i - y_mean)^2.)
F = common.FLOOR
var_tot = ((y - y.mean()) ** 2).sum()
for name, m in (('floor y==-10.0', y <= F + 1e-6),
                ('bulk -10<y<=-6', (y > F + 1e-6) & (y <= -6)),
                ('tail y>-6', y > -6)):
    share = ((y[m] - y.mean()) ** 2).sum() / var_tot * 100
    print(f'  {name:<16} n={m.sum():>5}  share of total SS = {share:5.1f}%')

# --- 2) ceiling: perfect on non-floor, floor -> its test mean ---
p_ceil = p_te.copy()
p_ceil[~floor_te] = y_te[~floor_te]          # oracle on the uncensored rows
p_ceil[floor_te] = y_te[floor_te].mean()     # floor -> its mean (uninformative)
r2_ceil = r2_score(y_te, p_ceil)
# note: with floor->mean the floor contribution to ss_res is 0 only if the
# floor test mean equals the global test mean; report both conventions
p_ceil2 = p_te.copy(); p_ceil2[~floor_te] = y_te[~floor_te]
p_ceil2[floor_te] = y_te.mean()
print(f'\nceiling (uncensored perfect, floor->test-mean)      R2 = {r2_ceil:.4f}')
print(f'ceiling (uncensored perfect, floor->global-mean)     R2 = {r2_score(y_te, p_ceil2):.4f}')

# --- 3) oracle curve on the CURRENT model (k of 47 true floor rows set to -10) ---
print('\noracle sweep (current regressor, k true floor rows set to -10):')
floor_pos = np.where(floor_te)[0]
for k in (0, 10, 20, 30, 40, 47):
    p_or = p_te.copy(); p_or[floor_pos[:k]] = F
    print(f'  k={k:>2}: R2={r2_score(y_te, p_or):.4f}')

# --- 4) the model's own floor-rank signal ---
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(floor_te.astype(int), -p_te)
print(f'\nAUC of (MLP pred -> floor) on test = {auc:.4f}  (0.5 = no signal)')
print('Done.')
