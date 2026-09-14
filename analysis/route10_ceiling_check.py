import numpy as np, pandas as pd, json
from sklearn.metrics import r2_score

# same canonical split as route10
df = pd.read_csv('data/pepadmet_caco2.csv')
smiles = df['smiles'].astype(str).tolist()
y = df['Caco2'].to_numpy(dtype=np.float64)
N = len(smiles)
uniq, inv = np.unique(np.asarray(smiles, dtype=object), return_inverse=True)
rng = np.random.default_rng(42)
perm = rng.permutation(len(uniq))
n_tr = int(round(len(uniq) * 0.70)); n_va = int(round(len(uniq) * 0.10))
tr_ids = set(perm[:n_tr].tolist()); va_ids = set(perm[n_tr:n_tr+n_va].tolist())
te = np.array([i for i in range(N) if inv[i] not in tr_ids and inv[i] not in va_ids], dtype=np.int64)
y_te = y[te]
floor_te = y_te <= -10.0 + 1e-6

# Convention identical to PAMPA r2_ceiling.py (the 0.5387 derivation):
#   non-floor rows -> PERFECT (y itself); floor rows -> GLOBAL test mean
p_ceil = y_te.copy()
p_ceil[floor_te] = y_te.mean()          # floor -> global mean (uninformative)
ceil_global = r2_score(y_te, p_ceil)

# also the test-set-only variant: floor -> test-set floor mean is trivial (floor
# rows are exactly -10.0 -> perfect), so report the informative convention only.
floor_ss_share = ((y_te[floor_te] - y_te.mean())**2).sum() / ((y_te - y_te.mean())**2).sum()
print(f'test n={len(te)} floor rows={int(floor_te.sum())} ({floor_te.mean()*100:.1f}% of test)')
print(f'floor SS share of test SS (about global mean): {floor_ss_share:.4f}')
print(f'oracle ceiling Caco-2 (perfect non-floor + floor->global mean): {ceil_global:.4f}')
print(f'check: 1 - floor_ss_share = {1-floor_ss_share:.4f} (must match)')
