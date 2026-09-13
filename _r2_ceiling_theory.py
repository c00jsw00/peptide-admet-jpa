import pandas as pd, numpy as np

# Theoretical ceiling: for a model whose features are fully determined by
# `key` (sequence / SMILES), the best predictor is E[y|key].
#   R2_ceil = 1 - E[Var(y | key)] / Var(y)
# Anything above R2_ceil is mathematically impossible for ANY model on these
# features, because it would require predicting the irreducible label noise.
base = "data"
jobs = [
    ("Half_life",  "pepadmet_half_life.csv", "sequence", "Half_life", "log10"),
    ("Caco2",      "pepadmet_caco2.csv",     "smiles",   "Caco2",     "identity"),
    ("PAMPA_MDCK", "pepadmet_pampa_mdck.csv","smiles",   "PAMPA_MDCK","identity"),
]
print(f"{'endpoint':12s} {'n':>6s} {'Var(y)':>8s} {'E[Var(y|key)]':>14s} "
      f"{'R2_ceil':>8s}  note")
for name, file, key, col, tf in jobs:
    df = pd.read_csv(f"{base}/{file}")
    y = df[col].to_numpy(dtype=float)
    if tf == "log10":
        y = np.log10(y)
    keys = df[key].to_numpy()
    s = pd.Series(y, index=pd.Index(keys))
    cond_mean = s.groupby(level=0).transform("mean")
    cond_var  = s.groupby(level=0).var(ddof=1)
    cond_var = cond_var.fillna(0.0)          # single-observation groups -> 0
    # weight each group's variance by its group size (proper E[Var(y|X)])
    group_sizes = s.groupby(level=0).transform("count")
    e_var_given_x = float((cond_var * group_sizes).sum() / len(y))
    var_y = float(np.var(y, ddof=1))
    r2_ceil = 1.0 - e_var_given_x / var_y
    n_groups = int(s.groupby(level=0).ngroups)
    multi = int((group_sizes > 1).sum())
    print(f"{name:12s} {len(y):6d} {var_y:8.4f} {e_var_given_x:14.4f} "
          f"{r2_ceil:8.4f}  groups={n_groups} multi-row-groups={multi}")
    print(f"            -> max achievable R2 with {key}-only features "
          f"= {r2_ceil:.3f};  0.8 target "
          f"{'REACHABLE' if r2_ceil >= 0.8 else 'IMPOSSIBLE (noise-limited)'}")
