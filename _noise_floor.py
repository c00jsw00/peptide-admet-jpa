import pandas as pd
import numpy as np

base = "data"
for name, file in [("Half_life","pepadmet_half_life.csv"),
                   ("Caco2","pepadmet_caco2.csv"),
                   ("PAMPA_MDCK","pepadmet_pampa_mdck.csv")]:
    df = pd.read_csv(f"{base}/{file}")
    key = "sequence" if "sequence" in df.columns else "smiles"
    counts = df.groupby(key)[name].agg(['count','std'])
    counts = counts.fillna(0)
    multi = counts[counts['count'] > 1]
    nonzero = counts[(counts['count'] > 1) & (counts['std'] > 1e-9)]
    overall_sd = df[name].std()
    print(f"{name}: n={len(df)}, unique keys={df[key].nunique()}, "
          f"keys with >1 row={len(multi)} (rows={int(multi['count'].sum())}), "
          f"nonzero-spread keys={len(nonzero)}")
    if len(nonzero):
        print(f"   dup-key spread: mean={nonzero['std'].mean():.4f} max={nonzero['std'].max():.4f} "
              f"(overall sd={overall_sd:.4f})")
    print(f"   y: min={df[name].min():.3f} max={df[name].max():.3f} sd={overall_sd:.4f} "
          f"skew={df[name].skew():.2f} nans={int(df[name].isna().sum())}")
