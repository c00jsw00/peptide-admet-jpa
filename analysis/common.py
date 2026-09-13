"""Shared setup for the PAMPA R2-ceiling analysis scripts.

Each experiment script in this directory runs from the repo root by default.
Importing this module re-roots the CWD to the repo root so the relative paths
(``data/pepadmet_pampa_mdck.csv`` and the feature cache) resolve no matter
where the script is launched from. Import it near the top of a script:

    import common  # noqa: F401  (re-roots CWD to the repo root)

It also defines the ONE canonical split that every experiment must use, so the
reported test R2 values are directly comparable to the committed v4.2 number
(0.4642). The split is a VERBATIM copy of ``split_molecular`` in
``train_pepadmet_model.py`` (seed 42, unique-SMILES random 70/10/20) -- if the
pipeline's split ever changes, change it here too.
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_REPO_ROOT)
# Scripts live in analysis/ so Python puts analysis/ (not the repo root) on
# sys.path; add the repo root so `import admet_model` / `feature_extractor`
# / `endpoint_config` resolve.
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SEED = 42
DATA = os.path.join(_REPO_ROOT, 'data')
DATA_CSV = 'data/pepadmet_pampa_mdck.csv'
TARGET = 'PAMPA_MDCK'
FEAT_CACHE = '_pampa_feat_cache.npz'
FLOOR = -10.0000  # left-censoring floor of the assay (logPapp)


def load_data():
    """Return (smiles_list, y) for the PAMPA endpoint (7283 rows)."""
    df = pd.read_csv(DATA_CSV)
    return df['smiles'].astype(str).tolist(), df[TARGET].to_numpy(dtype=np.float64)


def split_smiles(smiles_list, seed=SEED):
    """VERBATIM copy of train_pepadmet_model.split_molecular (seed 42).

    Returns (tr, va, te) index arrays partitioning the rows by their
    unique-SMILES group (no SMILES appears in more than one split).
    """
    N = len(smiles_list)
    uniq, inv = np.unique(np.asarray(smiles_list, dtype=object), return_inverse=True)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    n_tr = int(round(len(uniq) * 0.70))
    n_va = int(round(len(uniq) * 0.10))
    tr_ids = set(perm[:n_tr].tolist())
    va_ids = set(perm[n_tr:n_tr + n_va].tolist())
    tr = np.array([i for i in range(N) if inv[i] in tr_ids], dtype=np.int64)
    va = np.array([i for i in range(N) if inv[i] in va_ids], dtype=np.int64)
    te = np.array([i for i in range(N)
                   if inv[i] not in tr_ids and inv[i] not in va_ids], dtype=np.int64)
    return tr, va, te
