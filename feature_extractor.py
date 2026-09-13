#!/usr/bin/env python3
"""
feature_extractor.py
====================

Shared, deterministic feature extraction for the 4-endpoint PepADMET pipeline.

Two modalities (the four endpoints split cleanly into these two):

* **Sequence modality** — Hemolysis (binary) and Half-life (regression).
  Both ship clean 20-standard-amino-acid one-letter sequences.  We reuse the
  established 428-dim layout so the numbers stay comparable to the old
  synthetic pipeline:

      [AAC (20, AMINO_ACIDS order)
       DPC (400, row-major over (a,b) in AMINO_ACIDS order)
       physchem (8): mw, avg_hydropathy, hydropathy_range,
                     net_charge, pi_estimate, gravy,
                     hydrophobic_ratio, charged_ratio]

* **Molecular modality** — Caco-2 (regression) and PAMPA/MDCK (regression).
  These datasets (PepLand / CycPeptMPDB) are non-standard, mostly-cyclic
  peptides whose "Sequence" column is a *residue-name list* (MEL, DP, DL,
  3-PYRIDYLETHYL_GLY, ...) that the 20-standard-AA encoder cannot read.
  Per the design decision for this rebuild, we encode the molecule itself from
  its SMILES via RDKit:

      [217 standard 2D descriptors (Descriptors.CalcMolDescriptors, fixed
       order = rdkit.Chem.Descriptors._descList)
       2048-bit Morgan fingerprint (radius 2)]  -> 2265 dims

  Rows whose SMILES fails to parse contribute an all-zero vector and are
  counted (never silently dropped from the label table), so the feature dim
  is stable and the data loss is auditable.

The sequence extractor is pure numpy (no RDKit); the molecular extractor
requires rdkit.  Both return float64 numpy arrays and are fully
deterministic given the input and (for molecular) the RDKit version, which we
record in the saved model metadata.
"""

from typing import List, Optional, Sequence as SeqType

import numpy as np

# --------------------------------------------------------------------------- #
# Amino-acid alphabet (canonical order — must match the old pipeline exactly)
# --------------------------------------------------------------------------- #
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
N_AA = len(AMINO_ACIDS)
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

SEQ_DIM = 428  # 20 AAC + 400 DPC + 8 physchem

# Physicochemical feature offsets inside the 428-dim vector
I_MW, I_HYDRO, I_HYDRO_R, I_NETQ, I_PI, I_GRAVY, I_HR, I_CR = 420, 421, 422, 423, 424, 425, 426, 427

# --- physicochemical constants (identical to the old PeptideFeatureExtractor)
HYDROPATHY = {
    'I': 4.5, 'V': 4.2, 'L': 3.8, 'F': 2.8, 'C': 2.5,
    'M': 1.9, 'A': 1.8, 'G': -0.4, 'T': -0.7, 'S': -0.8,
    'W': -0.9, 'Y': -1.3, 'P': -1.6, 'H': -3.2, 'E': -3.5,
    'Q': -3.5, 'D': -3.5, 'N': -3.5, 'K': -3.9, 'R': -4.5
}
CHARGE = {'R': 1.0, 'K': 1.0, 'H': 0.1, 'D': -1.0, 'E': -1.0}

_AA_HYDRO = np.array([HYDROPATHY[aa] for aa in AMINO_ACIDS])
_AA_CHARGE = np.array([CHARGE.get(aa, 0.0) for aa in AMINO_ACIDS])
_CHARGE_IDX = list(AA_TO_IDX[a] for a in CHARGE)


def sequence_features(sequences: SeqType[str]) -> np.ndarray:
    """
    Vectorized 428-dim feature extraction for clean 20-AA sequences.

    Produces exactly the legacy layout:
        [AAC (20)  DPC (400)  physchem (8)]

    Every character of every sequence must be in the 20 standard AAs (callers
    validate upstream; a non-standard residue raises ValueError here rather
    than being silently dropped).
    """
    n = len(sequences)
    X = np.zeros((n, SEQ_DIM), dtype=np.float64)

    codes = np.empty((n,), dtype=object)
    for i, seq in enumerate(sequences):
        s = ''.join(seq)
        bad = [c for c in s if c not in AA_TO_IDX]
        if bad:
            raise ValueError(
                f'sequence_features: non-standard residue(s) {sorted(set(bad))} '
                f'in sequence {s!r} — molecular-modality endpoints must go '
                f'through molecule_features instead')
        codes[i] = np.fromiter((AA_TO_IDX[c] for c in s), dtype=np.int64)

    for i in range(n):
        c = codes[i]
        L = len(c)
        if L == 0:
            continue
        # AAC
        X[i, 0:20] = np.bincount(c, minlength=20) / L
        # DPC
        if L >= 2:
            di = c[:-1] * 20 + c[1:]
            X[i, 20:420] = np.bincount(di, minlength=400) / (L - 1)
        # physchem
        hydro = _AA_HYDRO[c]
        charge = _AA_CHARGE[c]
        mw = L * 110
        avg_hydro = hydro.mean()
        hydro_range = hydro.max() - hydro.min()
        net_charge = charge.sum()
        basic = int((c == AA_TO_IDX['R']).sum() + (c == AA_TO_IDX['K']).sum())
        acidic = int((c == AA_TO_IDX['D']).sum() + (c == AA_TO_IDX['E']).sum())
        pi_est = 7.0 + (basic - acidic) / L * 2
        gravy = hydro.sum() / L
        hydro_ratio = (hydro > 0).mean()
        charged_ratio = np.isin(c, _CHARGE_IDX).mean()
        X[i, 420:428] = [mw, avg_hydro, hydro_range,
                         net_charge, pi_est, gravy,
                         hydro_ratio, charged_ratio]
    return X


# --------------------------------------------------------------------------- #
# Molecular modality (RDKit)
# --------------------------------------------------------------------------- #
def _lazy_rdkit():
    import warnings
    warnings.filterwarnings('ignore')
    from rdkit import Chem
    from rdkit.Chem import Descriptors, DataStructs
    from rdkit.Chem import rdFingerprintGenerator as RFG
    from rdkit import rdBase
    return Chem, Descriptors, DataStructs, RFG, rdBase.rdkitVersion


MORGAN_RADIUS = 2
MORGAN_BITS = 2048

# Resolved lazily so importing this module never requires rdkit.
_DESC_NAMES: Optional[List[str]] = None
_RDKIT_VERSION: Optional[str] = None


def rdkit_version() -> str:
    global _RDKIT_VERSION
    if _RDKIT_VERSION is None:
        _RDKIT_VERSION = _lazy_rdkit()[4]
    return _RDKIT_VERSION


def descriptor_names() -> List[str]:
    """Fixed, deterministic descriptor order (rdkit.Chem.Descriptors._descList)."""
    global _DESC_NAMES
    if _DESC_NAMES is None:
        _Chem, _Desc, _DS, _RFG, _ver = _lazy_rdkit()
        _DESC_NAMES = [nm for (nm, _fn) in _Desc._descList]
    return list(_DESC_NAMES)


MOL_DIM = None  # filled in descriptor_names(); equals len(descriptor_names()) + MORGAN_BITS


def molecule_features(smiles: SeqType[str]) -> np.ndarray:
    """
    Vectorized RDKit molecular feature extraction.

    Returns (n, MOL_DIM) float64 where MOL_DIM = len(descriptor_names()) +
    MORGAN_BITS.  Each row is the concatenation of:

        [217 standard 2D descriptors (fixed order)
         2048-bit Morgan fingerprint (radius 2, 0/1)]

    A SMILES that is empty or fails to parse yields an all-zero row (the
    descriptor block is left zero and the Morgan block is zero).  Callers can
    detect unparseable rows independently with ``parseable_smiles_mask``.
    """
    names = descriptor_names()
    ddim = len(names)
    dim = ddim + MORGAN_BITS
    n = len(smiles)
    X = np.zeros((n, dim), dtype=np.float64)

    Chem, Descriptors, DataStructs, RFG, _ver = _lazy_rdkit()
    morgan_gen = RFG.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_BITS)
    for i, s in enumerate(smiles):
        if not s or (isinstance(s, float) and np.isnan(s)):
            continue
        try:
            mol = Chem.MolFromSmiles(str(s))
        except Exception:
            continue
        if mol is None:
            continue
        try:
            d = Descriptors.CalcMolDescriptors(mol)
            for j, nm in enumerate(names):
                v = d.get(nm)
                if isinstance(v, (int, float)) and np.isfinite(v):
                    X[i, j] = float(v)
        except Exception:
            pass
        fp = morgan_gen.GetFingerprint(mol)
        arr = np.zeros(MORGAN_BITS, dtype=np.float64)
        DataStructs.ConvertToNumpyArray(fp, arr)
        X[i, ddim:dim] = arr
    return X


def parseable_smiles_mask(smiles: SeqType[str]) -> np.ndarray:
    """Boolean mask: True where the SMILES parses to a non-null RDKit mol."""
    Chem, _, _, _, _ver = _lazy_rdkit()
    out = np.zeros(len(smiles), dtype=bool)
    for i, s in enumerate(smiles):
        if not s or (isinstance(s, float) and np.isnan(s)):
            continue
        try:
            out[i] = Chem.MolFromSmiles(str(s)) is not None
        except Exception:
            out[i] = False
    return out
