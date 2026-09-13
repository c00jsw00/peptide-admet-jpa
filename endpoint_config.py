#!/usr/bin/env python3
"""
endpoint_config.py
==================

Single source of truth for the endpoint set of the openclaw peptide ADMET
pipeline (v4.1 — real-data + ESMC edition).

v4.1 extends v4.0 (Chemit797/PepADMET-Dataset) with frozen **ESMC-600M**
(Biohub ESM Cambrian) sequence embeddings for the two sequence-modality
endpoints (Hemolysis, Half_life): the 428-dim classical sequence vector
(AAC+DPC+phys-chem) is concatenated with the 1152-dim ESMC embedding ->
1580-dim model input.  The embeddings are precomputed offline
(`esmc_embed.py`, Python>=3.12 env) and cached in `data/esmc/*.npz`, so
training and inference run on CPU with no ESMC dependency.  Caco-2 / PAMPA
stay purely molecular (their source "sequences" are non-standard
peptidomimetic residue lists, ~0.2% standard AA — not embeddable).

v4.0 replaces the synthetic 9-endpoint set with the four endpoints requested
for the **Chemit797/PepADMET-Dataset** release (cleaned ``整理/`` tables):

  * Hemolysis      — binary     (hemolytic 1 / non-hemolytic 0)   [SEQUENCE modality]
  * Half-life      — regression (plasma half-life, seconds)        [SEQUENCE modality]
  * Caco-2         — regression (apparent permeability, logPapp)   [MOLECULAR modality]
  * PAMPA/MDCK     — regression (PAMPA apparent permeability)      [MOLECULAR modality]

Two modalities
--------------
The four datasets are **disjoint molecules** that fall into two feature spaces:

  * ``sequence``   — hemolysis & half-life ship clean one-letter 20-AA peptide
    sequences -> the 428-dim sequence feature vector (AAC+DPC+phys-chem).
  * ``molecular``  — caco-2 & pampa ship only usable SMILES (their ``Sequence``
    column is a non-standard CycPeptMPDB residue-name list, not a 20-AA string)
    -> the ~2265-dim RDKit molecular feature vector (2D descriptors + Morgan).

Each ``Endpoint`` therefore carries a ``modality`` so the trainer / predictor
can route a row to the right feature extractor and load the matching model.
The four datasets do not overlap, so the pipeline trains **four focused
single-task models**, one per endpoint (a single shared trunk would just learn
two disconnected zero-padded subspaces).

Data provenance & honesty
-------------------------
  * Hemolysis ``label`` and Half-life ``half_life_seconds`` are the dataset's
    curated measured values (see the source repo's ``dataset_stats`` /
    ``Final_Datasets_Overview``).  We do NOT fabricate or impute labels.
  * Caco-2 ``Permeability`` and PAMPA ``PAMPA`` are already log-scale apparent
    permeability (logPapp) values; we train on them directly.
  * Rows with a null/unparseable input for their modality are dropped at
    ingestion (never zero-filled into training) so no fake features enter the
    model.  Nulls are counted and reported in the data meta JSON.

Column naming
-------------
``column`` is the canonical internal name (also the CSV column the prepared
per-endpoint table writes).  ``source_column`` is the raw column in the
Chemit797 release we read from; ``seq_column`` / ``smiles_column`` name the
raw input feature column for the endpoint's modality.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional

# --------------------------------------------------------------------------- #
# Endpoint kinds
# --------------------------------------------------------------------------- #
KIND_BINARY = 'binary'
KIND_MULTICLASS = 'multiclass'
KIND_REGRESSION = 'regression'

# --------------------------------------------------------------------------- #
# Feature modalities
# --------------------------------------------------------------------------- #
MODALITY_SEQUENCE = 'sequence'    # clean 20-AA one-letter peptide string
MODALITY_MOLECULAR = 'molecular'  # SMILES -> RDKit 2D descriptors + Morgan

# Acceptable one-letter peptide sequence length window (sequence modality).
SEQUENCE_MIN_LEN = 4
SEQUENCE_MAX_LEN = 120
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

# --------------------------------------------------------------------------- #
# ESMC-600M (Biohub, ESM Cambrian) sequence embeddings  [v4.1]
# --------------------------------------------------------------------------- #
# The two sequence-modality endpoints (Hemolysis, Half_life) also receive a
# frozen 1152-dim ESMC-600M mean-pooled embedding, concatenated on to the
# 428-dim classical sequence vector (AAC+DPC+phys-chem) -> 1580-dim input.
# Embeddings are precomputed offline (esmc_embed.py, Python>=3.12 env) and
# cached in data/esmc/*.npz so training / inference never need the ESMC env.
ESMC_MODEL = 'biohub/ESMC-600M'
ESMC_DIM = 1152
ESMC_CACHE_DIR = 'data/esmc'
# per-endpoint cached embedding file (slug = name.lower().replace(' ', '_'))
def esmc_cache_path(name: str) -> str:
    slug = name.lower().replace(' ', '_')
    return f'{ESMC_CACHE_DIR}/esmc_emb_{slug}.npz'


# --------------------------------------------------------------------------- #
# MoLFormer-XL molecular embeddings  [v4.2]
# --------------------------------------------------------------------------- #
# The two molecular-modality endpoints (Caco-2, PAMPA/MDCK) now ALSO receive a
# frozen MoLFormer-XL (IBM, ~60M params, hidden 768) CLS-token embedding,
# concatenated on to the 2265-dim RDKit vector (2D descriptors + Morgan) ->
# 3033-dim model input.  This is the v4.2 "new weapon" for the two
# permeability endpoints: a pretrained SMILES transformer representation added
# to the hand-engineered 2D descriptors.  Embeddings are precomputed offline
# (molformer_embed.py, plain CPU torch in the main .venv) and cached in
# data/molformer/*.npz so training / inference never need transformers at
# runtime.  (The MoLFormer weights are in the main .venv, not .venv-esmc.)
MOLEFORMER_MODEL = 'ibm-research/MoLFormer-XL-both-10pct'
MOLEFORMER_DIM = 768
MOLEFORMER_CACHE_DIR = 'data/molformer'
# per-endpoint cached embedding file (slug = name.lower().replace(' ', '_'))
def molformer_cache_path(name: str) -> str:
    slug = name.lower().replace(' ', '_')
    return f'{MOLEFORMER_CACHE_DIR}/molformer_emb_{slug}.npz'


@dataclass(frozen=True)
class Endpoint:
    """One prediction target."""
    name: str                 # canonical name (also the CSV column)
    kind: str                 # 'binary' | 'multiclass' | 'regression'
    modality: str = MODALITY_SEQUENCE   # 'sequence' | 'molecular'
    num_classes: int = 1      # for multiclass
    # raw-column mapping into the Chemit797 release
    source_file: str = ""     # relative path under the release root
    source_column: str = ""   # raw label column
    seq_column: Optional[str] = None     # raw sequence column (sequence modality)
    smiles_column: Optional[str] = None  # raw SMILES column (molecular modality)
    # label transform (regression only)
    target_transform: str = 'identity'   # 'identity' | 'log10' | 'log10plus'
    raw_units: str = ''                  # human units of the RAW source column
    description: str = ""
    # composite-score role
    in_composite: bool = True      # does it enter the overall "risk" score?
    higher_is_worse: bool = True   # for composite aggregation direction
    # direction hint used to fold a probability/estimate into [0,1] risk
    risk_direction: str = 'prob'   # 'prob' (P(positive)) | 'class' | 'value'
    # v4.1: append the frozen ESMC-600M embedding to the sequence features
    esmc: bool = False             # True -> input = 428-dim seq + 1152-dim ESMC
    # v4.2: append the frozen MoLFormer-XL embedding to the molecular features
    molformer: bool = False        # True -> input = 2265-dim mol + 768-dim MoLFormer


# --------------------------------------------------------------------------- #
# The 4 endpoints (v4.0 — Chemit797/PepADMET-Dataset)
# --------------------------------------------------------------------------- #
ENDPOINTS: List[Endpoint] = [
    Endpoint('Hemolysis', KIND_BINARY, MODALITY_SEQUENCE, 1,
             source_file='整理/hemolysis_unified/hemolysis_unified.csv',
             source_column='label',
             seq_column='sequence_std',
             description='Hemolytic activity of the peptide (1 = hemolytic, 0 = non-hemolytic)',
             in_composite=True, higher_is_worse=True, risk_direction='prob',
             esmc=True),

    Endpoint('Half_life', KIND_REGRESSION, MODALITY_SEQUENCE, 1,
             source_file='整理/half_life_unified/half_life_final_minimal.csv',
             source_column='half_life_seconds',
             seq_column='sequence',
             target_transform='log10',
             raw_units='seconds',
             description='Plasma half-life (seconds); modelled in log10 space',
             in_composite=True, higher_is_worse=False, risk_direction='value',
             esmc=True),

    Endpoint('Caco2', KIND_REGRESSION, MODALITY_MOLECULAR, 1,
             source_file='整理/caco2_out/caco2_unified.csv',
             source_column='Permeability',
             smiles_column='SMILES',
             target_transform='identity',
             raw_units='logPapp',
             description='Caco-2 apparent permeability (logPapp); molecular features',
             in_composite=True, higher_is_worse=False, risk_direction='value',
             molformer=True),

    Endpoint('PAMPA_MDCK', KIND_REGRESSION, MODALITY_MOLECULAR, 1,
             source_file='整理/permeability_out/permeability_unified.csv',
             source_column='PAMPA',
             smiles_column='SMILES',
             target_transform='identity',
             raw_units='logPapp',
             description='PAMPA apparent permeability (logPapp); molecular features',
             in_composite=True, higher_is_worse=False, risk_direction='value',
             molformer=True),
]

ENDPOINT_NAMES: List[str] = [e.name for e in ENDPOINTS]
ENDPOINT_BY_NAME: Dict[str, Endpoint] = {e.name: e for e in ENDPOINTS}
N_ENDPOINTS = len(ENDPOINTS)

# Convenience splits
BINARY_NAMES = [e.name for e in ENDPOINTS if e.kind == KIND_BINARY]
MULTICLASS_NAMES = [e.name for e in ENDPOINTS if e.kind == KIND_MULTICLASS]
REGRESSION_NAMES = [e.name for e in ENDPOINTS if e.kind == KIND_REGRESSION]
SEQUENCE_NAMES = [e.name for e in ENDPOINTS if e.modality == MODALITY_SEQUENCE]
MOLECULAR_NAMES = [e.name for e in ENDPOINTS if e.modality == MODALITY_MOLECULAR]
ESMC_NAMES = [e.name for e in ENDPOINTS if e.esmc]
MOLEFORMER_NAMES = [e.name for e in ENDPOINTS if e.molformer]

# Composite-score inputs (the endpoints folded into the overall risk number)
COMPOSITE_NAMES = [e.name for e in ENDPOINTS if e.in_composite]

# Classical (non-embedding) per-modality feature widths, for docs / checks.
SEQ_FEATURE_DIM = 428      # 20 AAC + 400 DPC + 8 phys-chem
MOL_FEATURE_DIM = 2265     # 217 RDKit 2D descriptors + 2048-bit Morgan r=2


def input_dim_for(name: str) -> int:
    """Effective model input dim for an endpoint (embeddings included when set)."""
    e = ENDPOINT_BY_NAME[name]
    if e.modality == MODALITY_SEQUENCE:
        return SEQ_FEATURE_DIM + (ESMC_DIM if e.esmc else 0)
    return MOL_FEATURE_DIM + (MOLEFORMER_DIM if e.molformer else 0)


def check_esmc_cache(root: str = '.'):
    """Raise if any ESMC endpoint's precomputed embedding cache is missing."""
    from pathlib import Path
    missing = [n for n in ESMC_NAMES
               if not (Path(root) / esmc_cache_path(n)).exists()]
    if missing:
        raise FileNotFoundError(
            f'missing ESMC embedding cache for {missing}: '
            f'run `.venv-esmc/Scripts/python.exe esmc_embed.py` '
            f'(expected {esmc_cache_path(missing[0])})')


def check_molformer_cache(root: str = '.'):
    """Raise if any MoLFormer endpoint's precomputed embedding cache is missing."""
    from pathlib import Path
    missing = [n for n in MOLEFORMER_NAMES
               if not (Path(root) / molformer_cache_path(n)).exists()]
    if missing:
        raise FileNotFoundError(
            f'missing MoLFormer embedding cache for {missing}: '
            f'run `.venv/Scripts/python.exe molformer_embed.py` '
            f'(expected {molformer_cache_path(missing[0])})')


def endpoint(name: str) -> Endpoint:
    return ENDPOINT_BY_NAME[name]


def n_multiclass() -> int:
    return sum(1 for e in ENDPOINTS if e.kind == KIND_MULTICLASS)


if __name__ == '__main__':
    print(f'{N_ENDPOINTS} endpoints (v4.0 real-data):')
    for e in ENDPOINTS:
        extra = f' (num_classes={e.num_classes})' if e.kind == KIND_MULTICLASS else ''
        comp = '' if e.in_composite else '  [not in composite]'
        print(f'  {e.name:12s} {e.kind:11s} {e.modality:10s}{extra}{comp}')
