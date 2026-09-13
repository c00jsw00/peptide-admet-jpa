# Usage Guide for Peptide ADMET Prediction Model

> **2026-08 v4.2**: this guide reflects the **real-data + ESMC + MoLFormer**
> pipeline — four endpoints (Hemolysis, Half-life, Caco-2, PAMPA/MDCK),
> dual-modality features (sequence endpoints: 428-dim classical + 1,152-dim
> frozen ESMC-600M = 1,580-dim; molecular endpoints: 2,265-dim RDKit +
> 768-dim frozen MoLFormer-XL CLS = 3,033-dim), Huber-loss regression,
> Half-life deduplicated to 768 unique sequences, and four independent
> single-task models. See `README.md` for the full rationale and
> `PREDICTOR_SUMMARY.md` for the endpoint summary.

## Installation

```bash
# main env (Python 3.11, CPU torch is fine) — needs rdkit + sentencepiece
uv pip install --python .venv/Scripts/python.exe rdkit sentencepiece

# OPTIONAL ESMC env (Python >= 3.12) — only needed to (a) regenerate the
# frozen ESMC embeddings or (b) predict a brand-new sequence not in the
# committed cache.  Retrain + predict on cached sequences do NOT need it.
# Build per the header of esmc_embed.py (uv venv --python 3.12 + esm@git main).
# NOTE: MoLFormer (v4.2 molecular embeddings) runs in the MAIN env (.venv),
# no separate 3.12 env needed.
```

## Pipeline (train from scratch)

```bash
python prepare_pepadmet_data.py            # load + clean the 4 real CSVs -> data/pepadmet_*.csv

# OPTIONAL: regenerate the frozen embeddings (first run, or if the data
# changed).  Already-committed npz works as-is.
.venv-esmc/Scripts/python.exe esmc_embed.py      # v4.1 sequence embeddings -> data/esmc/*.npz
.venv/Scripts/python.exe molformer_embed.py      # v4.2 molecular embeddings -> data/molformer/*.npz

python train_pepadmet_model.py --epochs 80 --seed 42   # -> models_v4/ (weights + metrics.json)
```

## Quick Start (Python API)

```python
from peptide_admet_predictor import load_endpoint_models, build_features

models = load_endpoint_models(['Hemolysis', 'Half_life'], model_root='models_v4')

# one sequence -> both sequence endpoints (ESMC resolved from the committed
# cache for known sequences; a novel sequence shells out to .venv-esmc)
seq = "ACDEFGHIKLMNPQRSTVWY"
X, _, _ = build_features('Hemolysis', [seq], [None])
p_hemo = models['Hemolysis'].predict(X)[0]          # P(hemolytic)

# one SMILES -> a molecular endpoint
smi = "CC(=O)N[C@@H](C)C(=O)N[C@@H](CCCNC(=N)N)C(=O)O"
X, _, _ = build_features('Caco2', [None], [smi])
caco2 = models['Caco2'].to_readable(models['Caco2'].predict(X))[0]  # logPapp
```

`predict_rows(rows, endpoints, model_root)` is the batch driver: pass a list
of dicts each with an optional `sequence` and/or `smiles`; it returns one
result dict per row (`{'value', 'unit', 'ok'}` per endpoint).

## Command Line

```bash
# sequence endpoints (Hemolysis + Half_life)
python peptide_admet_predictor.py --sequence "ACDEFGHIKLMNPQRSTVWY"

# molecular endpoints (Caco2 + PAMPA_MDCK)
python peptide_admet_predictor.py --smiles "CC(=O)N[C@@H](C)C(=O)N[C@@H](CCCNC(=N)N)C(=O)O"

# a batch CSV (optional 'sequence' and/or 'smiles' columns), JSON out
python peptide_admet_predictor.py --csv data/test_candidates.csv --out results.json

# restrict which endpoints to run
python peptide_admet_predictor.py --sequence "ACDEF..." --endpoints Hemolysis
```

The CLI auto-routes: a `--sequence` runs the two sequence endpoints, a
`--smiles` runs the two molecular endpoints; a `--csv` runs all four.

## Model Files (models_v4/, committed)

```
models_v4/<endpoint>/admet_mlp.pt   # PyTorch weights + architecture metadata
models_v4/<endpoint>/scaler.pt      # StandardScaler (fitted on train only)
models_v4/<endpoint>/metrics.json   # measured metrics (both splits) + leakage audit
models_v4/summary.json              # 4-endpoint aggregate

# frozen embedding caches (committed; no extra env needed to USE them)
data/esmc/esmc_emb_hemolysis.npz          # (8719, 1152) float32, mean-pooled ESMC-600M
data/esmc/esmc_emb_half_life.npz          # (1763, 1152) float32, mean-pooled ESMC-600M
data/molformer/molformer_emb_caco2.npz    # (7429, 768)  float32, CLS token MoLFormer-XL
data/molformer/molformer_emb_pampa_mdck.npz # (7283, 768) float32, CLS token MoLFormer-XL
```

`<endpoint>` ∈ {`hemolysis`, `half_life`, `caco2`, `pampa_mdck`}.
The predictor's `EndpointModel.predict(X)` applies the shipped scaler
internally — pass **raw** features (do not pre-scale).

## Feature Engineering

| Modality | Dim | Content |
|---|---|---|
| Sequence (classical) | 428 | AAC 20 + DPC 400 + physchem 8 (MW proxy, avg hydropathy, net charge @ pH 7, pI estimate, GRAVY, hydrophobic/charged fraction) |
| Sequence (+ESMC, v4.1) | 1,580 | 428 classical **concat** 1,152-dim frozen ESMC-600M (attention-mask mean-pooled) |
| Molecular (RDKit) | 2,265 | RDKit 2D descriptors 217 + Morgan fingerprint (radius 2) 2,048 |
| Molecular (+MoLFormer, v4.2) | 3,033 | 2,265 RDKit **concat** 768-dim frozen MoLFormer-XL CLS token |

Identical code in training and inference (`feature_extractor.py` + the two
embedding generators). Both ESMC and MoLFormer are **frozen** (no fine-tune,
no gradient) — the npz files are precomputed and committed.

> **v4.2 Half-life dedup**: the trainer collapses the 1,763 raw rows to
> 768 unique sequences (same sequence → average of its log10 half-lives)
> before splitting, so the headline metric is reported at the **sequence
> level**. The prepared CSV (`data/pepadmet_half_life.csv`) still holds all
> 1,763 rows; dedup happens inside `train_pepadmet_model.py`.

## Interpretation Guide

| Endpoint | Kind | Higher value means |
|---|---|---|
| Hemolysis | binary | higher P = more hemolytic (worse) |
| Half-life | regression (log10 s) | longer plasma half-life |
| Caco-2 | regression (logPapp) | higher = better intestinal permeability |
| PAMPA/MDCK | regression (logPapp) | higher = better apparent permeability |

These four datasets are disjoint molecules in two different feature spaces,
so the models are **not** combined into a composite score.

## Best Practices

- Predicting a **novel sequence** (not in `data/esmc/*.npz`) shells out to
  `.venv-esmc` to embed it once; the first call is slow (loads the 573.6M
  model), subsequent cached sequences are instant.
- Predicting a **novel SMILES** (not in `data/molformer/*.npz`) shells out
  to `.venv` (main env, MoLFormer-XL 60M) to embed it once; cached SMILES
  are instant.
- Do not extrapolate far outside the trained length window (~4–120 aa, most
  sequences ~19).
- Headline metrics are on the **homology-controlled** (sequence) /
  unique-SMILES (molecular) split; the random split is a leakage contrast only.
- For the molecular endpoints, near-isomer (same chemistry, different SMILES)
  pairs may cross the split boundary — a documented SMILES-only limitation.
- Regression endpoints use **Huber loss** (v4.2): predictions in log-space
  (log10 seconds / logPapp); convert back with `to_readable()` for physical
  units.

## Troubleshooting

**`ESMC-600M env not found`** — a novel sequence needs `.venv-esmc` next to
the script; cached sequences and the molecular endpoints do not.
**`MoLFormer env not found` / slow first novel-SMILES call** — the main env
(`.venv`) needs `transformers` + `sentencepiece` (installed via
requirements.txt); the first call loads the 60M model.
**Invalid sequence** — standard 20 amino acids only for the sequence endpoints.
**`ModuleNotFoundError: rdkit`** — install rdkit (main env) for the molecular path.
**Model files missing** — run the pipeline commands above.

## References

1. `README.md` — pipeline + leakage discussion + version history.
2. `PREDICTOR_SUMMARY.md` — endpoint summary + measured metrics.
3. `peptide_admet_manuscript_jcim.md` — manuscript.
4. ESMC (ESM Cambrian), Biohub — `biohub/ESMC-600M`.
5. MoLFormer-XL, IBM Research — `ibm-research/MoLFormer-XL-both-10pct`.
6. Chemit797/PepADMET-Dataset — source of the four real endpoints.

**Version**: 4.2
**Last Updated**: 2026-08-26
**Author**: OpenClaw Team
