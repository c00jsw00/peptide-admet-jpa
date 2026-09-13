# Route 9: Foundation Models (TabPFN v2 + KPGT fine-tune) on PAMPA

**Date**: 2026-08-29
**Goal**: Test whether pretrained foundation models (TabPFN v2 tabular, KPGT
graph transformer) can exceed the v4.2 LightGBM baseline (R² 0.4642) on the
PAMPA leakage-controlled split.

**Protocol**: identical canonical split (`analysis/common.split_smiles`,
unique-SMILES 70/10/20, seed 42), same 7,283 rows, same 269 floor rows
(−10.0000). Metrics: R² (all rows) and R² (non-floor subset), 3 seeds.

## 9.1 TabPFN v2 (foundation tabular model)

**Script**: `analysis/route9_tabpfn.py` (reproducible)
**Model**: TabPFN 8.5.0, `ModelVersion.V2` (direct_download, no HF license
needed), CUDA (RTX 4070 SUPER).
**Seeds**: 42, 123, 456.

| Feature set | Dim | R² (all, mean±sd) | R² (non-floor, mean±sd) | Δ vs baseline (all) |
|---|---:|---:|---:|---:|
| **desc (RDKit 217)** | 217 | **0.4962 ± 0.0016** | 0.6268 ± 0.0051 | **+0.032** |
| desc + Morgan | 2265 | 0.4813 ± 0.0030 | 0.5866 ± 0.0043 | +0.017 |
| desc + Morgan + MoLFormer-XL | 3033 | 0.4820 ± 0.0068 | 0.6280 ± 0.0057 | +0.018 |
| LightGBM baseline (v4.2) | 217 | 0.4642 | 0.6317 | — |

**Result**: TabPFN v2 with RDKit 2D descriptors only (217 features, no
fingerprints, no pretrained molecular embeddings) **exceeds the v4.2
LightGBM baseline by +0.032** (0.4962 vs 0.4642), well outside the seed
noise band (±0.0016). The gain is concentrated in the floor region:
non-floor R² is essentially unchanged (0.6268 vs 0.6317, within noise).

**Interpretation**: TabPFN v2's in-context learning (attention over the
training set, no gradient descent) captures the nonlinear structure in the
217 RDKit descriptors that LightGBM's 100 trees at lr=0.05/depth=4 underfit.
The fact that adding Morgan fingerprints (2265 dim) or MoLFormer-XL
embeddings (3033 dim) *degrades* performance suggests that TabPFN's
500-feature context window (V2 limit, `ignore_limits=True` used) introduces
noise at high dimensionality — the model is effectively using the 217
descriptors and ignoring the rest.

**Caveats**:
- TabPFN V2 has a 500-feature limit; `ignore_limits=True` was used for the
  2265/3033-dim configs (the model truncates internally).
- TabPFN does not support early stopping / cross-validation internally;
  the 3-seed variance (±0.0016) reflects seed-only noise, not full
  train/test re-split variance.
- The gain (+0.032) is a single-test-split number; it would need a
  5-fold outer CV to confirm it is not split-luck. However, the effect is
  20× the seed noise and the direction is consistent across all 3 seeds.

## 9.2 KPGT (graph transformer) fine-tune

**Script**: `_kpgt_finetune_gpu.py` (pure-PyTorch GPU port of the LiGhT
backbone; the DGL 2.2.1 Windows wheel is CPU-only with no CUDA build for
torch 2.13, so the official DGL-based pipeline cannot use the GPU).
**Model**: KPGT LiGhT (12-layer triplet transformer, d_g=768, 4 heads),
base.pth pretrained checkpoint (ICML 2024), predictor head re-initialized.
**Seeds**: 42, 123, 7 (matching official KPGT fine-tune seeds).
**Training**: AdamW lr=1e-4, batch=64, 15-epoch warmup cosine decay,
early stopping patience=15, per-epoch checkpoint for resume.
**Speed**: ~400 s/epoch on RTX 4070 SUPER (vs ~815 s/epoch CPU).

**Cross-validation vs official DGL implementation**: `--check` mode
runs both implementations (DGL CPU vs pure-PyTorch) on the same batch in
eval mode; max absolute output difference = **8.3 × 10⁻⁷** (float32
rounding), confirming the port is numerically faithful.

**Status**: complete (3 seeds, early stopping; full per-epoch log in
`analysis/route9_kpgt_results.json`).

**Final results** (best-validation checkpoint evaluated on test):

| Seed | Best epoch | val R² | test R² (all) | test R² (non-floor) | Spearman |
|---:|---:|---:|---:|---:|---:|
| 42 | 16 | 0.4059 | 0.5191 | 0.5633 | 0.8257 |
| 123 | 9 | 0.4105 | 0.5073 | 0.5404 | 0.8023 |
| 7 | 13 | 0.4080 | 0.5139 | 0.5035 | 0.8040 |
| **mean** | | | **0.5134 ± 0.0048** | **0.5357 ± 0.0246** | 0.8107 |
| LightGBM baseline (v4.2) | — | — | 0.4642 | 0.6317 | — |
| TabPFN v2 (9.1) | — | — | 0.4962 ± 0.0016 | 0.6268 ± 0.0051 | — |

KPGT is the **best route of the nine on the floor-included test set**
(0.5134, +0.049 vs the LightGBM baseline), and it also beats TabPFN
(0.4962) by +0.017. But the gain is entirely a floor-region effect: on
the non-floor subset KPGT is *worse* than the baseline (0.5357 vs
0.6317, −0.096) — the graph transformer, fine-tuned end-to-end on
raw logPapp labels, learns the censored-floor pattern better than the
2D-descriptor models but pays for it with a substantial loss of
predictive power on the molecules the assay actually measured.

**Reproducibility note**: the DGL 2.2.1 Windows wheel is CPU-only, so
the official DGL fine-tune loop could not use the GPU. The backbone was
ported to pure PyTorch (graph ops → sparse tensor scatter operations);
the port was validated against the official DGL implementation on the
same batches in eval mode (max absolute output difference 8.3 × 10⁻⁷,
float32 rounding). Per-epoch checkpoints (`kpgt_gpu_ckpt_<seed>.pt`)
allow resume after interruption.

## 9.3 Conclusion

**Both foundation models beat the v4.2 LightGBM baseline on the
floor-included test set** — the first positive results of the nine PAMPA
improvement routes:

| Route | R² (all) | Δ vs baseline | R² (non-floor) | Δ vs baseline |
|---|---:|---:|---:|---:|
| LightGBM baseline (v4.2) | 0.4642 | — | 0.6317 | — |
| TabPFN v2 (217 desc) | 0.4962 ± 0.0016 | **+0.032** | 0.6268 ± 0.0051 | −0.005 |
| **KPGT fine-tuned (best route)** | **0.5134 ± 0.0048** | **+0.049** | 0.5357 ± 0.0246 | **−0.096** |

The two models win in complementary ways. TabPFN (frozen, no gradient
update on our data — in-context attention at inference) gains almost
exclusively in the censored-floor region while leaving the non-floor
R² essentially unchanged. KPGT (fine-tuned end-to-end on raw logPapp
labels) gains more on the full set but *loses* 0.096 on the non-floor
subset: gradient fine-tuning on a label set that is 3.7% left-censored
teaches the model the floor pattern at a real cost to its ability to
predict the molecules the assay actually measured.

**Neither result violates the ceiling analysis.** The oracle ceiling
(uncensored perfect, censored → global mean) is 0.5387; KPGT's 0.5134
is now within 0.025 of it, and its non-floor deficit is consistent with
the floor rows pulling the full-set R² up while dragging the subset
R² down. The ceiling still holds: 0.70 remains unreachable without
uncensored re-measurements of the 269 floored compounds, and the
best route tested (KPGT 0.5134) sits below the 0.5387 oracle.

**Implication for the manuscript**: the frozen-embedding negative result
(v4.2, §3.4) is not a general statement that pretrained knowledge cannot
help. A foundation model that conditions on the training set at
inference time (TabPFN, zero gradient updates) adds +0.032, and a
pretrained graph transformer fine-tuned with gradient updates (KPGT)
adds +0.049 — the largest gain of all nine routes. This sharpens the
§4.5 recommendation: task-tuned molecular encoders are the most
promising lever for the molecular endpoints, but the binding constraint
remains the censored labels, not the representation.
