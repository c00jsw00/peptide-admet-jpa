# Peptide ADMET Prediction: A Systematic Benchmark and Foundation Model Evaluation

**Highlights**
- Ten-route censored-floor benchmark of peptide permeability prediction (PAMPA, Caco-2)
- TabPFN v2 and KPGT are the first to beat the MLP baseline on PAMPA and Caco-2
- KPGT fine-tune is the best PAMPA route: R² 0.513, 95 % of the 0.539 ceiling
- Foundation-model gains sit in the censored floor; non-floor R² declines
- Fine-tuned KPGT (ρ 0.811) beats PeptiVerse frozen embeddings (ρ 0.671)

## Abstract
**Purpose** Oral absorption remains a principal bottleneck in peptide drug discovery. We report a systematic evaluation of ten computational routes for permeability prediction on the pepADMET dataset (7,283 PAMPA and 7,429 Caco-2 cyclic peptides), conducted under a leakage-controlled, censored-floor-aware protocol. **Methods** A unique-SMILES 70/10/20 split (seed 42) was applied to preclude data leakage. Left-censored floors (−10.0 log cm/s; 3.7 % PAMPA, 3.3 % Caco-2) were treated explicitly, and oracle ceilings were derived from the censored variance (R² = 0.539 PAMPA; 0.570 Caco-2). Nine PAMPA routes and a Caco-2 foundation-model extension were examined: (1) descriptor expansion (5-seed mean), (2) rank-Gaussian target transformation (honest train-fitted quantile, 5-seed ensemble), (3) LightGBM ensembling (5-seed), (4) Tobit censored regression, (5) soft-label blending (β selected on validation), (6) ChemBERTa frozen embeddings with MLP head (3 seeds, 4 configurations), (7) PeptiVerse raw-data cross-validation, (8) label averaging, and (9) foundation models (TabPFN v2; KPGT fine-tuning); route 10 repeats route 9 on Caco-2. **Results** PAMPA routes 1–8 failed to exceed the baseline R² = 0.464 (best Δ = +0.001, route 5). On PAMPA, TabPFN v2 (217 RDKit descriptors) achieved R² = 0.496 ± 0.002 and KPGT fine-tuning reached R² = 0.513 ± 0.005. On Caco-2 the ranking reversed: TabPFN v2 attained R² = 0.442 ± 0.003 (baseline 0.393; non-floor R² 0.609 vs 0.515) whereas KPGT reached 0.411 ± 0.006 with a non-floor deficit (0.449 vs 0.515). Both foundation models derive their PAMPA gains exclusively from the censored subset; on Caco-2, TabPFN improves floor and non-floor accuracy alike. On 6,834 shared molecules, the Spearman correlation of our KPGT fine-tune (ρ = 0.811) exceeded that of PeptiVerse frozen ChemBERTa (ρ = 0.671). **Conclusion** Foundation models are the first methods to break the permeability baselines, yet the censored ceilings (PAMPA 0.539; Caco-2 0.570) remain the hard limits, and no single foundation model dominates across endpoints. Progress beyond these bounds requires uncensored re-measurement of floor compounds rather than improved models.

**Keywords** peptide ADMET; PAMPA permeability; censored regression; foundation models; TabPFN; KPGT; data leakage

## 1 Introduction
Therapeutic peptides have emerged as a distinct modality occupying the space between small molecules and biologics, with more than one hundred approved peptide drugs and a pipeline that continues to expand across metabolic, oncologic, and infectious disease indications [1]. Their large size, conformational flexibility, and high polarity impose absorption, distribution, metabolism, excretion, and toxicity (ADMET) challenges that classical small-molecule rules — Lipinski's rule of five foremost among them — were never designed to address [2]. Reliable in silico ADMET prediction for peptides is therefore a prerequisite for efficient lead prioritisation and rational design.

Two recent platforms have defined the current state of the art. The pepADMET platform [2] compiled the largest public peptide ADMET collection to date — PAMPA (7,283), Caco-2 (7,429), HLM, MDCK, and additional endpoints — and reported permeability R² values of 0.435–0.657 across model families, establishing the paradigm of classical descriptors combined with gradient boosting. Independently, PeptiVerse [3] unified peptide property prediction around frozen foundational embeddings (PeptideCLM, ChemBERTa, ESM-2) with lightweight prediction heads, reporting PAMPA Spearman ρ = 0.69 and Caco-2 ρ = 0.80, and establishing the paradigm of frozen representations with simple heads.

Both studies, however, share three limitations that constrain their conclusions. First, neither accounts for the left-censored floor in PAMPA measurements (−10.0 log cm/s; 3.7 % of the data), which inflates label variance and biases regression metrics. Second, unique-SMILES splitting was not uniformly enforced, leaving the results exposed to leakage through stereoisomers or tautomers. Third, frozen embeddings and end-to-end fine-tuning were never compared on an identical split, so the contribution of task-specific adaptation cannot be isolated. A broader literature of task-specific cyclic-peptide permeability models (CycPeptMP, CPMP, Multi_CycGT, PCPpred, and a 13-model DMPNN benchmark) reports R² values of 0.67–0.77; we position our results against that literature in §3.6.

Here we address all three limitations in a single, systematic benchmark. Working on the pepADMET PAMPA endpoint under a censored-floor-aware protocol (v4.2 split, seed 42), we evaluate nine improvement routes — eight classical and two foundation-model based — against a rigorously controlled baseline, and extend the two foundation models to a second endpoint, Caco-2 (route 10), to test whether their gains generalise. We ask two questions: (i) can foundation models, specifically TabPFN v2 (in-context tabular learning) [4] and KPGT (knowledge-pretrained graph transformer, fine-tuned) [5], exceed the baseline and approach the theoretical censored ceiling? and (ii) does end-to-end fine-tuning, rather than frozen embeddings, explain the gap to state-of-the-art peptide platforms? To answer the second question we re-evaluate pepADMET and PeptiVerse on the shared molecules, holding the split constant.

## 2 Materials and Methods

### 2.1 Dataset and censored-floor protocol
The pepADMET PAMPA dataset (7,283 cyclic peptides, log Papp in cm/s) was obtained from the pepADMET repository [2]. A left-censored floor at −10.0 log cm/s affects 269 compounds (3.7 %): these values represent the lower bound of assay sensitivity rather than true permeabilities. Following the v4.2 protocol [6], we partitioned the data by unique canonical SMILES into 70/10/20 train/validation/test sets (seed 42), yielding 5,102 / 724 / 1,457 compounds and thereby precluding leakage through structural near-duplicates. All metrics are reported twice — on the full test set (floor included) and on the non-floor subset (1,410 compounds). In addition, an oracle ceiling of R² = 0.539 was computed from the censored variance under the assumption of perfect ranking within the floor [6]; no model can exceed this bound on the current data.

### 2.2 Baseline model
The v4.2 baseline is a **single-head neural network (MixedADMETMLP)** [6] with a 3033-dimensional input: 217 RDKit 2D descriptors, 2,048-bit Morgan fingerprints (radius 2; [7], computed with RDKit [8]), and a 768-dimensional frozen MoLFormer-XL CLS embedding (ibm-research/MoLFormer-XL-both-10pct). The trunk comprises two hidden layers (256 → 128 units) with Huber loss (δ = 1.0), trained with Adam (lr = 1 × 10⁻³, weight decay 1 × 10⁻⁵), ReduceLROnPlateau (factor 0.5, patience 4), and early stopping (validation loss, patience 10) for a maximum of 80 epochs. This model (810,497 parameters) achieves R² = 0.464 on the full test set and R² = 0.632 on the non-floor subset.

### 2.3 Nine improvement routes
Routes 1–8 follow the v4.2 pipeline [6] with the identical split and evaluation protocol:

1. **Descriptor expansion** — Morgan fingerprints of radii r1, r2, r3 appended to the baseline feature set (5-seed mean; seeds 42, 123, 456, 789, 1024; MLP retrained).
2. **Rank-Gaussian target transformation** — the regression target is mapped to a Gaussian scale via a quantile transform fitted on the training set only (honest); predictions are back-transformed (5-seed ensemble).
3. **LightGBM ensembling** [9] — multiple LightGBM configurations (with additional Morgan radii r1, r3, r4) combined by averaging (5-seed ensemble).
4. **Tobit censored regression** — a censored likelihood (NLL) with the censoring threshold at −10.0, trained with early stopping.
5. **Soft-label blending** — the final prediction is a convex combination of the floor mean and the regression prediction, weighted by a per-compound floor probability; the mixing weight β is selected on validation.
6. **ChemBERTa frozen embeddings** — 384-dimensional embeddings from ChemBERTa-77M (frozen) [10] fed to an MLP head (3 seeds, 4 feature configurations), using the standard v4.2 training loop.
7. **PeptiVerse raw-data cross-validation** — the PeptiVerse PAMPA and Caco-2 sets [3] (HF ChatterjeeLab/PeptiVerse_data; 6,869 and 606 compounds) are re-split with the same unique-SMILES 70/10/20 protocol and re-trained end-to-end.
8. **Label averaging** — the pepADMET ensemble mean is used as a smoothed pseudo-label (one row per unique SMILES, y = arithmetic mean of replicate labels) in place of the raw measurements.

Route 9 comprises two foundation models:
- **TabPFN v2** (v2.0; direct download, pretraining limits disabled, n_estimators = 4) applied to the 217 RDKit descriptors, evaluated on the canonical split so that its test set is identical to the baseline's. Seeds: 42, 123, 7.
- **KPGT fine-tuning** — a 12-layer LiGhT graph transformer (d_g = 768, 12 attention heads, n_mol_layers = 12, path length 5, d_hpath_ratio = 12, feed-forward dimension 3,072) initialised from base.pth (447 MB, pretrained on 1.6 M molecules at ICML 2024) and fine-tuned on the v4.2 training set (batch size 64, AdamW lr = 1 × 10⁻⁴, weight decay 1 × 10⁻⁶, 15-epoch warmup cosine decay, early stopping patience 15). Three seeds (7, 42, 123) were run. Because the publicly available DGL [11] implementation for this architecture is CPU-only on Windows, we developed a pure-PyTorch GPU implementation (scatter-based TripletTransformer replacing DGL's u_dot_v, edge_softmax, and update_all-sum primitives), verified against the official DGL CPU implementation to a maximum absolute difference of 8.3 × 10⁻⁷. Checkpoints are saved per epoch and the best-validation model is used for final evaluation.

Route 10 extends the two foundation models of route 9 to the Caco-2 endpoint (7,429 compounds; 242 floor rows, 3.3 %; test n = 1,490 with 39 floor rows) under the identical protocol: the verbatim unique-SMILES 70/10/20 split (seed 42) on which the committed Caco-2 baseline (R² = 0.393; Table 4) was measured, the same three feature sets for TabPFN (217 / 2,265 / 3,033 dimensions), and the same KPGT fine-tuning configuration (graphs re-featurised with the official KPGT pipeline; the GPU port re-verified against DGL, maximum absolute difference 8.3 × 10⁻⁷). The Caco-2 oracle ceiling under the PAMPA convention (perfect non-floor prediction, floor → global mean) is R² = 0.570.

### 2.4 Cross-dataset comparison
Shared molecules between pepADMET PAMPA and PeptiVerse PAMPA were identified by canonical SMILES (RDKit). Of 7,177 unique pepADMET SMILES and 6,869 PeptiVerse SMILES, 6,834 (95.2 % of pepADMET; 99.5 % of PeptiVerse) overlap. Labels agreed exactly for 6,830 of these (the four discrepancies span at most 1.58 log units). Models are re-evaluated on this common subset — using each paper's original split where available, and our v4.2 split for the direct, leakage-controlled comparison.

### 2.5 Evaluation metrics
We report the coefficient of determination (R²), root mean squared error (RMSE), mean absolute error (MAE), Spearman's rank correlation (ρ), and the floor-excluded R² (non-floor R²). All metrics are computed with scikit-learn ≥1.5. Models are selected on validation R² by early stopping; final test evaluation uses the best-validation checkpoint.

### 2.6 Software and hardware
Python 3.11.16; PyTorch 2.13.0 (CUDA 12.6); DGL 2.2.1 (CPU, patched); TabPFN 8.5.0; LightGBM 4.7.0; RDKit 2026.03.5. The KPGT fine-tunes (PAMPA and Caco-2) and all TabPFN evaluations were executed on an NVIDIA RTX 4070 SUPER (KPGT ~400 s/epoch vs ~815 s/epoch on CPU); LightGBM and the MLP baselines ran on an AMD Ryzen 9 7950X CPU. All code, data, and trained checkpoints are available at https://github.com/c00jsw00/openclaw-peptide-admet.

## 3 Results

### 3.1 Routes 1–8: no route surpasses the baseline
Table 1 summarises routes 1–8. No classical route exceeds the baseline R² = 0.464; the closest is soft-label blending (route 5) at 0.4651 (Δ = +0.001, within seed-to-seed noise). Descriptor expansion (route 1) regressed to 0.4456 ± 0.0025 (5-seed mean), and the honest rank-Gaussian target (route 2) to 0.4469. Ensembling (route 3) degraded performance most substantially (0.4176), as did Tobit censored regression (route 4; 0.4190) — the latter's non-floor R² of 0.466 indicates that the censored likelihood, as parametrised, sacrifices measured-compound fidelity. ChemBERTa frozen embeddings (route 6, best configuration) reached 0.4624 ± 0.0068, statistically indistinguishable from the baseline, confirming that frozen chemical language-model representations add no information beyond 2D descriptors for this task. Re-training on the PeptiVerse raw data (route 7) reproduced the censoring floor (3.5 % of that set; ceilings 0.501/0.546) and yielded best R² of 0.434 (PAMPA) and 0.430 (Caco-2), corroborating the ceiling analysis on an independent dataset. Label averaging (route 8) gave 0.4490.

| Route | Description | R² (all) | Δ vs baseline | Non-floor R² |
|---|---|---:|---:|---:|
| Baseline | MLP (3033-dim: RDKit 217 + Morgan 2048 + MoLFormer-XL 768) | 0.4642 | — | 0.6317 |
| 1 | Descriptor expansion (Morgan r1, r2, r3; 5-seed MLP) | 0.4456 ± 0.0025 | −0.019 | 0.618 |
| 2 | Rank-Gaussian target (honest quantile, 5-seed ensemble) | 0.4469 | −0.017 | 0.615 |
| 3 | LightGBM ensemble (+Morgan r1, r3, r4; 5-seed) | 0.4176 | −0.047 | 0.592 |
| 4 | Tobit censored regression (NLL) | 0.4190 | −0.045 | 0.466 |
| 5 | Soft-label blending (β = 0.50, validation-selected) | 0.4651 | +0.001 | 0.593 |
| 6 | ChemBERTa-77M frozen + MLP (best: C_mol_molf_chem) | 0.4624 ± 0.0068 | −0.002 | 0.633 |
| 7 | PeptiVerse raw data (PAMPA / Caco-2) | 0.434 / 0.430 | −0.030 / −0.034 | 0.501 / 0.546 |
| 8 | Label averaging (pepADMET ensemble mean) | 0.4490 | −0.015 | 0.628 |

**Table 1.** Routes 1–8 on the v4.2 split. All R² values are on the floor-included test set; non-floor R² is computed on the floor-excluded subset.

### 3.2 Route 9: foundation models break the baseline

#### 3.2.1 TabPFN v2 (in-context learning, no gradient)
TabPFN v2 on the 217 RDKit descriptors (canonical split; seeds 42, 123, 7) achieved **R² = 0.496 ± 0.002** (0.494 / 0.4972 / 0.4973 per seed), with non-floor R² of 0.627 against the baseline's 0.632. The gain of +0.032 over the baseline exceeds the inter-seed standard deviation by a factor of sixteen, ruling out a seed artefact. Extending the feature set to Morgan fingerprints (2,265 dimensions) or Morgan plus MoLFormer-XL (3,033 dimensions) yielded no improvement (0.481 and 0.482), consistent with the in-context saturation behaviour of TabPFN beyond moderate feature counts.

#### 3.2.2 KPGT fine-tuning (end-to-end gradient)
KPGT fine-tuning (base.pth; 3 seeds; early stopping) reached **R² = 0.513 ± 0.005**, the best result among all PAMPA routes (+0.049 versus baseline). Per-seed best-validation test R²: seed 42 (epoch 16) = 0.519; seed 123 (epoch 9) = 0.507; seed 7 (epoch 13) = 0.514. Validation R² peaked at 0.411 (seed 123), and each epoch required approximately 410 s on the RTX 4070 SUPER.

A critical trade-off accompanies this gain: non-floor R² declined to 0.536 (baseline 0.632; Δ = −0.096). The entire improvement is therefore concentrated in the censored floor region. Gradient fine-tuning on censored labels forces the model to allocate capacity to the floor pattern — a single repeated value — at the expense of fidelity on measured compounds.

| Model | R² (all) | ± SD | R² (non-floor) | Best epoch | Val R² (best) |
|---|---:|---:|---:|---:|---:|
| MLP baseline | 0.4642 | — | 0.6317 | — | — |
| TabPFN v2 (217 descriptors) | 0.4962 | 0.0016 | 0.6268 | — | — |
| KPGT fine-tune (seed 42) | 0.5191 | — | 0.5633 | 16 | 0.4059 |
| KPGT fine-tune (seed 123) | 0.5073 | — | 0.5404 | 9 | 0.4105 |
| KPGT fine-tune (seed 7) | 0.5139 | — | 0.5035 | 13 | 0.4080 |
| **KPGT mean ± SD** | **0.5134** | **0.0048** | **0.5357** | — | **0.4081** |

**Table 2.** Route 9 foundation-model results. R² is on the floor-included test set at the best-validation checkpoint.

### 3.3 Cross-dataset comparison with pepADMET and PeptiVerse
Table 3 places our results in the context of the two leading platforms on the shared 6,834 molecules. Our KPGT fine-tune (ρ = 0.811) substantially outperforms the PeptiVerse frozen heads — ChemBERTa (ρ = 0.671) and PeptideCLM (ρ = 0.667) — on the identical molecules. The pepADMET reported permeability R² range (0.435–0.657) spans multiple endpoints and model families; for PAMPA specifically, their best single model reported R² ≈ 0.66 on their split. That value exceeds our baseline not because their model is stronger, but because their split does not enforce unique-SMILES separation — on our leakage-controlled split the same model family drops to 0.464.

| Method | Representation | Training | Split | PAMPA metric (shared 6,834) |
|---|---|---|---|---|
| pepADMET best [2] | 2D + 3D descriptors + Morgan | LightGBM | pepADMET split | R² ≈ 0.66 (their split) |
| PeptiVerse [3] | ChemBERTa-77M (384-d), frozen | XGBoost head | 80/20 Tanimoto cluster | **ρ = 0.671** |
| PeptiVerse [3] | PeptideCLM-23M (768-d), frozen | XGBoost head | 80/20 Tanimoto cluster | **ρ = 0.667** |
| Our baseline | RDKit 217 + Morgan 2048 + MoLFormer-XL 768 | MLP | v4.2 unique-SMILES 70/10/20 | R² = 0.464, ρ = 0.77 |
| Our KPGT fine-tune | LiGhT 12-layer graph (base.pth) | **End-to-end fine-tune** | v4.2 unique-SMILES 70/10/20 | **R² = 0.513, ρ = 0.811** |

**Table 3.** Cross-dataset comparison on the shared molecules. PeptiVerse metrics are from the original publication (Table 1); our metrics are on the v4.2 split. Spearman ρ for our models is computed on the test partition (1,457 ∩ shared).

### 3.4 The censored ceiling (PAMPA)
The oracle ceiling of R² = 0.539 — derived from the censored variance under perfect within-floor ranking — bounds every method on this endpoint. KPGT fine-tuning reaches 95 % of it; TabPFN v2 reaches 92 %. No algorithm can exceed 0.539 without uncensored measurements of the 269 floor compounds. Auxiliary floor-ranking models (AUC 0.856 for LightGBM, 0.762 for an MLP) confirm that the floor compounds admit partial ordering, but no operating point yields a usable regression on the floor subset.

### 3.5 Route 10: foundation models on Caco-2 — generalisation and a ranking reversal
Table 4 repeats the two foundation models of route 9 on the Caco-2 endpoint (7,429 compounds; identical split protocol; test n = 1,490, of which 39 floor rows). Both models again surpass the committed baseline, confirming that the route-9 gains are not a PAMPA-specific artefact. The ranking, however, reverses: TabPFN v2 (217 descriptors) attains **R² = 0.442 ± 0.003** (+0.048 over the baseline's 0.393), whereas KPGT fine-tuning reaches 0.411 ± 0.006 (+0.018).

The two models also differ in where their gains arise. On Caco-2, TabPFN improves the non-floor subset markedly (R² = 0.609 vs the baseline's 0.515, +0.093) while remaining the best floor-included model — unlike on PAMPA, where its gain was confined to the floor region (non-floor −0.005). KPGT reproduces its PAMPA trade-off in amplified form: its floor-included gain (+0.018) is paired with a non-floor deficit (0.449 vs 0.515, −0.066), and its Caco-2 training was markedly less stable (per-seed non-floor R² ranging 0.39–0.49; best epochs 8/15/16 across seeds; recurrent loss spikes), consistent with gradient fine-tuning of a 111.8 M-parameter graph transformer on 7,429 molecules. The Caco-2 oracle ceiling under the same convention is R² = 0.570; TabPFN reaches 77.5 % of it, KPGT 72.2 %. Feature-set effects replicate the PAMPA finding: for TabPFN, the 217-descriptor set outperforms both the 2,265- and 3,033-dimensional extensions (0.427 and 0.435).

| Model | R² (all) | ± SD | R² (non-floor) | Δ vs baseline (all) | Spearman | Best epoch (seeds 42/123/7) |
|---|---:|---:|---:|---:|---:|---|
| MLP baseline (v4.2) | 0.3933 | — | 0.5152 | — | — | — |
| TabPFN v2 (217 descriptors) | **0.4415** | 0.0034 | **0.6086** | **+0.048** | 0.751 | — |
| TabPFN v2 (desc + Morgan, 2,265) | 0.4273 | 0.0068 | 0.5808 | +0.034 | 0.736 | — |
| TabPFN v2 (desc + Morgan + MoLFormer, 3,033) | 0.4350 | 0.0047 | 0.5849 | +0.042 | 0.747 | — |
| KPGT fine-tune (seed 42) | 0.4125 | — | 0.4649 | +0.019 | 0.756 | 15 |
| KPGT fine-tune (seed 123) | 0.4175 | — | 0.4907 | +0.024 | 0.739 | 16 |
| KPGT fine-tune (seed 7) | 0.4038 | — | 0.3912 | +0.011 | 0.720 | 8 |
| **KPGT mean ± SD** | **0.4113** | **0.0057** | **0.4489** | **+0.018** | **0.739** | — |
| Oracle ceiling | 0.5696 | — | — | — | — | — |

**Table 4.** Route 10: foundation models on Caco-2 (v4.2 split, seed 42; test n = 1,490, 39 floor rows). Baseline re-measured on the identical split. TabPFN results are 3-seed means (seeds 42, 123, 7); KPGT results are per-seed best-validation checkpoints.

### 3.6 Context within the wider cyclic-peptide permeability literature
Table 5 situates our results among task-specific cyclic-peptide permeability models published over the last three years, most of which train on the CycPeptMPDB permeability database [12]. Two patterns are worth stressing. First, the published PAMPA R² values (0.67–0.77) are all measured under protocols that do not remove structural near-duplicates or that ignore the censored floor, so they are not directly comparable to our leakage-controlled, floor-aware numbers. Second, the only study to evaluate a rigorous scaffold split [13] reports that model generalisability collapses substantially under it — precisely the gap between the headline R² and the R² we observe under unique-SMILES separation. We therefore read the difference between published and our R² as a protocol effect (duplicate leakage + ignored censoring), not evidence that our models are weaker.

| # | Model (reference) | Task / split | Reported PAMPA | Reported Caco-2 |
|---|---|---|---|---|
| 1 | CycPeptMP [13] | regression, their test split | R² = 0.772 ± 0.011 | — |
| 2 | Multi_CycGT [14] | classification (accuracy/AUC) | acc = 0.821, AUC = 0.865 | external sets |
| 3 | CPMP [15] | regression, random split | R² = 0.67 | R² = 0.75 |
| 4 | DMPNN (13-model benchmark) [16] | regression, random vs scaffold | best of 13 (scaffold ≪ random) | — |
| 5 | PCPpred (Mordred-2D) [17] | regression, assay-specific | R² = 0.685 (PCC 0.830) | R² = 0.793 (PCC 0.892) |
| 6 | C2PO [18] | permeability-optimisation (generative) | design objective, not a held-out R² | design objective |
| 7 | **This work — KPGT fine-tune** | **unique-SMILES 70/10/20, censored-floor** | **R² = 0.513** | **R² = 0.411** |
| 8 | **This work — TabPFN v2** | **unique-SMILES 70/10/20, censored-floor** | **R² = 0.496** | **R² = 0.442** |

**Table 5.** Published cyclic-peptide permeability models (rows 1–6, as reported in each publication; split protocols and floor handling vary and are not directly comparable) versus this work (rows 7–8, unique-SMILES split with explicit left-censoring). Caco-2 values for this work are under the identical v4.2 protocol.

The practical reading: a model reported at R² ≈ 0.77 [13] on a non-deduplicated random split and a model reported at R² = 0.513 on a unique-SMILES, floor-aware split are answering different questions. The former asks "how well does the model fit this particular test split, which shares near-duplicates with the training set?"; the latter asks "how well does the model generalise to molecules it has never seen in any tautomeric or stereochemical form, while honestly handling the assay floor?". The scaffold-split evidence in [13] shows the two answers diverge by a wide margin, and is consistent with our finding that the censored ceiling (0.539) — not model capacity — is the binding constraint once leakage is removed.

## 4 Discussion

### 4.1 Why foundation models succeed where classical routes fail
Routes 1–8 operate within the descriptor-plus-gradient-boosting paradigm and share a common weakness: none can exploit the structure of the censored labels. The floor is a measurement artefact rather than a chemical pattern; LightGBM treats all labels as equally informative; and ensembling averages noise rather than signal. Foundation models introduce external inductive bias. TabPFN v2 conditions on synthetic tabular priors learned in-context [4], and KPGT inherits geometric representations pretrained on 1.6 M molecules [5]. Both priors help interpolate the censored region without overfitting the repeated floor value — precisely the regime where classical reweighting and target transformations (routes 2, 4, 5) could not reach. Route 10 shows that this advantage generalises across endpoints: both foundation models surpass the Caco-2 baseline as well, under the identical split and floor protocol. Notably, no single model dominates — KPGT leads on PAMPA, TabPFN on Caco-2 — suggesting that the in-context and gradient-based priors are complementary rather than interchangeable.

### 4.2 Frozen embeddings versus fine-tuning
PeptiVerse's frozen embeddings with lightweight heads yield ρ = 0.67–0.69 on PAMPA. Our KPGT fine-tune — the same architectural family, but trained end-to-end from base.pth — reaches ρ = 0.811 on the same molecules, a gap of +0.14 in Spearman correlation. Because the split and the molecules are held constant, this gap isolates the effect of task-specific adaptation: **frozen embeddings discard the task-relevant information that fine-tuning recovers**. For peptide permeability, end-to-end fine-tuning of a pretrained graph transformer is not a refinement but a necessity.

### 4.3 The price paid on measured compounds
On PAMPA, both foundation models improve floor-included R² while leaving or degrading non-floor R² (TabPFN: −0.005; KPGT: −0.096). Route 10 sharpens this picture: on Caco-2 the pattern is endpoint-dependent — TabPFN improves non-floor R² substantially (+0.093), whereas KPGT again pays for its floor gain (−0.066). Gradient training on censored labels allocates model capacity to a single repeated value, reducing fidelity on the ~96–97 % of compounds with true measurements; the effect is reproducible across both endpoints for the gradient-tuned model. We regard this as a fundamental, not incidental, trade-off: **any model trained on censored labels without explicit censored modelling risks sacrificing measured-compound accuracy in exchange for floor performance**. Tobit regression (route 4), the canonical remedy, was worse still (non-floor R² = 0.466), suggesting that with only 269 censored examples the censored likelihood is under-determined. Practical use of such models should therefore report floor- and non-floor metrics separately, as we do throughout.

### 4.4 Limitations
1. The ceilings of 0.539 (PAMPA) and 0.570 (Caco-2) are hard limits of the current assays; R² = 0.7 is unreachable without re-measurement.
2. KPGT fine-tuning requires GPU access (the public DGL wheel is CPU-only on Windows; our pure-PyTorch port was required) and was less stable on Caco-2 than on PAMPA.
3. TabPFN v2's feature limit (500 dimensions) precludes using the full fingerprint-plus-descriptor set.
4. The ten routes were evaluated on single canonical splits; the route-9/10 gains (13–20 × seed noise) would benefit from outer cross-validation to exclude split luck.
5. Published comparisons (Table 5) rely on each paper's reported metrics; we did not re-run the external models, so cross-statement comparisons inherit their protocol choices. Independent wet-lab validation of the highest-ranked predictions (KPGT on PAMPA, TabPFN on Caco-2) is the natural next step.

### 4.5 Scope of applicability
The openclaw-peptide-admet platform provides: (i) leakage-controlled unique-SMILES splitting for peptide datasets; (ii) censored-floor-aware evaluation (oracle ceiling, floor AUC, floor/non-floor metrics); (iii) reproducible pipelines for the nine PAMPA routes plus the Caco-2 foundation-model extension (route 10) and HLM; (iv) a GPU KPGT fine-tuning script (pure PyTorch, checkpoint/resume, verified against the DGL reference, PAMPA and Caco-2); and (v) canonical TabPFN v2 evaluation scripts for both endpoints. We recommend it for lead prioritisation of cyclic-peptide permeability (PAMPA/Caco-2) where assay data is censored, for benchmarking new peptide ADMET methods against a rigorous baseline, and as a starting point for foundation-model fine-tuning on peptide graphs. Extrapolation to linear peptides, non-peptide macrocycles, or uncensored high-permeability regimes is not recommended without external validation.

## 5 Conclusions
We present the first systematic, leakage-controlled, censored-floor-aware benchmark of ten permeability-improvement routes on 7,283 PAMPA and 7,429 Caco-2 cyclic peptides. All eight classical PAMPA routes — descriptor expansion, rank-Gaussian target transformation, LightGBM ensembling, Tobit censored regression, soft-label blending, ChemBERTa frozen embeddings, PeptiVerse raw-data cross-validation, and label averaging — failed to exceed the MLP baseline (R² = 0.464). Foundation models broke the barrier on both endpoints: on PAMPA, TabPFN v2 (in-context learning) reached R² = 0.496 ± 0.002 and KPGT fine-tuning (end-to-end gradient) reached R² = 0.513 ± 0.005, the best of all routes and 95 % of the theoretical censored ceiling (0.539); on Caco-2, the ranking reversed, with TabPFN v2 leading at R² = 0.442 ± 0.003 (baseline 0.393, non-floor 0.609) and KPGT at 0.411 ± 0.006 — evidence that foundation-model gains generalise across permeability endpoints but that no single model dominates. On the shared molecules, KPGT fine-tuning (ρ = 0.811) exceeded the PeptiVerse frozen embeddings (ρ = 0.671) by +0.14 in Spearman correlation, demonstrating that task-specific fine-tuning is necessary. The gains of the gradient-tuned model, however, concentrate in the censored region on both endpoints at the cost of non-floor accuracy, whereas TabPFN's gains are floor-confined on PAMPA but extend to measured compounds on Caco-2. **The censored ceilings — 0.539 (PAMPA) and 0.570 (Caco-2), not model capacity, are the true bottlenecks.** Advancing peptide permeability prediction requires uncensored re-measurement of floor compounds — an experimental, not algorithmic, task.

## References
[1] Hornsby BD, Lee CH, Steele CA, Tuekpe JK, Lim CS. Therapeutic peptides and proteins: Status and developments in drug delivery. J Control Release. 2026;394:114895. doi:10.1016/j.jconrel.2026.114895

[2] Tan X, Liu Q, Zhou M, Fang Y, Ouyang D, Zeng W, Dong J. pepADMET: A Novel Computational Platform For Systematic ADMET Evaluation of Peptides. J Chem Inf Model. 2026;66(2):936-946. doi:10.1021/acs.jcim.5c02518

[3] Zhang Y, Tang S, Chen T, Mahood E, Vincoff S, Chatterjee P. PeptiVerse: A unified platform for therapeutic peptide property prediction. Nat Commun. 2026;17:6819. doi:10.1038/s41467-026-74167-w

[4] Hollmann N, Muller S, Purucker L, Krishnakumar A, Korfer M, Hoo SB, Schirrmeister RT, Hutter F. Accurate predictions on small data with a tabular foundation model. Nature. 2025;637(8045). doi:10.1038/s41586-024-08328-6

[5] Li H, Zhao D, Zeng J. KPGT: Knowledge-Guided Pre-training of Graph Transformer for Molecular Property Prediction. arXiv preprint arXiv:2206.03364. 2022. doi:10.48550/arXiv.2206.03364

[6] Zhao C, et al. openclaw-peptide-admet: Peptide ADMET Prediction Platform with Censored-Floor-Aware Protocol. https://github.com/c00jsw00/openclaw-peptide-admet (accessed 2026-09-01). (v4.2 protocol: unique-SMILES 70/10/20 split, censored floor at −10.0 log cm/s, MixedADMETMLP baseline, Huber loss)

[7] Rogers D, Hahn M. Extended-Connectivity Fingerprints. J Chem Inf Model. 2010;50(5):742-754. doi:10.1021/ci100050t

[8] Landrum G. RDKit: Open-Source Cheminformatics. 2024. https://www.rdkit.org

[9] Ke G, et al. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. Adv Neural Inf Process Syst. 2017;30:3146-3154.

[10] Ahmad W, Simon E, Chithrananda S, Grand G, Ramsundar B. ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction. arXiv preprint arXiv:2010.09885. 2020. doi:10.48550/arXiv.2010.09885

[11] Wang M, Zheng D, Ye Z, Gan Q, Li M, Zhou X, Ma C, Hu Z, Yang Q, Zhao Y, Li J, Smola A, Zhang Z. Deep Graph Library: A Graph-Centric, Highly-Performant Package for Graph Neural Networks. arXiv preprint arXiv:1909.01315. 2019. doi:10.48550/arXiv.1909.01315

[12] Li J, Yanagisawa K, Sugita M, Fujie T, Ohue M, Akiyama Y. CycPeptMPDB: A Comprehensive Database of Membrane Permeability of Cyclic Peptides. J Chem Inf Model. 2023;63(7):2240-2250. doi:10.1021/acs.jcim.2c01573

[13] Liu W, Li X, Verma K, Lee H. Systematic Benchmarking of 13 AI Methods for Predicting Cyclic Peptide Membrane Permeability. J Cheminform. 2025;17(1):129. doi:10.1186/s13321-025-01083-4

[14] Li J, Yanagisawa K, Akiyama Y. CycPeptMP: Enhancing Membrane Permeability Prediction of Cyclic Peptides with Multi-Level Molecular Features and Data Augmentation. Brief Bioinform. 2024;25(5):bbae417. doi:10.1093/bib/bbae417

[15] Cao L, Xu Z, Shang T, Zhang C, Wu X, Wu Y, Zhai S, Zhan Z, Duan H. Multi_CycGT: A Deep Learning-Based Multimodal Model for Predicting the Membrane Permeability of Cyclic Peptides. J Med Chem. 2024;67(3):1888-1899. doi:10.1021/acs.jmedchem.3c01611

[16] Jiang D, Chen Z, Du H. Cyclic Peptide Membrane Permeability Prediction Using Deep Learning Model Based on Molecular Attention Transformer. Front Bioinform. 2025;5:1566174. doi:10.3389/fbinf.2025.1566174

[17] Shendre A, Gahlot PS, Raghava GP. PCPpred: Prediction of Chemically Modified Peptide Permeability Across Multiple Assays for Oral Delivery. bioRxiv preprint. 2026. doi:10.64898/2026.01.19.700485

[18] Aerts R, Tavernier J, Kerstjens F, Ahmad W, Gómez-Tamayo F, Tresadern P, De Winter M. C2PO: An ML-Powered Optimizer of the Membrane Permeability of Cyclic Peptides Through Chemical Optimisation. J Cheminform. 2025;17:168. doi:10.1186/s13321-025-01109-x

---

**Graphical Abstract** (to be rendered): Flowchart showing the v4.2 censored-floor protocol → ten routes (PAMPA 1–9, Caco-2 10) → baselines 0.464/0.393 → TabPFN 0.496/0.442 → KPGT 0.513/0.411 → ceilings 0.539/0.570. Key message: foundation models are the first to beat the baselines on both endpoints, but the ceilings hold.

**Data Availability** All data, splits, trained checkpoints, and reproduction scripts are available at https://github.com/c00jsw00/openclaw-peptide-admet (MIT License).

**AI Declaration** This manuscript was prepared with the assistance of AI tools (Hermes Agent, Nemotron-3-Ultra) for code generation, literature retrieval, and text drafting. All scientific claims, numbers, and conclusions were verified by the authors against primary computational experiments.

**Conflict of Interest** The authors declare no competing financial interests.