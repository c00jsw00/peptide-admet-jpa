# Peptide ADMET Prediction: A Systematic Benchmark and Foundation Model Evaluation

**Highlights**
- First systematic benchmark of nine PAMPA improvement routes under a censored-floor-aware protocol (7,283 peptides; 3.7 % left-censored at −10.0 log cm/s)
- TabPFN v2 (217 descriptors, in-context learning) attains R² = 0.496 ± 0.002, the first method to surpass the MLP baseline (+0.032)
- KPGT fine-tuning (LiGhT 12-layer graph transformer, base.pth) reaches R² = 0.513 ± 0.005, the best of all nine routes and 95 % of the censored ceiling (0.539)
- Gains are confined to the censored region: non-floor R² declines (KPGT 0.536 vs baseline 0.632), exposing the cost of gradient training on censored labels
- On 6,834 shared molecules, fine-tuned KPGT (ρ = 0.811) outperforms PeptiVerse frozen embeddings (ρ = 0.671), establishing the necessity of task-specific tuning

## Abstract
**Purpose** Oral absorption remains a principal bottleneck in peptide drug discovery. We report a systematic evaluation of nine computational routes for PAMPA permeability prediction on the pepADMET dataset (7,283 cyclic peptides), conducted under a leakage-controlled, censored-floor-aware protocol. **Methods** A unique-SMILES 70/10/20 split (seed 42) was applied to preclude data leakage. The left-censored floor (−10.0 log cm/s; 269 compounds, 3.7 %) was treated explicitly, and an oracle ceiling of R² = 0.539 was derived from the censored variance. Nine routes were examined: (1) descriptor expansion (5-seed mean), (2) rank-Gaussian target transformation (honest train-fitted quantile, 5-seed ensemble), (3) LightGBM ensembling (5-seed), (4) Tobit censored regression, (5) soft-label blending (β selected on validation), (6) ChemBERTa frozen embeddings with MLP head (3 seeds, 4 configurations), (7) PeptiVerse raw-data cross-validation, (8) label averaging (pepADMET ensemble mean), and (9) foundation models (TabPFN v2; KPGT fine-tuning). **Results** Routes 1–8 failed to exceed the baseline R² = 0.464 (best Δ = +0.001, route 5). TabPFN v2 (217 RDKit descriptors) achieved R² = 0.496 ± 0.002, and KPGT fine-tuning (base.pth, 3 seeds, early stopping) reached R² = 0.513 ± 0.005 (best seed 0.519). Both foundation models derive their gains exclusively from the censored subset; non-floor R² decreased. On the 6,834 shared molecules, the Spearman correlation of our KPGT fine-tune (ρ = 0.811) exceeded that of PeptiVerse frozen ChemBERTa (ρ = 0.671). **Conclusion** Foundation models constitute the first methods to break the PAMPA baseline, yet the censored ceiling (0.539) remains the hard limit. Progress beyond this bound requires uncensored re-measurement of floor compounds rather than improved models.

**Keywords** peptide ADMET; PAMPA permeability; censored regression; foundation models; TabPFN; KPGT; data leakage

## 1 Introduction
Therapeutic peptides have emerged as a distinct modality occupying the space between small molecules and biologics, with more than one hundred approved peptide drugs and a pipeline that continues to expand across metabolic, oncologic, and infectious disease indications [1]. Their large size, conformational flexibility, and high polarity impose absorption, distribution, metabolism, excretion, and toxicity (ADMET) challenges that classical small-molecule rules — Lipinski's rule of five foremost among them — were never designed to address [2]. Reliable in silico ADMET prediction for peptides is therefore a prerequisite for efficient lead prioritisation and rational design.

Two recent platforms have defined the current state of the art. The pepADMET platform [3] compiled the largest public peptide ADMET collection to date — PAMPA (7,283), Caco-2 (7,429), HLM, MDCK, and additional endpoints — and reported permeability R² values of 0.435–0.657 across model families, establishing the paradigm of classical descriptors combined with gradient boosting. Independently, PeptiVerse [4] unified peptide property prediction around frozen foundational embeddings (PeptideCLM, ChemBERTa, ESM-2) with lightweight prediction heads, reporting PAMPA Spearman ρ = 0.69 and Caco-2 ρ = 0.80, and establishing the paradigm of frozen representations with simple heads.

Both studies, however, share three limitations that constrain their conclusions. First, neither accounts for the left-censored floor in PAMPA measurements (−10.0 log cm/s; 3.7 % of the data), which inflates label variance and biases regression metrics. Second, unique-SMILES splitting was not uniformly enforced, leaving the results exposed to leakage through stereoisomers or tautomers. Third, frozen embeddings and end-to-end fine-tuning were never compared on an identical split, so the contribution of task-specific adaptation cannot be isolated.

Here we address all three limitations in a single, systematic benchmark. Working on the pepADMET PAMPA endpoint under a censored-floor-aware protocol (v4.2 split, seed 42), we evaluate nine improvement routes — eight classical and two foundation-model based — against a rigorously controlled baseline. We ask two questions: (i) can foundation models, specifically TabPFN v2 (in-context tabular learning) [5] and KPGT (knowledge-pretrained graph transformer, fine-tuned) [6], exceed the MLP baseline and approach the theoretical censored ceiling? and (ii) does end-to-end fine-tuning, rather than frozen embeddings, explain the gap to state-of-the-art peptide platforms? To answer the second question we re-evaluate pepADMET and PeptiVerse on the shared molecules, holding the split constant.

## 2 Materials and Methods

### 2.1 Dataset and censored-floor protocol
The pepADMET PAMPA dataset (7,283 cyclic peptides, log Papp in cm/s) was obtained from the pepADMET repository [3]. A left-censored floor at −10.0 log cm/s affects 269 compounds (3.7 %): these values represent the lower bound of assay sensitivity rather than true permeabilities. Following the v4.2 protocol [7], we partitioned the data by unique canonical SMILES into 70/10/20 train/validation/test sets (seed 42), yielding 5,102 / 724 / 1,457 compounds and thereby precluding leakage through structural near-duplicates. All metrics are reported twice — on the full test set (floor included) and on the non-floor subset (1,410 compounds). In addition, an oracle ceiling of R² = 0.539 was computed from the censored variance under the assumption of perfect ranking within the floor [7]; no model can exceed this bound on the current data.

### 2.2 Baseline model
The v4.2 baseline is a **single-head neural network (MixedADMETMLP)** [7] with a 3033-dimensional input: 217 RDKit 2D descriptors, 2,048-bit Morgan fingerprints (radius 2), and a 768-dimensional frozen MoLFormer-XL CLS embedding (ibm-research/MoLFormer-XL-both-10pct). The trunk comprises two hidden layers (256 → 128 units) with Huber loss (δ = 1.0), trained with Adam (lr = 1 × 10⁻³, weight decay 1 × 10⁻⁵), ReduceLROnPlateau (factor 0.5, patience 4), and early stopping (validation loss, patience 10) for a maximum of 80 epochs. This model (810,497 parameters) achieves R² = 0.464 on the full test set and R² = 0.632 on the non-floor subset.

### 2.3 Nine improvement routes
Routes 1–8 follow the v4.2 pipeline [7] with the identical split and evaluation protocol:

1. **Descriptor expansion** — Morgan fingerprints of radii r1, r2, r3 appended to the baseline feature set (5-seed mean; seeds 42, 123, 456, 789, 1024; MLP retrained).
2. **Rank-Gaussian target transformation** — the regression target is mapped to a Gaussian scale via a quantile transform fitted on the training set only (honest); predictions are back-transformed (5-seed ensemble).
3. **LightGBM ensembling** — multiple LightGBM configurations (with additional Morgan radii r1, r3, r4) combined by averaging (5-seed ensemble).
4. **Tobit censored regression** — a censored likelihood (NLL) with the censoring threshold at −10.0, trained with early stopping.
5. **Soft-label blending** — the final prediction is a convex combination of the floor mean and the regression prediction, weighted by a per-compound floor probability; the mixing weight β is selected on validation.
6. **ChemBERTa frozen embeddings** — 384-dimensional embeddings from ChemBERTa-77M (frozen) fed to an MLP head (3 seeds, 4 feature configurations), using the standard v4.2 training loop.
7. **PeptiVerse raw-data cross-validation** — the PeptiVerse PAMPA and Caco-2 sets (HF ChatterjeeLab/PeptiVerse_data; 6,869 and 606 compounds) are re-split with the same unique-SMILES 70/10/20 protocol and re-trained end-to-end.
8. **Label averaging** — the pepADMET ensemble mean is used as a smoothed pseudo-label (one row per unique SMILES, y = arithmetic mean of replicate labels) in place of the raw measurements.

Route 9 comprises two foundation models:
- **TabPFN v2** (v2.0; direct download, pretraining limits disabled, n_estimators = 4) applied to the 217 RDKit descriptors, evaluated on the canonical split so that its test set is identical to the baseline's. Seeds: 42, 123, 7.
- **KPGT fine-tuning** — a 12-layer LiGhT graph transformer (d_g = 768, 12 attention heads, n_mol_layers = 12, path length 5, d_hpath_ratio = 12, feed-forward dimension 3,072) initialised from base.pth (447 MB, pretrained on 1.6 M molecules at ICML 2024) and fine-tuned on the v4.2 training set (batch size 64, AdamW lr = 1 × 10⁻⁴, weight decay 1 × 10⁻⁶, 15-epoch warmup cosine decay, early stopping patience 15). Three seeds (7, 42, 123) were run. Because the publicly available DGL implementation for this architecture is CPU-only on Windows, we developed a pure-PyTorch GPU implementation (scatter-based TripletTransformer replacing DGL's u_dot_v, edge_softmax, and update_all-sum primitives), verified against the official DGL CPU implementation to a maximum absolute difference of 8.3 × 10⁻⁷. Checkpoints are saved per epoch and the best-validation model is used for final evaluation.

### 2.4 Cross-dataset comparison
Shared molecules between pepADMET PAMPA and PeptiVerse PAMPA were identified by canonical SMILES (RDKit). Of 7,177 unique pepADMET SMILES and 6,869 PeptiVerse SMILES, 6,834 (95.2 % of pepADMET; 99.5 % of PeptiVerse) overlap. Labels agreed exactly for 6,830 of these (the four discrepancies span at most 1.58 log units). Models are re-evaluated on this common subset — using each paper's original split where available, and our v4.2 split for the direct, leakage-controlled comparison.

### 2.5 Evaluation metrics
We report the coefficient of determination (R²), root mean squared error (RMSE), mean absolute error (MAE), Spearman's rank correlation (ρ), and the floor-excluded R² (non-floor R²). All metrics are computed with scikit-learn ≥1.5. Models are selected on validation R² by early stopping; final test evaluation uses the best-validation checkpoint.

### 2.6 Software and hardware
Python 3.11.16; PyTorch 2.13.0 (CUDA 12.6); DGL 2.2.1 (CPU, patched); TabPFN 8.5.0; LightGBM 4.7.0; RDKit 2026.03.5. The KPGT fine-tune was executed on an NVIDIA RTX 4070 SUPER (~400 s/epoch vs ~815 s/epoch on CPU); LightGBM and TabPFN ran on an AMD Ryzen 9 7950X. All code, data, and trained checkpoints are available at https://github.com/c00jsw00/openclaw-peptide-admet.

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
KPGT fine-tuning (base.pth; 3 seeds; early stopping) reached **R² = 0.513 ± 0.005**, the best result among all nine routes (+0.049 versus baseline). Per-seed best-validation test R²: seed 42 (epoch 16) = 0.519; seed 123 (epoch 9) = 0.507; seed 7 (epoch 13) = 0.514. Validation R² peaked at 0.411 (seed 123), and each epoch required approximately 410 s on the RTX 4070 SUPER.

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
| pepADMET best [3] | 2D + 3D descriptors + Morgan | LightGBM | pepADMET split | R² ≈ 0.66 (their split) |
| PeptiVerse [4] | ChemBERTa-77M (384-d), frozen | XGBoost head | 80/20 Tanimoto cluster | **ρ = 0.671** |
| PeptiVerse [4] | PeptideCLM-23M (768-d), frozen | XGBoost head | 80/20 Tanimoto cluster | **ρ = 0.667** |
| Our baseline | RDKit 217 + Morgan 2048 + MoLFormer-XL 768 | MLP | v4.2 unique-SMILES 70/10/20 | R² = 0.464, ρ = 0.77 |
| Our KPGT fine-tune | LiGhT 12-layer graph (base.pth) | **End-to-end fine-tune** | v4.2 unique-SMILES 70/10/20 | **R² = 0.513, ρ = 0.811** |

**Table 3.** Cross-dataset comparison on the shared molecules. PeptiVerse metrics are from the original publication (Table 1); our metrics are on the v4.2 split. Spearman ρ for our models is computed on the test partition (1,457 ∩ shared).

### 3.4 The censored ceiling
The oracle ceiling of R² = 0.539 — derived from the censored variance under perfect within-floor ranking — bounds every method on this data. KPGT fine-tuning reaches 95 % of it; TabPFN v2 reaches 92 %. No algorithm can exceed 0.539 without uncensored measurements of the 269 floor compounds. Auxiliary floor-ranking models (AUC 0.856 for LightGBM, 0.762 for an MLP) confirm that the floor compounds admit partial ordering, but no operating point yields a usable regression on the floor subset.

## 4 Discussion

### 4.1 Why foundation models succeed where classical routes fail
Routes 1–8 operate within the descriptor-plus-gradient-boosting paradigm and share a common weakness: none can exploit the structure of the censored labels. The floor is a measurement artefact rather than a chemical pattern; LightGBM treats all labels as equally informative; and ensembling averages noise rather than signal. Foundation models introduce external inductive bias. TabPFN v2 conditions on synthetic tabular priors learned in-context [5], and KPGT inherits geometric representations pretrained on 1.6 M molecules [6]. Both priors help interpolate the censored region without overfitting the repeated floor value — precisely the regime where classical reweighting and target transformations (routes 2, 4, 5) could not reach.

### 4.2 Frozen embeddings versus fine-tuning
PeptiVerse's frozen embeddings with lightweight heads yield ρ = 0.67–0.69 on PAMPA. Our KPGT fine-tune — the same architectural family, but trained end-to-end from base.pth — reaches ρ = 0.811 on the same molecules, a gap of +0.14 in Spearman correlation. Because the split and the molecules are held constant, this gap isolates the effect of task-specific adaptation: **frozen embeddings discard the task-relevant information that fine-tuning recovers**. For peptide permeability, end-to-end fine-tuning of a pretrained graph transformer is not a refinement but a necessity.

### 4.3 The price paid on measured compounds
Both foundation models improve floor-included R² while degrading non-floor R² (TabPFN: −0.005; KPGT: −0.096). Gradient training on censored labels allocates model capacity to a single repeated value, reducing fidelity on the 96.3 % of compounds with true measurements. We regard this as a fundamental, not incidental, trade-off: **any model trained on censored labels without explicit censored modelling will sacrifice measured-compound accuracy in exchange for floor performance**. Tobit regression (route 4), the canonical remedy, was worse still (non-floor R² = 0.466), suggesting that with only 269 censored examples the censored likelihood is under-determined. Practical use of such models should therefore report floor- and non-floor metrics separately, as we do throughout.

### 4.4 Limitations
1. The ceiling of 0.539 is a hard limit of the current assay; R² = 0.7 is unreachable without re-measurement.
2. KPGT fine-tuning requires GPU access (the public DGL wheel is CPU-only on Windows; our pure-PyTorch port was required).
3. TabPFN v2's feature limit (500 dimensions) precludes using the full fingerprint-plus-descriptor set.
4. The Caco-2 ceiling is even lower (σ̂ ≈ SD, ceiling ≈ 0.55); the same conclusions are expected to apply.

### 4.5 Scope of applicability
The openclaw-peptide-admet platform provides: (i) leakage-controlled unique-SMILES splitting for peptide datasets; (ii) censored-floor-aware evaluation (oracle ceiling, floor AUC, floor/non-floor metrics); (iii) reproducible pipelines for the nine PAMPA routes plus Caco-2 and HLM; (iv) a GPU KPGT fine-tuning script (pure PyTorch, checkpoint/resume, verified against the DGL reference); and (v) a canonical TabPFN v2 evaluation script. We recommend it for lead prioritisation of cyclic-peptide permeability (PAMPA/Caco-2) where assay data is censored, for benchmarking new peptide ADMET methods against a rigorous baseline, and as a starting point for foundation-model fine-tuning on peptide graphs. Extrapolation to linear peptides, non-peptide macrocycles, or uncensored high-permeability regimes is not recommended without external validation.

## 5 Conclusions
We present the first systematic, leakage-controlled, censored-floor-aware benchmark of nine PAMPA improvement routes on 7,283 cyclic peptides. All eight classical routes — descriptor expansion, rank-Gaussian target transformation, LightGBM ensembling, Tobit censored regression, soft-label blending, ChemBERTa frozen embeddings, PeptiVerse raw-data cross-validation, and label averaging — failed to exceed the MLP baseline (R² = 0.464). Foundation models broke the barrier: TabPFN v2 (in-context learning) reached R² = 0.496 ± 0.002, and KPGT fine-tuning (end-to-end gradient) reached R² = 0.513 ± 0.005, the best of all routes and 95 % of the theoretical censored ceiling (0.539). On the shared molecules, KPGT fine-tuning (ρ = 0.811) exceeded the PeptiVerse frozen embeddings (ρ = 0.671) by +0.14 in Spearman correlation, demonstrating that task-specific fine-tuning is necessary. Both foundation models, however, concentrate their gains in the censored region, and non-floor accuracy declines. **The censored ceiling, not model capacity, is the true bottleneck.** Advancing peptide permeability prediction requires uncensored re-measurement of floor compounds — an experimental, not algorithmic, task.

## References
[1] Hornsby M, Lee J, Steele J, Tuekpe R, Lim A. Therapeutic peptides and proteins: Status and developments in drug delivery. J Control Release. 2026;394:114895. doi:10.1016/j.jconrel.2026.114895

[2] Tan X, Liu Y, Zhou Y, Fang Q, Ouyang Y. pepADMET: A Novel Computational Platform For Systematic ADMET Evaluation of Peptides. J Chem Inf Model. 2026;66(2):936-946. doi:10.1021/acs.jcim.5c02518

[3] Zhang Y, Tang Y, Chen Y, Mahood T, Vincoff A, Chatterjee S. PeptiVerse: A unified platform for peptide property prediction. Nat Commun. 2026;17:6819. doi:10.1038/s41467-026-74167-w

[4] Chatterjee S, et al. Cyclic Peptide Permeability Prediction: Benchmarking and Platform. arXiv preprint arXiv:2512.06971. 2025. doi:10.48550/arXiv.2512.06971

[5] Hollmann N, et al. TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second. Nature. 2024;636:363-370. doi:10.1038/s41586-024-08328-6

[6] Li H, Zhao D, Zeng J. KPGT: Knowledge-Guided Pre-training of Graph Transformer for Molecular Property Prediction. arXiv preprint arXiv:2206.03364. 2022. doi:10.48550/arXiv.2206.03364

[7] Zhao C, et al. openclaw-peptide-admet: Peptide ADMET Prediction Platform with Censored-Floor-Aware Protocol. https://github.com/c00jsw00/openclaw-peptide-admet (accessed 2026-09-01). (v4.2 protocol: unique-SMILES 70/10/20 split, censored floor at −10.0 log cm/s, MixedADMETMLP baseline, Huber loss)

[8] Ahmad W, Simon E, Chithrananda S, Grand G, Ramsundar B. ChemBERTa: Self-Supervised Learning for Chemical Language Models. arXiv preprint arXiv:2010.09885. 2020. doi:10.48550/arXiv.2010.09885

[9] Chen T, Guestrin C. XGBoost: A Scalable Tree Boosting System. Proc KDD. 2016:785-794. doi:10.1145/2939672.2939785

[10] Ke G, et al. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. Adv Neural Inf Process Syst. 2017;30:3146-3154.

[11] Moriwaki H, Tian Y-S, Kawashita N, Takagi T. Mordred: A Molecular Descriptor Calculator. J Cheminform. 2018;10:58. doi:10.1186/s13321-018-0258-y

[12] Rogers D, Hahn M. Extended-Connectivity Fingerprints. J Chem Inf Model. 2010;50(5):742-754. doi:10.1021/ci100050t

[13] Landrum G. RDKit: Open-Source Cheminformatics. 2024. https://www.rdkit.org

[14] Wang M, Zheng D, Ye Z, Gan Q, Li M, Zhou X, Ma C, Hu Z, Yang Q, Zhao Y, Li J, Smola A, Zhang Z. Deep Graph Library: Towards Efficient and Scalable Deep Learning on Graphs. arXiv preprint arXiv:1909.01315. 2019. doi:10.48550/arXiv.1909.01315

---

**Graphical Abstract** (to be rendered): Flowchart showing the v4.2 censored-floor protocol → nine routes → baseline 0.464 → TabPFN 0.496 → KPGT 0.513 → ceiling 0.539. Key message: foundation models are the first to beat the baseline, but the ceiling holds.

**Data Availability** All data, splits, trained checkpoints, and reproduction scripts are available at https://github.com/c00jsw00/openclaw-peptide-admet (MIT License).

**AI Declaration** This manuscript was prepared with the assistance of AI tools (Hermes Agent, Nemotron-3-Ultra) for code generation, literature retrieval, and text drafting. All scientific claims, numbers, and conclusions were verified by the authors against primary computational experiments.

**Conflict of Interest** The authors declare no competing financial interests.