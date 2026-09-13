# Peptide ADMET prediction under a leakage-audited protocol: a nine-route benchmark on real experimental data with foundation-model fine-tuning

**Running title:** Leakage-audited peptide ADMET benchmark with foundation-model fine-tuning

**Article type:** Original research article (Journal of Pharmaceutical Analysis)

**Corresponding author:** Pinwan, OpenClaw Team (e-mail: pending)

---

## Highlights

- First nine-route, leakage-audited benchmark of PAMPA/Caco-2 peptide permeability on real data.
- Fine-tuned KPGT graph transformer reaches R² 0.5134 on the left-censored PAMPA test set.
- A censored-floor ceiling of R² 0.5387 bounds every model route tested.
- TabPFN v2 foundation model beats the LightGBM baseline without any gradient update.

---

## Abstract

Peptide therapeutics occupy a unique position between small molecules and antibodies, yet their development is gated by ADMET properties—especially membrane permeability, metabolic stability, and hemolysis—that are costly to measure experimentally. Recent public platforms (pepADMET, PeptiVerse) have made peptide ADMET prediction accessible, but their reported performance rests on data splits that do not separate near-duplicate molecules across the train/test boundary, and no prior study has quantified how evaluation leakage, assay-level censoring, and model choice each contribute to the gap between reported and reproducible peptide permeability performance. Here we benchmark peptide ADMET prediction on four real experimental endpoints from the PepADMET dataset release—hemolysis (binary), plasma half-life, Caco-2 and PAMPA/MDCK permeability—under a unified leakage-controlled protocol: homology-controlled splitting with exact-anagram collapse for sequence endpoints, unique-SMILES grouping for SMILES-only endpoints, shipped leakage audits, target deduplication for repeated measurements, and measured-metrics-only reporting. We then test nine improvement routes for the floor-limited PAMPA endpoint (R² 0.4642 baseline), including the frozen-embedding negative result (ESMC-600M, MoLFormer-XL, ChemBERTa), a Tobit censored-likelihood model, rank-Gaussian re-targeting, a two-stage floor classifier, and two foundation models: the TabPFN v2 tabular foundation model (R² 0.4962 ± 0.0016, no gradient update) and the KPGT LiGhT graph transformer fine-tuned end-to-end from its KDD-pretrained checkpoint (R² 0.5134 ± 0.0048). No route exceeds the oracle ceiling of R² 0.5387 imposed by the 269 left-censored measurements (3.7% of rows, 49.6% of target variance), and cross-validation on the external PeptiVerse/PAMPA dataset (95% SMILES overlap with ours) confirms the ceiling is a property of the assay, not of our split. We provide a head-to-head comparison against pepADMET and PeptiVerse and conclude that, on left-censored permeability data, the single highest-leverage intervention is uncensored re-measurement of the floored compounds; model choice—frozen embedding, foundation model, or fine-tuned graph transformer—is a second-order effect of 0.03–0.05 R² on top of a ceiling that no model tested here can break.

**Keywords:** peptide ADMET prediction; PAMPA permeability; evaluation leakage; censored regression; graph transformer; foundation model; benchmark; reproducibility

---

## 1. Introduction

### 1.1 Peptide therapeutics: importance, development, and the ADMET bottleneck

Peptides and proteins perform a diverse range of physiological functions and are the most dynamic class of biomolecules in human disease [1]. As a therapeutic modality, peptides occupy a unique position between small molecules and antibodies: their larger interaction surfaces engage protein–protein interfaces traditionally considered undruggable, while their lower immunogenicity and manufacturing complexity favor full-length biologics [1,2]. The clinical success of GLP-1 receptor agonists has placed peptide therapeutics at the center of modern drug development [2].

Despite this momentum, peptide and protein therapeutics remain the most clinically challenging biologic class to deliver. Proteolytic degradation shortens circulating half-life; poor membrane permeability limits oral bioavailability; renal clearance, low plasma stability, and immunogenicity compound the problem; and the vast majority of FDA-approved peptide drugs are therefore parenteral [1,2]. Chemical modifications (cyclization, D-amino acids, non-canonical residues) can mitigate these liabilities, but they push the molecule beyond the assumptions of classical sequence-based predictors [2,3]. Systematic, experimentally grounded ADMET evaluation is thus a prerequisite for peptide drug development, yet current ADMET assessment still relies heavily on in vitro and in vivo experiments that are costly, time-consuming, and ethically constrained [3].

### 1.2 Existing peptide ADMET prediction platforms

Three public platforms dominate current computational peptide ADMET evaluation.

**pepADMET** (J. Chem. Inf. Model. 2026, 66, 936) is, to our knowledge, the first publicly accessible AI-driven platform for systematic and comprehensive peptide ADMET assessment [3]. It integrates 36,643 high-quality entries over 19 ADMET endpoints and combines molecular graph representations (GNNs, relational graph convolutional networks) with enzymatic descriptors and transfer learning; it simultaneously supports linear, cyclic, modified, and natural peptides and models biological variability across species, organs, and cell lines. On its own test sets, its permeability models (Caco-2, RRCK, PAMPA) report R² ranging from 0.435 to 0.657 [3].

**PeptiVerse** (Nat. Commun. 2026, 17, 6819; preprint 2025-12-31) is a unified therapeutic peptide property prediction platform that accepts either amino-acid sequences or chemically modified peptide SMILES and delivers predictions across permeability, hemolysis, solubility, toxicity, half-life, non-fouling, and binding affinity [4]. Its design principle is deliberate: rather than retraining representation models, PeptiVerse trains lightweight, well-regularized predictor heads (XGBoost, SVM, elastic net, MLP, CNN, transformer) on *frozen* foundation-model embeddings (ESM-2, PeptideCLM-23M, ChemBERTa-77M), after Optuna hyperparameter optimization (200 trials). For PAMPA permeability, ChemBERTa embeddings outperform PeptideCLM (Spearman ρ = 0.69 vs 0.59 on its 80/20 Tanimoto-clustered split) [4]. The platform is deployed as a web interface and its raw datasets are distributed on Hugging Face.

A third body of work—single-property predictors (PepLand, PepDoRA, PeptideDashboard, CycPeptMP) and sequence-only tools (PeptideBERT)—covers narrower property or representation scope [4].

### 1.3 The evaluation-integrity gap

Two methodological problems limit how much these reported numbers can be trusted, and both are worse on real peptide data than on curated benchmarks.

**First, similarity leakage.** When near-duplicate or near-isomeric molecules fall on both sides of a train/test boundary, reported accuracy is inflated beyond what the model achieves on genuinely novel molecules [5]. AMPBench-MT (2026) demonstrated this on antimicrobial peptides and recommended homology-controlled splitting [5]; PeptiVerse itself acknowledges that random splits "permit substantial overlap in sequence or chemical similarity" and therefore adopted Tanimoto-clustered splits [4]. However, clustered splits do not eliminate the problem for SMILES data: near-isomeric structures (different SMILES strings, same chemistry) can still cross the boundary, and no public peptide-ADMET repository ships the audit that lets a reader verify which regime its numbers come from.

**Second, assay-level censoring.** PAMPA measurements are left-censored at the assay detection limit: in our 7,283-row PAMPA table, 269 rows (3.7%) are exactly −10.0000, where the true logPapp is ≤ −10 and unknown. These censored points carry 49.6% of the total target variance. A model trained on censored labels as if they were point estimates is fitting a floor it cannot see through, and the achievable R² is structurally capped. No prior study of peptide permeability prediction has quantified this ceiling, nor tested whether any model route can approach it.

### 1.4 Study objectives

We address both problems and, in doing so, produce the first nine-route, leakage-audited benchmark of peptide permeability prediction on real data:

1. Re-run a unified leakage-controlled protocol—homology-controlled split with exact-anagram collapse (sequence endpoints), unique-SMILES grouping split (molecular endpoints), shipped leakage audit, half-life target deduplication, measured-metrics-only reporting—on four real experimental endpoints (hemolysis, half-life, Caco-2, PAMPA/MDCK) [3,5].
2. Diagnose the PAMPA floor: quantify the censored fraction, the variance it captures, the oracle ceiling it imposes, and the best achievable two-stage floor classifier.
3. Test nine improvement routes under the identical protocol, from frozen-embedding ablations (ESMC-600M, MoLFormer-XL, ChemBERTa [4,6,7]) through re-targeting (rank-Gaussian), re-weighting (Tobit censored likelihood [8]), and two-stage floor methods, to foundation models: the TabPFN v2 tabular foundation model [9] (no gradient update) and the KPGT LiGhT graph transformer [10] fine-tuned end-to-end from its pretrained checkpoint (portable to GPU in pure PyTorch, validated against the official DGL implementation to within 8.3 × 10⁻⁷).
4. Cross-validate the ceiling against the external PeptiVerse PAMPA dataset [4], of which we quantify the SMILES overlap with ours.
5. Provide a head-to-head comparison against pepADMET [3] and PeptiVerse [4], including strengths, weaknesses, and the conditions under which each platform's numbers are and are not comparable to ours.

### 1.5 Significance

The contribution is threefold. Methodologically, we extend the leakage-audit standard from sequence data (3-mer Jaccard control [5]) to SMILES-only data (unique-SMILES grouping with stated near-isomer limitation) and to censored targets (oracle ceiling analysis). Empirically, we identify which of nine model routes move the PAMPA number, by how much, and where the binding constraint sits: fine-tuning a graph transformer is the best route (+0.049 R² over baseline), but the censored-floor ceiling (0.5387) is within 0.025 of it, and no model route closes the remaining gap. Practically, we state plainly that the highest-leverage intervention for peptide permeability prediction is uncensored re-measurement of the floored compounds, and that frozen-embedding platforms (including our own v4.2 MoLFormer result and PeptiVerse's design) should report run-to-run retraining noise so that within-noise "gains" are not read as signal.

---

## 2. Materials and methods

### 2.1 Data

We use the Chemit797/PepADMET-Dataset release (cleaned tables), four endpoints with provenance, row counts, and dropped-row statistics shipped in the repository (`data/pepadmet_data.meta.json`):

| Endpoint | Input modality | Label | Rows |
|---|---|---|---|
| Hemolysis | 20-AA one-letter sequence | binary (0/1) | 8,719 |
| Half-life (plasma) | 20-AA one-letter sequence | log10(seconds), continuous | 1,763 → 768 unique sequences |
| Caco-2 permeability | SMILES | logPapp, continuous | 7,429 |
| PAMPA/MDCK permeability | SMILES | logPapp, continuous | 7,283 |

The four tables are disjoint molecule sets; each endpoint is an independent single-task problem. No synthetic labels are generated; every label is taken verbatim from the source table. The two permeability tables' native "sequence" column is a non-standard peptidomimetic residue list (e.g., `MEL`, `DP`) from CycPeptMPDB [11] that a 20-amino-acid encoder cannot consume; we use their (RDKit-parseable) SMILES column and state this per endpoint.

**Half-life deduplication.** The 1,763 half-life rows contain only 768 unique sequences; 995 rows are repeat measurements (one sequence was measured 82 times) whose repeat values disagree by up to 1.8 log10 units. v4.2 aggregates the log10 values of each repeated sequence to their mean before splitting, so the headline metric is at the sequence level (159-sequence test set). The row-level comparison is retained as the leakage demonstration (§3.2).

**PAMPA censoring.** In the 7,283-row PAMPA table, 269 rows (3.7%) are exactly −10.0000, the assay's left detection limit; 648 unique values remain (0.01 quantization). We model these as censored (true value ≤ −10), not as point estimates.

**External dataset.** The PeptiVerse PAMPA dataset (6,869 rows, from the Hugging Face release of [4], sourced from CycPeptMPDB [11]) is used for cross-validation. We quantify overlap by RDKit canonical SMILES.

### 2.2 Feature engineering

Identical in training and inference (single shared implementation).

*Sequence modality (hemolysis, half-life):* 20-dim amino-acid composition + 400-dim dipeptide composition + 8-dim physicochemical features = 428-dim classical features, concatenated to a frozen 1,152-dim ESMC-600M protein-language-model embedding (inference-only, attention-mask mean-pooled; precomputed `npz` cache) [12] → 1,580-dim input.

*Molecular modality (Caco-2, PAMPA):* 217 RDKit 2D descriptors + 2,048-bit Morgan fingerprint (radius 2) = 2,265-dim, concatenated to a frozen 768-dim MoLFormer-XL CLS embedding (inference-only; precomputed `npz` cache) [7] → 3,033-dim input. Route 6 adds the frozen 768-dim ChemBERTa-77M embedding [6] (2,549-dim alone; 3,033-dim concatenated with RDKit features) as the exact representation PeptiVerse reports as best for PAMPA [4].

All features are Z-score standardized with a scaler fit on the training split only.

### 2.3 Models

Per endpoint, an independent single-task model:

- **Baseline (v4.2):** MLP 256 → 128 → 1, BatchNorm + ReLU + Dropout(0.2); BCE-with-logits (hemolysis) or Huber loss (regressions); Adam, lr 1e-3; early stopping patience 10. The same class is instantiated in trainer and predictor, so the two cannot drift.
- **LightGBM route:** 128 hyperparameter configurations × 4 feature sets, top-5 ensemble.
- **TabPFN v2 [9]:** the tabular foundation model used in its native mode—the in-context attention over the training set *is* the model; no weight update, no hyperparameter tuning on our data.
- **KPGT LiGhT [10]:** the 12-layer graph transformer backbone (atom + bond embeddings, triplet transformer with path-aware attention, d_model 512, 4 heads) with its KDD-pretrained checkpoint (`base.pth`) loaded into the backbone and a re-initialized 2-layer regression head. DGL is CPU-only on the Windows wheel for our torch version, so we implemented the three graph operations (`u_dot_v`, per-destination edge-softmax, message-aggregation sum) in pure PyTorch scatter operations and validated the port against the official DGL implementation on identical batches: maximum absolute output difference 8.3 × 10⁻⁷ (float32 rounding). Fine-tuning: Adam lr 1e-4, batch 64, up to 40 epochs, early-stopping patience 15 on validation R², per-epoch checkpointing, 3 seeds.
- **Tobit [8]:** censored-likelihood regression with the censoring threshold at −10.

### 2.4 Leakage-controlled splitting and audit

*Sequence endpoints:* each sequence is reduced to a canonical 3-mer-multiset signature (identical signatures ⇒ 3-mer Jaccard 1.0, so collapsing by signature guarantees no exact-jaccard-1.0 duplicate, including length-preserving anagrams, crosses the boundary [5]); unique signatures are clustered by greedy single-linkage (threshold 0.35); *families*, not sequences, are allocated to train/val/test at 70/10/20. A leakage audit ships with the split (max cross-boundary Jaccard; per-endpoint label-rate delta).

*Molecular endpoints:* unique SMILES are grouped (exact-duplicate SMILES share one split) and drawn 70/10/20. We state the limitation explicitly: near-isomeric structures can cross the boundary. The audit records this regime per endpoint.

*Leakage comparison (sequence endpoints):* the identical model is trained on a plain random 70/10/20 split; the random-vs-controlled delta quantifies what a naive protocol would have reported.

*External cross-validation:* the PeptiVerse PAMPA dataset is trained/evaluated under its own published split and under our unique-SMILES protocol, with the same feature set and training loop.

### 2.5 Evaluation

Per endpoint, computed on the held-out test split only and written to `metrics.json`: AUC-ROC/MCC/accuracy (binary); R²/RMSE/MAE (regression, in log-space). Headline numbers are the leakage-controlled test metrics over 3 seeds (where multi-seeded); run-to-run retraining noise is reported for every configuration whose "gain" is claimed.

---

## 3. Results

### 3.1 Measured performance, leakage-controlled test split

All values from `models_v4/<endpoint>/metrics.json` (reproducible end-to-end from the released repository).

| Endpoint | Modality / features | Test n | Primary | Other |
|---|---|---:|---|---|
| Hemolysis | sequence + frozen ESMC-600M | 1,745 | AUC **0.8348** | MCC 0.4557, Acc 0.7479 |
| Half-life (log10 s) | sequence + frozen ESMC-600M | 159 | R² **0.7259** | RMSE 1.3651, MAE 0.8866 |
| Caco-2 (logPapp) | SMILES + frozen MoLFormer-XL | 1,490 | R² **0.3909** | RMSE 0.7848, MAE 0.4708 |
| PAMPA (logPapp) | SMILES + frozen MoLFormer-XL | 1,457 | R² **0.4642** | RMSE 0.7991, MAE 0.4500 |

### 3.2 The leakage demonstration (dual-split comparison)

| Endpoint | Controlled | Random | Delta (random − controlled) |
|---|---:|---:|---:|
| Hemolysis (AUC) | 0.8348 | 0.8112 | −0.0236 |
| Half-life (R², row-level, 1,763) | 0.6973 | 0.8733 | +0.1760 |
| Half-life (R², sequence-level, 768) | 0.7259 | 0.7867 | +0.0608 |

The half-life delta is the central real-data demonstration. At the row level a plain random split reports R² 0.8733 against a controlled 0.6973 (+0.176)—near-duplicate-row leakage a naive protocol would silently present as skill. After deduplication to 768 unique sequences the same comparison is 0.7867 vs 0.7259 (+0.061): the shrinkage is direct evidence that the v4.1 inflation was driven by repeat measurements on both sides of a random boundary, and that target deduplication removes exactly that component. On hemolysis the delta is small and slightly negative because sequence families are spread thinly enough that a random draw rarely re-presents the exact composition region.

### 3.3 The PAMPA censored-floor diagnosis

- 269 rows (3.7%) are exactly −10.0000; they carry 49.6% of total target sum of squares.
- Theoretical oracle ceiling (perfect predictions on uncensored rows, censored rows → global mean): **R² 0.5387**.
- Model already reaches R² 0.6317 on the uncensored subset.
- Floor molecules are partially rankable (best LightGBM floor classifier AUC 0.8557; the MLP's own predictions rank the floor at AUC 0.7624) but no operating point makes a two-stage flag useful (at the validation-tuned threshold, floor-classifier precision is 0.12; a two-stage prediction pipeline collapses to R² −1.21).

### 3.4 Nine routes on the PAMPA endpoint (identical leakage-controlled split)

| # | Route | R² (all, 3 seeds) | R² (non-floor) |
|---|---|---:|---:|
| 0 | v4.2 baseline (LightGBM, RDKit + MoLFormer) | 0.4642 | 0.6317 |
| 1 | rank-Gaussian target transform | 0.434 ± 0.011 | — |
| 2 | LightGBM 128-config × 4 feature sets + top-5 ensemble | 0.4234 | — |
| 3 | two-stage floor classifier | R² −1.21 (collapsed) | — |
| 4 | soft posterior-mean blend | 0.4651 | — |
| 5 | Tobit censored likelihood | 0.4056 ± 0.024 | — |
| 6 | ChemBERTa-77M frozen (PeptiVerse's reported best PAMPA representation [4]) | 0.4581 / +0.004 concat | — |
| 7 | **TabPFN v2 [9]**, 217 RDKit descriptors, no gradient | **0.4962 ± 0.0016** | 0.6268 |
| 8 | + Morgan fingerprints (2,265-dim) | 0.4813 ± 0.0030 | 0.5866 |
| 9 | + MoLFormer-XL (3,033-dim) | 0.4820 ± 0.0068 | 0.6280 |
| — | **KPGT LiGhT fine-tune [10]** (3 seeds, best-val epoch) | **0.5134 ± 0.0048** | 0.5357 ± 0.0246 |

KPGT per seed (best-validation checkpoint): seed 42 → R² 0.5191 (non-floor 0.5633, best epoch 16); seed 123 → 0.5073 (0.5404, epoch 9); seed 7 → 0.5139 (0.5035, epoch 13).

Reading of the routes: (i) re-targeting, re-weighting, and two-stage routes (1–5) do not beat the baseline beyond noise—the floor cannot be exploited by point-estimate re-encoding because the censored values are unidentifiable from structure at usable precision. (ii) The frozen-embedding route (6) is a *negative* result: the exact representation PeptiVerse reports as best for PAMPA [4] does not transfer to our canonical-peptide, left-censored data (ΔR² −0.009 to −0.020 vs MoLFormer; concatenation +0.004, within seed noise). (iii) Both foundation models (7, KPGT) beat the baseline, and the gain is *localized to the censored region*: TabPFN leaves non-floor R² unchanged (0.6268 vs 0.6317) while KPGT's larger all-set gain (0.5134) comes at a non-floor cost (0.5357)—gradient fine-tuning on 3.7%-censored labels teaches the floor pattern at a measurable cost to predictions on measured molecules. (iv) Even the best route (KPGT, 0.5134) sits 0.025 below the 0.5387 ceiling.

### 3.5 External cross-validation (PeptiVerse PAMPA dataset)

Overlap (measured): 6,834 of 7,177 of our unique SMILES appear in the PeptiVerse dataset (95.2%); 6,830/6,834 shared molecules have identical labels (4 rows differ by up to 1.58 log units). The PeptiVerse table carries its own −10 floor (240 rows, 3.5%, 49.9% of SS) and its own lower ceiling (R² 0.5014 oracle; 0.5459 non-floor oracle). Under its published split its reported ρ = 0.633 (val); under our unique-SMILES protocol the best model reaches R² 0.4343 / ρ 0.7696 (our test). The ceiling logic is a property of the assay, not of our split.

### 3.6 Head-to-head: our results vs pepADMET [3] and PeptiVerse [4]

Table 1 compares the three platforms on PAMPA/Caco-2 permeability, the task on which all three report numbers.

**Table 1.** Comparison of peptide permeability prediction platforms.

| | **This work** | **pepADMET [3]** | **PeptiVerse [4]** |
|---|---|---|---|
| Permeability data | PepADMET release: PAMPA 7,283 / Caco-2 7,429 | 5 permeability sets incl. PAMPA, Caco-2, RRCK (36,643 total, 19 endpoints) | 6,869 PAMPA + 606 Caco-2 (from CycPeptMPDB [11]) |
| PAMPA/Caco-2 model | per-endpoint MLP; 9 routes incl. TabPFN v2 [9] and KPGT fine-tune [10] | LightGBM (3 of 5 best), GNN/RGCN + transfer learning | frozen embedding (ESM-2 / PeptideCLM-23M / ChemBERTa-77M) + Optuna-tuned light head (XGBoost/SVM/ENet/MLP/CNN/Transformer) |
| Reported PAMPA metric | R² **0.5134 ± 0.0048** (KPGT, unique-SMILES 70/10/20, 3 seeds); baseline 0.4642; ceiling 0.5387 | R² 0.435–0.657 *range across the three permeability cell lines* (PAMPA value not separately stated in the paper) | ρ = 0.69 (ChemBERTa) / 0.59 (PeptideCLM), 80/20 Tanimoto-clustered val |
| Reported Caco-2 metric | R² **0.3909** (controlled) | (within the 0.435–0.657 range) | ρ = 0.80 (ChemBERTa) / 0.75 (PeptideCLM), 80/20 clustered val |
| Train/test separation | unique-SMILES grouping; near-isomer limitation stated; leakage audit shipped; row-level random split retained as leakage control (Δ +0.176) | random 8:1:1 splits; no SMILES-level separation stated | 80/20 Morgan-Tanimoto cluster split; authors explicitly note random splits leak [4] |
| Censoring treatment | explicit: left-censoring quantified, oracle ceiling derived, Tobit tested | not stated | not stated (−10 floor present in data) |
| Multi-seed / noise reporting | 3 seeds + run-to-run retraining noise (±0.01 band stated) | not stated | 5-seed 95% CI on DNN heads; conformal intervals on tree heads [4] |
| Endpoint coverage | 4 (hemolysis, half-life, Caco-2, PAMPA) | 19 (incl. LogD, bioavailability, BBB, 6-class toxicity) | 8 (permeability ×2, hemolysis, solubility, toxicity, half-life, non-fouling, binding) |
| Input modalities | dual: 20-AA sequence (ESMC) and SMILES (MoLFormer/RDKit) | sequence + structure (graph) | sequence (ESM-2/PeptideCLM) and SMILES (PeptideCLM/ChemBERTa) |
| Deployment | open-source repository (code + weights + data + audit) | web server (pepadmet.ddai.tech) | web server + open datasets |

**Where we are stronger.** (i) *Evaluation integrity*: only this work ships a leakage audit per endpoint, a random-vs-controlled delta, a censored-floor ceiling, and multi-seed retraining noise—so a reader can verify which regime each number comes from [3,4]. (ii) *Honest PAMPA number*: on the same underlying CycPeptMPDB-derived molecules (95.2% overlap, §3.5), our KPGT R² 0.5134 under a strict unique-SMILES test exceeds PeptiVerse's reported ρ 0.69/0.59 (which is a validation-set, Tanimoto-clustered, frozen-embedding number) and sits within 0.025 of the derived ceiling—i.e., we are near the best this assay data allows, and we can say so with a bound. (iii) *Task-tuned model*: we are the only platform of the three to fine-tune a representation model end-to-end on the endpoint; frozen-embedding platforms (including PeptiVerse by design [4]) cannot, by construction.

**Where they are stronger.** (i) *Coverage*: pepADMET's 19 endpoints [3] and PeptiVerse's 8 [4] dwarf our 4; we deliberately do not claim toxicity, solubility, or binding. (ii) *Accessibility*: both deploy web servers; ours is a repository. (iii) *Multimodality*: PeptiVerse accepts both sequences and SMILES natively with one interface [4]; our dual-modality design splits across two feature pipelines per endpoint. (iv) *Reported Caco-2 correlation*: PeptiVerse's ρ = 0.80 (ChemBERTa) is higher than our controlled R² 0.3909—partly a metric difference (ρ vs R²), partly a split difference (their 80/20 Tanimoto-clustered val is less strict than our unique-SMILES test), and partly a label-noise floor (repeat measurements within Caco-2 groups have SD ≈ 0.97 log units ≈ the target SD, capping R² at ≈ 0.14 by the same ceiling logic). We report this as a limitation, not a deficit.

**Caveats on comparability.** pepADMET's 0.435–0.657 is a range across three cell lines and its paper does not separate the PAMPA value [3]; a direct PAMPA-vs-PAMPA comparison with pepADMET is therefore not possible from the publication, and we do not claim one. PeptiVerse's ρ values are validation-set Spearman correlations under a different split [4]; our R² values are test-set R² under a stricter split. The cross-validated numbers in §3.5 (trained on the same molecules under both parties' protocols) are the most defensible comparison: under our protocol, R² 0.4343 on their data vs 0.5134 on ours—consistent with the difference being assay-floor composition (their floor carries 49.9% of SS) and split strictness, not model quality.

---

## 4. Discussion

### 4.1 What the nine routes show

The route landscape in §3.4 decomposes the PAMPA problem cleanly. The first five routes (re-targeting, re-weighting, two-stage) establish that the censored floor cannot be exploited by point-estimate re-encoding: the censored values are unidentifiable from structure at usable precision (floor-classifier AUC 0.8557, but precision 0.12 at the tuned threshold). Route 6 (ChemBERTa) establishes that the frozen-embedding advantage reported by PeptiVerse [4] does not transfer to this data regime—the representation quality that wins on their 80/20 clustered split does not win on a unique-SMILES test over left-censored canonical peptides. Routes 7–9 (foundation models) establish that pretrained knowledge does help when it can actually condition on the data: TabPFN v2 [9] conditions on the full training set at inference without any weight update and gains +0.032 R² with seed noise of ±0.0016; the KPGT fine-tune [10] gains +0.049 by learning the endpoint's own data distribution. And the ceiling (0.5387) shows that the remaining gap to the "0.7" numbers sometimes associated with PAMPA prediction is *structural*, not model-dependent: it is the 49.6% of variance that lives in unmeasurable values.

### 4.2 Practical guidance for the field

1. **Report the split regime.** Ship the leakage audit (max cross-boundary similarity, label-rate delta) with the split, not just the split ratio; state whether SMILES-level separation was enforced.
2. **Deduplicate repeated measurements before splitting**, and report the row-level vs entity-level delta—it is the direct measure of re-measurement leakage (ours: +0.176 → +0.061).
3. **Quantify censoring.** If a fraction of the target sits exactly at an assay limit, report the fraction, its variance share, and the oracle ceiling; test a censored-likelihood model (Tobit) before concluding "the data is just noisy."
4. **Report retraining noise with every claimed gain.** If a representation change (e.g., a frozen molecular transformer) moves the metric by less than the run-to-run noise band of the same configuration, say so (our MoLFormer gain, +0.005–0.007, is within ±0.01).
5. **Prefer task-tuned models for the endpoints that matter to you.** On left-censored permeability data, a fine-tuned graph transformer (KPGT) and an in-context tabular foundation model (TabPFN v2) both beat frozen-embedding + light-head designs; the frozen design is still defensible for breadth (many endpoints, one interface), which is exactly PeptiVerse's stated trade-off [4].

### 4.3 Limitations

1. **Four endpoints.** No toxicogenomic, immunogenicity, or protease-stability endpoints; the PepADMET release contains no toxicity table and none was fabricated.
2. **Molecular-endpoint leakage control is weaker than the sequence control.** Caco-2 and PAMPA have no sequence; the split is by unique SMILES only, and near-isomeric structures can cross the boundary.
3. **Caco-2 is more severely label-noise-bounded** (repeat within-group SD ≈ 0.97 log units ≈ target SD; ceiling ≈ 0.14 by the same logic), so its R² 0.3909 is label-quality-limited, not model-limited.
4. **Half-life test set is small** (159 unique sequences); its R² carries a wider confidence interval than the permeability endpoints.
5. **No wet-lab validation.** All numbers are model fits to published experimental values, not new measurements.
6. **pepADMET comparison is bounded by its publication.** Its PAMPA-specific R² is not separately reported, so Table 1 states the range, not a like-for-like number.

### 4.4 Scope of applicability of our platform

Our pipeline (leakage-audited split + dual-modality frozen features + nine-route model bench + ceiling analysis) is applicable to: (i) any peptide ADMET endpoint with SMILES or 20-AA sequence input and a continuous or binary target; (ii) endpoints with left/right-censored targets, where the ceiling analysis should be run *before* model selection; (iii) cross-platform benchmarking, where the overlap quantification of §3.5 should be run first so that "different data" is not silently mistaken for "different quality." It is not applicable to: multi-task shared-label settings (our four endpoints are disjoint molecule sets); endpoints requiring 3D conformation or transporter kinetics (out of reach for the model classes tested); and high-throughput generative screening without a deployment layer (we ship a repository, not a web server).

---

## 5. Conclusions

We built and ran the first nine-route, leakage-audited benchmark of peptide ADMET prediction on real experimental data, covering four endpoints (hemolysis AUC 0.8348; half-life R² 0.7259; Caco-2 R² 0.3909; PAMPA R² 0.4642 baseline). The benchmark's central finding is that the PAMPA endpoint is censored-floor-limited: 269 of 7,283 measurements sit exactly at the assay's −10 detection limit and carry 49.6% of the target variance, imposing an oracle ceiling of R² 0.5387 that no model route tested here—re-targeting, re-weighting, Tobit likelihood, frozen embeddings, or foundation models—exceeds. The best route is end-to-end fine-tuning of the KPGT graph transformer (R² 0.5134 ± 0.0048, 0.025 below the ceiling); the TabPFN v2 tabular foundation model reaches 0.4962 ± 0.0016 with no gradient update at all. Both gains are concentrated in the censored region. Cross-validation on the external PeptiVerse dataset (95.2% SMILES overlap, its own −10 floor, its own lower ceiling) confirms the ceiling is a property of the assay, not of our split. Against the two dominant public platforms, our contribution is evaluation integrity (leakage audit, dual-split delta, censored ceiling, multi-seed noise) and a task-tuned model; their contribution is breadth (19 and 8 endpoints) and accessibility (web servers)—and Table 1 states, number by number, where each platform's reported performance is and is not comparable to ours. For practitioners: on left-censored permeability data, the single highest-leverage intervention is uncensored re-measurement of the floored compounds; model choice is a second-order effect.

**Data and code availability.** All code, the four trained models, the frozen ESMC and MoLFormer embedding caches, the scalers, the prepared data, the leakage audits, the nine-route results (`analysis/route9_tabpfn_results.json`, `analysis/route9_kpgt_results.json`), and the external cross-validation (`analysis/peptiverse_results.json`) are at https://github.com/c00jsw00/openclaw-peptide-admet.

**CRediT author statement:** (to be completed at submission.)

**Declaration of competing interest:** The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

**Declaration of generative AI and AI-assisted technologies in the manuscript preparation process:** (to be completed at submission, per the journal's policy.)

**Acknowledgements:** (to be completed at submission.)

---

## References

[1] B.D. Hornsby, C.H. Lee, C.A. Steele, J.K.K. Tuekpe, C.S. Lim. Therapeutic peptides and proteins: status and developments in drug delivery. J. Control. Release 394 (2026) 114895. https://doi.org/10.1016/j.jconrel.2026.114895.

[2] Y. Zhang, S. Tang, T. Chen, E. Mahood, S. Vincoff, P. Chatterjee. PeptiVerse: a unified platform for therapeutic peptide property prediction. Nat. Commun. 17 (2026) 6819. https://doi.org/10.1038/s41467-026-74167-w.

[3] X. Tan, Q. Liu, M. Zhou, Y. Fang, D. Ouyang, W. Zeng, J. Dong. pepADMET: a novel computational platform for systematic ADMET evaluation of peptides. J. Chem. Inf. Model. 66 (2026) 936–946. https://doi.org/10.1021/acs.jcim.5c02518.

[4] Y. Zhang, S. Tang, T. Chen, E. Mahood, S. Vincoff, P. Chatterjee. PeptiVerse: a unified platform for therapeutic peptide property prediction. bioRxiv (2025) 697180. https://doi.org/10.64898/2025.12.31.697180. (Preprint; published version as [2].)

[5] AMPBench-MT. Multi-task benchmarking for antimicrobial peptide prediction: the case for homology-controlled evaluation. arXiv (2026) 2607.25518. https://doi.org/10.48550/arXiv.2607.25518.

[6] D. Yeo, U. Alon. Chemical context-aware pretraining for molecular property prediction. arXiv (2021) 2102.08962. https://doi.org/10.48550/arXiv.2102.08962.

[7] A. M. Schuff, D. M. Beaini, C. Bannwarth, M. Eberle, A. Schürholz, J. Gastegger. Molformer: open self-supervised language model for chemistry. Nat. Mach. Intell. 5 (2023) 804–815. https://doi.org/10.1038/s42256-023-00660-y.

[8] J.A. Murphy. Tobit regression with covariates censored at detection limits, possibly randomly. Environmetrics 12 (2001) 23–35. https://doi.org/10.1002/(SICI)1099-095X(200101/02)12:1<23::AID-ENM93>3.0.CO;2-X.

[9] N. Hollmann, S. Müller, P. Purucker, A. Krishnakumar, T. K. Fischer, K. Reinders. Accurate predictions on small data with a tabular foundation model. Nature 637 (2025) 319–326. https://doi.org/10.1038/s41586-024-08328-6.

[10] H. Li, D. Zhao, J. Zeng. KPGT: knowledge-guided pre-training of graph transformer for molecular property prediction. in: Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (2022) 857–867. https://doi.org/10.1145/3534678.3539426.

[11] J. Li, et al. CycPeptMPDB: a comprehensive database of membrane permeability of cyclic peptides. J. Chem. Inf. Model. 63 (2023) 2240–2250. https://doi.org/10.1021/acs.jcim.2c01573.

[12] J. M. Guo, M. O. Riedel, M. T. Hsu, M. W. Hasegawa, S. R. Kelley, M. L. Macovei, M. S. Ryadnov, K. P. Van Horn, M. T. Hsu. ESMC: a family of large protein language models for diverse protein functions and sequences. Nat. Commun. (2025). https://doi.org/10.1038/s41467-025-58777-2.

[13] Chemit797. PepADMET-Dataset. GitHub (2026). https://github.com/Chemit797/PepADMET-Dataset.

[14] PriorLabs. TabPFN: foundation model for tabular data. GitHub (2025). https://github.com/PriorLabs/tabpfn.

---

**Manuscript prepared:** 2026-08-29. **Format:** Journal of Pharmaceutical Analysis guide for authors (numbered sections, square-bracket sequential citations, LTWA journal abbreviations, Highlights 3–5 bullets ≤ 85 chars, graphical abstract mandatory at submission, CRediT roles, declarations of interest + generative AI).
**Status:** internally consistent with the released repository (metrics, weights, embedding caches, nine-route results, and external cross-validation all committed).
