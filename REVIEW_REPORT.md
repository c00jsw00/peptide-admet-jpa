# Nature/Science-Level Reviewer Report
## Manuscript: "Peptide ADMET Prediction: A Systematic Benchmark and Foundation Model Evaluation"
### Journal: Journal of Pharmaceutical Analysis (JPA)
### Reviewer: Simulated High-Impact Journal Reviewer

---

## Executive Summary

**Recommendation: Major Revision**

This manuscript presents a systematic benchmark of 9 improvement routes for cyclic peptide PAMPA permeability prediction, culminating in foundation model fine-tuning (KPGT, TabPFN v2) that achieves the best reported results on this dataset. The work is **methodologically sound, honestly reported, and reproducible** — rare qualities in ML-for-drug-discovery papers. However, several **Major** issues must be addressed before publication in JPA.

---

## Major Concerns (Must Fix)

### 1. **Table 1 Contains Unverified Numbers for Routes 1–5, 8**
**Location:** Table 1, rows "Route 1" through "Route 5" and "Route 8"
**Issue:** The manuscript reports specific R² values for Routes 1–5 (descriptor expansion, rank-Gaussian, ensemble, two-stage floor, soft blending) and Route 8 (label averaging), but **no JSON result files exist** for these routes in `analysis/`. Only the Python scripts exist (`round1_feature_ablation.py`, `round2_rank_gaussian.py`, `round3_strong_2d.py`, `tobit_censored.py`, `soft_blend.py`, `floor_predictability.py`, `label_avg_experiment.py`).
**Evidence:** 
- `chemberta_results.json` (Route 6) ✓ exists
- `route9_tabpfn_results.json`, `route9_kpgt_results.json` (Route 9) ✓ exist
- `peptiverse_results.json` (Route 7) ✓ exists
- `label_avg_results.json` (Route 8) ✓ exists
- **Routes 1–5: NO result JSON files**
**Required Action:** 
- Run all 5 route scripts and save results to `analysis/route{1-5}_results.json`
- Update Table 1 with **actual measured numbers** (not estimates from script print statements)
- If scripts are too slow, state this limitation explicitly and report what *was* measured

### 2. **PeptiVerse Comparison Uses Frozen Embeddings Only**
**Location:** Table 3, Discussion Section 4.2
**Issue:** The comparison vs PeptiVerse (Table 3) uses **frozen ChemBERTa embeddings** (E1_chem, E2_chem_2d) — the PeptiVerse paper's *weakest* configuration. The PeptiVerse paper also reports fine-tuned results (their Table 2) which are substantially better.
**Evidence:** 
- PeptiVerse NC (our split) E2_chem_2d: R²=0.434, Spearman=0.770
- PeptiVerse *their* split E2_chem_2d: R²=0.151, Spearman=0.645 (showing their split is harder)
- But PeptiVerse paper reports fine-tuned results not reproduced here
**Required Action:**
- Either: fine-tune ChemBERTa on our split for fair comparison, OR
- Explicitly state "We compare against PeptiVerse's *frozen embedding* baselines only; fine-tuned PeptiVerse results were not reproduced due to [compute/access constraints]"
- Do not imply KPGT fine-tune beats *all* PeptiVerse methods

### 3. **Non-Floor Regression Degradation Not Quantified in Abstract/Highlights**
**Location:** Abstract (Results), Highlights bullet 3
**Issue:** The most scientifically important finding — **KPGT fine-tune improves floor prediction but degrades non-floor regression (0.536 vs 0.632, −15%)** — is buried in Discussion. This trade-off is the *central mechanistic insight* and belongs in Abstract/Highlights.
**Required Action:** 
- Add to Abstract Results: "KPGT fine-tuning (R²=0.513) reaches 95% of the censored ceiling (0.539) but sacrifices non-floor regression accuracy (0.536 vs 0.632, −15%)"
- Add Highlight: "Foundation models trade non-floor regression accuracy for floor prediction gains"

### 4. **Missing: pepADMET Platform Comparison on *Identical* Data**
**Location:** Table 3, Section 3.3
**Issue:** Table 3 compares KPGT vs PeptiVerse on 6,834 shared SMILES, but **no comparison vs pepADMET's own reported numbers** on the same molecules. The pepADMET paper (Tan et al., 2026) reports results on their full dataset — a direct comparison on the 7,283 ∩ pepADMET overlap is missing.
**Required Action:**
- Identify overlap between our 7,283 SMILES and pepADMET's dataset
- Run KPGT/TabPFN on that overlap using pepADMET's split (or report why not possible)
- Add column to Table 3

### 5. **Graphical Abstract Placeholder Only**
**Location:** After Abstract
**Issue:** "[Graphical abstract placeholder: schematic of 9 routes → ceiling → KPGT fine-tune → trade-off]"
**JPA Requirement:** Graphical abstract is **mandatory** (Guide for Authors: "A graphical abstract is required...")
**Required Action:** Create actual graphical abstract (PNG/SVG, 1200×800 px minimum) showing:
- 9 routes flowchart
- Censored floor → ceiling concept
- KPGT fine-tune → best result
- Floor/non-floor trade-off arrow

---

## Minor Concerns (Should Fix)

### 6. **LightGBM Reference DOI Unverified**
**Location:** Reference [10] (LightGBM)
**Issue:** The canonical LightGBM paper (Ke et al., NIPS 2017 "LightGBM: A Highly Efficient Gradient Boosting Decision Tree") has **no verified DOI in Crossref**. Current ref [10] cites a conference proceeding (ICCSE 2019) which is not the primary citation.
**Action:** Use `10.5555/3294771.3294832` (NIPS 2017 proceedings) or arXiv `10.48550/arXiv.1611.08797` (preprint). Verify via Crossref.

### 7. **Mordred DOI Incorrect**
**Location:** Reference [11]
**Issue:** Manuscript cites `10.1186/s13321-020-00455-y` but Crossref returns **no such DOI**. The correct Mordred paper is Moriwaki et al., J. Cheminform. 2018, `10.1186/s13321-018-0258-y`.
**Action:** Fix DOI and year (2018, not 2020).

### 8. **XGBoost Reference Year/DOI Mismatch**
**Location:** Reference [9]
**Issue:** Cites Chen & Guestrin, KDD 2016, DOI `10.1145/2939672.2939785` — this is correct for the *conference paper*, but the canonical citation is often the JMLR 2016 version. Verify which JPA prefers.

### 9. **Methods: Missing TabPFN Version/Pinning Detail**
**Location:** Section 2.3 (Route 9)
**Issue:** "TabPFN v2" mentioned but exact version not pinned. TabPFN v2 API changed between releases.
**Action:** Add `tabpfn==0.8.5.0` (or whatever `pip show tabpfn` shows) and `ModelVersion.V2, ignore_limits=True`.

### 10. **Methods: KPGT GPU Port Reproducibility**
**Location:** Section 2.3, Data Availability
**Issue:** GPU port checkpoint directory `C:/Users/c00jsw00/_kpgt_ckpt_gpu/` is **local Windows path**, not a shareable artifact.
**Action:** 
- Upload checkpoints to Zenodo/Figshare with DOI
- Or provide retraining script that reproduces them exactly (with seed)
- Update Data Availability statement

### 11. **Discussion: "6,834 Shared SMILES" — How Computed?**
**Location:** Section 3.3, first paragraph
**Issue:** Number stated but not derived. Need: `len(set(our_7283) & set(pv_6869)) = 6,834` with code reference.
**Action:** Add one sentence: "Shared SMILES computed via `common.py:split_smiles` intersection (script: `peptiverse_experiment.py:L45`)."

### 12. **Ceiling Calculation: Formula Not in Manuscript**
**Location:** Section 3.1, Discussion 4.1
**Issue:** Ceiling R² = 0.5387 is central but formula only in `r2_ceiling.py` (not in manuscript).
**Action:** Add equation to Methods or Supplementary:  
`R²_ceiling = 1 - (SS_floor / SS_total)` where SS_floor = sum of squared deviations of floor rows from their test mean.

### 13. **Highlights: Character Count Exceeds 85**
**Location:** Highlights section
**Issue:** JPA requires "3–5 highlights, each ≤85 characters".
**Current:**
- "First systematic benchmark of nine improvement routes on PAMPA permeability using a censored-floor-aware protocol" → **102 chars** ❌
- "Foundation models (TabPFN v2, KPGT) break the LightGBM baseline (R² 0.464 → 0.513) but hit a censored ceiling at R² 0.539" → **118 chars** ❌
- "KPGT fine-tuning reaches 95% of the ceiling yet sacrifices non-floor regression (R² 0.632 → 0.536)" → **101 chars** ❌
- "Cross-dataset validation on 6,834 shared SMILES shows KPGT fine-tune (ρ=0.811) outperforms PeptiVerse frozen ChemBERTa (ρ=0.770)" → **122 chars** ❌
- "Platform enables cyclic peptide lead prioritization, benchmarking, and foundation model fine-tuning for censored ADMET endpoints" → **109 chars** ❌
**Action:** Rewrite all 5 to ≤85 chars.

### 14. **Abstract: Structured Format Missing "Purpose" Label**
**Location:** Abstract
**Issue:** JPA requires **Purpose / Methods / Results / Conclusion** as explicit bold labels. Current abstract uses "Objective / Methods / Results / Conclusions" — close but not exact.
**Action:** Match JPA labels exactly: **Purpose**, **Methods**, **Results**, **Conclusion**.

### 15. **Missing: AI/ML Declaration**
**Location:** After Conclusions (or Cover Letter)
**Issue:** JPA (and Elsevier) requires: "Declaration of Generative AI and AI-assisted technologies in the writing process"
**Action:** Add standard Elsevier AI declaration statement.

---

## Positive Observations (Strengths)

| Aspect | Assessment |
|---|---|
| **Honesty** | Exceptional: ceiling quantification, negative results (Routes 1–8), trade-off admission, no cherry-picking |
| **Reproducibility** | Seeds, splits, hardware, checkpoints, verification scripts all provided; KPGT GPU port numerically validated (max diff 8.3e-7) |
| **Statistical Rigor** | 3-seed mean ± SD, early stopping on val, best-val checkpoint → test, non-floor evaluated separately |
| **Fair Comparison** | Same split (common.split_smiles seed=42), same data, same metrics for PeptiVerse comparison |
| **Domain Insight** | Censored floor (3.7% rows, 49.6% SS) correctly identified as fundamental limit |
| **Code Availability** | Full pipeline at GitHub (MIT), feature caches, reproduction scripts |

---

## Specific Line-Level Corrections

| Page/Line | Current | Corrected |
|---|---|---|
| Table 1, Route 4 | "Two-stage (floor classifier + regressor)" R² = -1.21 | Verify: two-stage R² from `floor_predictability.py` was **negative**? If so, keep but explain. |
| Table 1, Route 5 | "Soft blend" R² = 0.4651 | Verify against `soft_blend.py` output |
| Table 1, Route 8 | "Label averaging" Δ = −0.015 | ✓ Matches `label_avg_results.json` (PAMPA: −0.0152) |
| Sec 3.2 | "TabPFN v2: R² = 0.496 ± 0.002" | ✓ Matches JSON (0.4962 ± 0.0016) |
| Sec 3.2 | "KPGT: R² = 0.513 ± 0.005" | ✓ Matches JSON (0.5134 ± 0.0048) |
| Sec 3.3 | "KPGT ρ = 0.811" | Seed 42 = 0.8257; mean of 3 seeds = 0.8107 → **use 0.811** ✓ |
| Sec 3.3 | "PeptiVerse ρ = 0.770" | E2_chem_2d mean = 0.7696 → **use 0.770** ✓ |
| Sec 4.1 | "Ceiling 0.5387" | ✓ From `r2_ceiling.py` |
| Sec 4.2 | "Non-floor R² drops from 0.632 to 0.536" | ✓ Baseline 0.6317, KPGT 0.5357 |
| Ref [2] | pepADMET: Tan et al., J. Chem. Inf. Model. 2026 | ✓ DOI `10.1021/acs.jcim.5c02518` verified |
| Ref [3] | PeptiVerse: Zhang et al., Nat. Commun. 2026 | ✓ DOI `10.1038/s41467-026-74167-w` verified |
| Ref [4] | TabPFN: Hollmann et al., Nature 2025 | ✓ DOI `10.1038/s41586-024-08328-6` verified |
| Ref [5] | KPGT: Li et al., arXiv 2022 | ✓ DOI `10.48550/arXiv.2206.03364` verified |

---

## Final Checklist for Revision

- [ ] **Run Routes 1–5 scripts** → generate `analysis/route{1-5}_results.json` → update Table 1
- [ ] **Clarify PeptiVerse comparison scope** (frozen only) in Table 3 caption + text
- [ ] **Move non-floor trade-off to Abstract + Highlights**
- [ ] **Add pepADMET overlap comparison** or justify omission
- [ ] **Create real Graphical Abstract** (PNG/SVG)
- [ ] **Fix LightGBM reference** to primary NIPS 2017 paper
- [ ] **Fix Mordred DOI** to `10.1186/s13321-018-0258-y` (2018)
- [ ] **Pin TabPFN version** in Methods
- [ ] **Host KPGT checkpoints** on Zenodo/Figshare + update Data Availability
- [ ] **Add shared SMILES computation reference**
- [ ] **Add ceiling formula** to Methods/Supplementary
- [ ] **Rewrite all 5 Highlights** to ≤85 characters
- [ ] **Rename Abstract sections** to Purpose/Methods/Results/Conclusion
- [ ] **Add Elsevier AI Declaration**
- [ ] **Verify all 14 reference DOIs** via Crossref one final time

---

## Estimated Revision Effort

| Task | Effort |
|---|---|
| Run 5 route scripts + verify | 2–4 hrs (if scripts run cleanly) |
| Graphical abstract creation | 1–2 hrs |
| Text revisions (Abstract, Highlights, Discussion) | 1 hr |
| Reference fixes | 0.5 hr |
| **Total** | **~5–8 hours** |

---

**Bottom Line:** This is a **strong, honest, reproducible ML-for-science paper** that deserves publication after the Major issues (especially Route 1–5 verification and PeptiVerse comparison scope) are resolved. The negative results and trade-off analysis are more valuable than yet another "SOTA+1%" paper.

---

*Report generated by automated reviewer simulation using grounded citations and direct JSON log verification.*