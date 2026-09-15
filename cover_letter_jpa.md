# Cover Letter for Manuscript Submission to the Journal of Pharmaceutical Analysis (JPA)

---

**[Date: September 15, 2026]**

**To:**
Editorial Office
Journal of Pharmaceutical Analysis (JPA)

**Subject:** Submission of Original Research Article: "Peptide ADMET Prediction: A Systematic Benchmark and Foundation Model Evaluation"

Dear Editor,

We are pleased to submit our original research article entitled **"Peptide ADMET Prediction: A Systematic Benchmark and Foundation Model Evaluation"** for consideration in the *Journal of Pharmaceutical Analysis*.

## Research Significance

Oral absorption remains a principal bottleneck in peptide drug discovery, and machine-learning permeability prediction is increasingly used for lead prioritisation in early pharmaceutical development. Yet published cyclic-peptide permeability models (R² values of 0.67–0.77) have, to our knowledge, never been evaluated under a protocol that simultaneously (i) removes structural near-duplicates from the test set and (ii) treats the left-censored assay floor explicitly. Both effects inflate reported R²: near-duplicate leakage lets models memorise, and the repeated floor value dominates the variance budget of censored data.

## Key Contributions

1. **A leakage-controlled, censored-floor-aware benchmark protocol.** We partition the pepADMET collection (7,283 PAMPA and 7,429 Caco-2 cyclic peptides) by unique canonical SMILES (70/10/20, seed 42) so that no tautomer or stereoisomer appears in more than one split, and we derive explicit oracle ceilings from the censored variance (R² = 0.539 PAMPA; 0.570 Caco-2) that bound what any model can achieve on the current data.

2. **Ten systematic improvement routes.** Eight classical routes (descriptor expansion, rank-Gaussian target transformation, LightGBM ensembling, Tobit censored regression, soft-label blending, ChemBERTa frozen embeddings, PeptiVerse raw-data cross-validation, label averaging) and two foundation-model routes (TabPFN v2; KPGT fine-tuning), extended to the Caco-2 endpoint as route 10. All classical routes fail to exceed the MLP baseline (R² = 0.464); the foundation models break the barrier on both endpoints.

3. **First foundation-model results under an honest protocol.** TabPFN v2 reaches R² = 0.496 ± 0.002 on PAMPA; KPGT fine-tuning reaches R² = 0.513 ± 0.005 — the best of all routes and 95 % of the theoretical censored ceiling. On Caco-2 the ranking reverses (TabPFN 0.442 ± 0.003 vs KPGT 0.411 ± 0.006), and on 6,834 shared molecules our fine-tuned KPGT (Spearman ρ = 0.811) exceeds PeptiVerse frozen embeddings (ρ = 0.671) by +0.14.

4. **A reproducible GPU implementation of KPGT.** Because the official DGL implementation is CPU-only on our platform, we developed a pure-PyTorch port (scatter-based TripletTransformer) verified against DGL to a maximum absolute difference of 8.3 × 10⁻⁷.

5. **Contextualisation against the wider literature (Section 3.6, Table 5).** We compare our results with six recently published task-specific cyclic-peptide permeability models (CycPeptMP, Multi_CycGT, CPMP, a 13-method DMPNN benchmark, PCPpred, C2PO) and show that the gap between their headline R² and ours is best explained by split protocol (duplicate leakage + ignored censoring) rather than model inadequacy — a finding the 13-method scaffold-split study independently supports.

## Alignment with JPA Scope

- **Pharmacokinetics / ADME modelling:** quantitative permeability prediction (PAMPA, Caco-2) for oral peptide delivery.
- **Computational methods for pharmaceutical analysis:** a censored-floor-aware evaluation protocol applicable to any left-censored ADME assay.
- **Data integrity:** every reported number is re-derivable from the released code, splits, and checkpoints; all references carry verified DOIs.

## Novelty and Impact

We are the first, to our knowledge, to quantify the censored ceiling for peptide permeability prediction and to show that published high R² values are substantially protocol artefacts. Our central message is deliberately honest: foundation models improve the baselines, but the censored ceilings — not model capacity — are the true bottleneck. Progress beyond these bounds requires uncensored re-measurement of floor compounds, an experimental rather than algorithmic task, which we identify as the next step for the field.

## Data and Code Availability

All data splits, trained checkpoints, feature pipelines, and reproduction scripts are available under an MIT licence at:
**https://github.com/c00jsw00/openclaw-peptide-admet**

## Declaration of Originality

This manuscript has not been published and is not under consideration elsewhere. All authors have read and approved the final version. We declare no conflicts of interest. The manuscript was prepared with the assistance of AI tools for code generation, literature retrieval, and text drafting; all scientific claims, numbers, and conclusions were verified by the authors against primary computational experiments.

## Contact Information

**Corresponding Author:**
Pinwan (品丸)
[Affiliation]
Email: [your contact email]

**Manuscript Title:** Peptide ADMET Prediction: A Systematic Benchmark and Foundation Model Evaluation

---

Thank you for considering our manuscript for publication in the *Journal of Pharmaceutical Analysis*. We look forward to your response.

Sincerely,

**Pinwan (品丸)**
[Date: September 15, 2026]

---

*Note: Replace the bracketed affiliation/email before submission. The JCIM cover letter (cover_letter_jcim.md) is superseded by this JPA revision.*
