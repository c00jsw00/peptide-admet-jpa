# Cover Letter for Manuscript Submission to Journal of Chemical Information and Modeling (JCIM)

---

**[Date: August 24, 2026]**

**To:**
Editorial Office
Journal of Chemical Information and Modeling (JCIM)
American Chemical Society

**Subject:** Submission of Article (Benchmark / Methods): "An Honest, Reproducible Benchmark for Peptide ADMET Prediction: Homology-Controlled Evaluation of a Multi-Task Neural Network on a Regenerable Synthetic Demo Set"

Dear Editor,

We are pleased to submit our article entitled **"An Honest, Reproducible Benchmark for Peptide ADMET Prediction: Homology-Controlled Evaluation of a Multi-Task Neural Network on a Regenerable Synthetic Demo Set"** for consideration in the *Journal of Chemical Information and Modeling (JCIM)*.

## Research Significance

Peptide therapeutics represent one of the fastest-growing pharmaceutical classes. Yet the 2026 benchmarking literature (in particular AMPBench-MT, arXiv:2607.25518) has exposed a systematic evaluation-integrity problem in this field: sequence-similarity leakage between train and test sets inflates reported performance, and several public tools report metrics that cannot be reproduced from their released artifacts. We address this problem methodologically, with a small but fully reproducible reference implementation.

## Key Contributions

1. **A regenerable, provenance-labeled demo dataset.** 15,000 synthetic peptides with a fixed seed, an explicit `data_origin` column, and a metadata file describing the exact label model. No shipped artifact can ever again be silently missing or fabricated.

2. **An AMPBench-MT-style homology-controlled split with a shipped leakage audit.** Families are assigned whole to train/val/test (70/10/20); we compute and release the maximum pairwise composition-Jaccard across splits and per-endpoint label-rate deltas (in our run: max Jaccard 0.250, all label-rate deltas ≤ 0.013).

3. **A shared multi-task PyTorch MLP released end-to-end.** One model class is defined in a single file and imported by both trainer and predictor. The exact weights (144,133 parameters), scaler, and measured per-endpoint metrics are in the repository; the inference CLI prints only values it reads from `metrics.json`.

4. **Measured, un-inflated results.** Macro AUC 0.8684 / mean accuracy 0.7836 on the homology-controlled split, versus 0.8688 on a random split. We report the near-zero delta as our central finding and explain why 0.8–0.9 AUC is the expected ceiling for a latent-physicochemical label model with composition-level features.

5. **A multi-objective composite score** (geometric mean over favorable endpoint probabilities) for candidate prioritization, in the spirit of generative AMP campaigns such as AMPGAN v3 / PepCraft (arXiv, 2026-06).

## Alignment with JCIM Scope

- **Machine learning on chemical and biological data**: multi-task peptide property prediction.
- **Computational methods and benchmarking**: a leakage-audited split protocol and a measured-only reporting template for the field.
- **Reproducibility**: every number in the manuscript is re-derivable from the released code on CPU.

## Novelty and Impact

Unlike prior peptide ADMET tools (AdmetSAR 2.0, SwissADME, ADMETlab 3.0, pepADMET), which report performance on non-external or unevaluated splits, our contribution is a *protocol and a template*: audited homology-controlled splitting, dual-split reporting, and measured-only inference. We deliberately position the article as a benchmark/methods piece and make no real-peptide accuracy claim.

## Data and Code Availability

All code, the trained model, the scaler, the leakage audit, and `metrics.json` are available at:
**https://github.com/c00jsw00/openclaw-peptide-admet**

The pipeline regenerates end-to-end: `prepare_data.py → homology_split.py → train_peptide_admet_model.py → peptide_admet_predictor.py`.

## Declaration of Originality

This manuscript has not been published and is not under consideration elsewhere. All authors have read and approved the final version. We have no conflicts of interest to declare.

## Contact Information

**Corresponding Author**:
Pinwan (品丸)
OpenClaw Team
Email: [your contact email]

**Manuscript Title**: An Honest, Reproducible Benchmark for Peptide ADMET Prediction: Homology-Controlled Evaluation of a Multi-Task Neural Network on a Regenerable Synthetic Demo Set

---

Thank you for considering our manuscript for publication in *JCIM*. We look forward to your response.

Sincerely,

**Pinwan (品丸)**
OpenClaw Team
[Date: August 24, 2026]

---

*Note: Please customize the bracketed contact information before submission. The v1.0 cover letter (2026-03-24) is superseded by this revision.*
