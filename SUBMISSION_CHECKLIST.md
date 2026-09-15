# JPA Submission Checklist (Journal of Pharmaceutical Analysis)

*Last updated: 2026-09-15 · Status: Ready pending author contact details*

## Manuscript

| Item | Status |
|------|--------|
| Structured abstract (Purpose / Methods / Results / Conclusion) | ✅ |
| Highlights (5 bullets, each ≤ 85 chars) | ✅ |
| Graphical abstract (`graphical_abstract.png`, 7644×4080 RGB, 600 DPI) | ✅ rendered 2026-09-15 |
| Section order: Introduction → Materials and Methods → Results → Discussion → Conclusions | ✅ |
| Tables 1–5 (incl. §3.6 literature comparison, Table 5) | ✅ |
| 18 references, numbered by order of first appearance, all DOIs verified via doi.org + Crossref | ✅ (2026-09-15) |
| Data Availability / AI Declaration / Conflict of Interest statements | ✅ |
| Caco-2 baseline harmonised (0.393 everywhere) | ✅ |
| Route count consistent (ten routes; PAMPA 1–9 + Caco-2 route 10) | ✅ |

## Files (this repository, `peptide-admet-jpa`)

| File | Purpose |
|------|---------|
| `peptide_admet_manuscript_jcim.md` | Full manuscript (filename retained for history) |
| `cover_letter_jpa.md` | JPA cover letter (supersedes `cover_letter_jcim.md`) |
| `graphical_abstract.png` | Graphical abstract, ready to upload |
| `analysis/make_graphical_abstract.py` | Re-rendering script |
| `SUBMISSION_CHECKLIST.md` | This checklist |

## Before Submission — remaining author tasks

- [ ] Cover letter: fill affiliation + corresponding-author email (bracketed placeholders)
- [ ] Cover letter: set submission date
- [ ] Confirm final author list and affiliations
- [ ] (Optional) JPA requires double-blind: remove author names from the manuscript main file if the journal's submission system does not strip them

## Code / Data

- Repository: https://github.com/c00jsw00/openclaw-peptide-admet (MIT licence)
- All splits, checkpoints, feature pipelines, and reproduction scripts included
- GPU KPGT port: `analysis/_kpgt_finetune_gpu.py` (verified against DGL, max |Δ| = 8.3 × 10⁻⁷)

## Git state (both repositories in sync)

- `openclaw-peptide-admet` master: `bed292e`
- `peptide-admet-jpa` main: (this commit)
