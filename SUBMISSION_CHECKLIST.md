# JCIM Submission Checklist

## Manuscript Preparation

### ✅ Completed Items

- [x] Manuscript formatted according to JCIM guidelines
- [x] Abstract with keywords (8-10 keywords)
- [x] Graphical Abstract design specifications
- [x] Table of Contents (TOC) Graphic design specifications
- [x] Cover Letter prepared
- [x] References properly formatted (17 references)
- [x] Supporting Information section included

### 📄 Required Files

1. **Manuscript**: `peptide_admet_manuscript_jcim.md`
   - 23,592 bytes
   - Full-length research article
   - 6 sections + references

2. **Cover Letter**: `cover_letter_jcim.md`
   - 5,869 bytes
   - Addressed to Editor-in-Chief
   - Includes reviewer suggestions

3. **Graphical Abstract**: `graphical_abstract.png`
   - 4000x2400 px
   - RGB color mode
   - 300 DPI minimum
   - [Specs: graphical_abstract_spec.md]

4. **TOC Graphic**: `toc_graphic.png`
   - 1100x400 px
   - RGB color mode
   - 300 DPI minimum
   - [Specs: graphical_abstract_spec.md]

5. **Supporting Information**: Available at GitHub
   - Training data statistics
   - Feature correlation analysis
   - Model code and weights

## Code Repository

### ✅ GitHub Repository: `openclaw-peptide-admet-jcim`

**Location**: `/home/c00jsw00/.openclaw/workspace/openclaw-peptide-admet-jcim/`

**Contents**:
- [x] `peptide_admet_model.py` - Training framework
- [x] `peptide_admet_inference.py` - Inference tool
- [x] `train_model.py` - Training script
- [x] `README.md` - Project documentation
- [x] `USAGE_GUIDE.md` - Usage instructions
- [x] `requirements.txt` - Python dependencies
- [x] `LICENSE` - MIT License
- [x] `real_peptide_data/` - Training dataset
  - `real_peptide_admet_data.csv` (15,000 samples)
  - `X_train.npy`, `X_val.npy`, `X_test.npy`
  - `y_train.npy`, `y_val.npy`, `y_test.npy`

## Submission Requirements

### ACS Submission Portal

**URL**: https://pubs.acs.org/page/4authors/submission/index.html

### Required Information

1. **Manuscript Type**: Articles (Full-Length Research)
2. **Section**: Computational Chemistry / Cheminformatics
3. **Corresponding Author**: Pinwan (品丸)
4. **Contact Info**: [Your email], [Your phone]
5. **Total Pages**: ~25-30 pages (including references)
6. **Figures**: 0 (will upload separately)
7. **Tables**: 2-3 tables
8. **References**: 17 references

### Additional Requirements

- [ ] **Potential Reviewers**: Provide 3-5 names
  - Prof. John A. Tuszynski (University of Alberta)
  - Dr. Stephen E. Boyce (Scripps Research)
  - Prof. David Huang (UC San Diego)

- [ ] **Graphical Abstract**: Create PNG file (4000x2400 px)
  - Use design specs in `graphical_abstract_spec.md`

- [ ] **TOC Graphic**: Create PNG file (1100x400 px)
  - Use design specs in `graphical_abstract_spec.md`

- [ ] **Data Availability Statement**: 
  - GitHub: https://github.com/c00jsw00/openclaw-peptide-admet-jcim

- [ ] **Conflict of Interest**: None to declare

- [ ] **Funding Statement**: OpenClaw computational resources

## Before Submission

### Final Checks

1. [ ] Customize cover letter with your contact information
2. [ ] Create graphical abstract (4000x2400 px PNG)
3. [ ] Create TOC graphic (1100x400 px PNG)
4. [ ] Verify GitHub repository is accessible
5. [ ] Check all files are properly formatted
6. [ ] Proofread manuscript for typos
7. [ ] Verify reference formatting
8. [ ] Confirm author list and affiliations

### Submission Steps

1. Create ACS ID (if not already have)
2. Login to submission portal
3. Select "New Manuscript"
4. Choose manuscript type: "Article"
5. Enter manuscript title
6. Upload manuscript file
7. Upload graphical abstract
8. Upload TOC graphic
9. Enter author information
10. Enter corresponding author details
11. Enter potential reviewers (3-5 names)
12. Upload Supporting Information (if separate)
13. Answer copyright question
14. Review and submit

## After Submission

### Expected Timeline

- **Initial Screening**: 1-3 days
- **Editor Assignment**: 3-7 days
- **Peer Review**: 4-8 weeks
- **Decision**: Varies

### What to Expect

1. **Initial Check**: Editorial office checks format and completeness
2. **Editor Assessment**: Editor-in-Chief determines scope fit
3. **Peer Review**: Sent to 2-3 reviewers
4. **Decision**: Accept, Minor Revision, Major Revision, or Reject

### Revision Process (if needed)

- **Minor Revision**: 2-4 weeks to revise
- **Major Revision**: 4-8 weeks to revise
- **Resubmission**: Upload revised manuscript + response letter

## Contact Information

**Editor-in-Chief**: Prof. Kenneth M. Merz, Jr.  
**Email**: merz-office@jcim.acs.org  
**Phone**: 517-355-9715  
**Fax**: 517-353-7248  

**ACS Support**: https://pubs.acs.org/page/4authors/support.html

---

## Files Summary

| File | Size | Description |
|------|------|-------------|
| `peptide_admet_manuscript_jcim.md` | 23,592 B | Full research manuscript |
| `cover_letter_jcim.md` | 5,869 B | Submission cover letter |
| `graphical_abstract_spec.md` | 6,222 B | Graphic design specifications |
| `README.md` | 8,567 B | Project documentation |
| `USAGE_GUIDE.md` | 9,600 B | User instructions |
| `peptide_admet_inference.py` | 10,621 B | Inference script |
| `peptide_admet_model.py` | 42,410 B | Training framework |
| `train_model.py` | 3,756 B | Training script |
| `real_peptide_admet_data.csv` | ~1.5 MB | Training dataset |
| `.npy` files | ~5 MB | Preprocessed data |

**Total Repository Size**: ~10 MB

---

*Last Updated: 2026-03-24*  
*Status: Ready for submission pending graphical abstract creation*
