# Cohort and fold membership

These files accompany the adult-initialization study (DOI 10.3389/fneur.2026.2009806). The unit is an MRI examination. Identifiers are the de-identified identifiers supplied in BraTS-PEDs and BraTS 2021.

| Outer fold | Held-out source | Development examinations | Held-out examinations |
| --- | --- | ---: | ---: |
| 0 | CBTN | 141 | 116 |
| 1 | DFCI-BCH-BWH-PEDs-HGG (DFCI) | 196 | 61 |
| 2 | DIPGr | 177 | 80 |

- `pediatric_outer_folds.csv` gives a row for each of the 257 examinations with a reference segmentation. Its fold number identifies the run in which the examination was held out. Both initialization conditions used the same split for seeds 2026, 2027 and 2028. Case ID, fold and source were checked against all 18 original evaluation tables.
- `splits_final.json` is the original nnU-Net outer-fold file. List positions correspond to folds 0, 1 and 2. Within each fold, `train` contains the two development sources and `val` contains the held-out source. Its SHA-256 recorded at the training site is `65d1f58b0fb241f5f9a67790a2afa4bf9ee9511bc971458517e5aa7335fc52e7`.
- `formal_splits_v1.json` contains pediatric examination and adult pretraining case IDs (257 annotated pediatric examinations, 1,251 adult pretraining examinations), outer and nested source splits, and the separate 91-case pediatric blind validation list. The reported source-held-out comparison uses the 257 annotated examinations. The file matches the historical fingerprint `be1cb7a7f3ae341051065406731d139d8309a5040484815c963dbb21e2e16123`. The training-site copy was recovered by restoring its recorded input-manifest checksum in the local copy.
- `p1_internal_splits.json` contains the fitting and calibration cases used in the auxiliary P1 experiment, recorded for each fold and seed. Each pair partitions that outer fold’s development examinations. These internal splits differ from the nested source splits in the frozen protocol.

Pediatric data and annotations are available from [The Cancer Imaging Archive](https://doi.org/10.7937/DX5C-TJ86). Adult MRI data are from the [BraTS 2021 challenge](https://arxiv.org/abs/2107.02314v2). Follow the respective providers’ access and reuse conditions.
