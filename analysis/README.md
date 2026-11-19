# PedGlioNet evaluation and figures

This directory reproduces the accepted manuscript’s statistical analyses from released case-level segmentation scores and uncertainty summaries. It contains author code, derived numeric data, public BraTS-PEDs examination identifiers, and figure sources. The three folds hold out CBTN, DFCI-BCH-BWH-PEDs-HGG and DIPGr, respectively. A row in a metrics table identifies one examination and one region. The seed and training method are in the filename.

## Run locally

Use Python 3.9–3.12. From the repository root:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r analysis/requirements.txt
python analysis/run.py
```

The last command requires only the included files once the dependencies are installed. It checks case coverage, reference consistency, empty-case Dice, and seed pairing. It calculates single-seed and seed-averaged contrasts, stratified intervals (2,000 bootstrap replicates), the positive/absence decomposition and development-planning sensitivity. It also verifies the 257 frozen omission records against per-seed metrics. Results go to `analysis/output/`.

```sh
python analysis/run.py --figures --reliability
```

This also draws Figures 2–9, S1 and S2, and repeats discrimination, risk–coverage and omission analyses under both historical and development-source-only planning. For the changed plans, folds 0 and 2 use the alternate records and fold 1 reuses its historical plan, as specified in the study. `--figures` and `--reliability` also work independently.

## Expected results

| Readout | Expected value |
|---|---|
| ET-positive paired three-seed mean, 171 examinations | 0.5897 pediatric-only vs 0.6086 adult initialization, delta 0.0189 (95% interval 0.0054–0.0322) |
| ET-positive paired effect, additional seeds 2027–2028 | 0.0172 (95% interval 0.0007–0.0327) |
| Historical pediatric probability-ensemble Dice | WT 0.863523, TC 0.833249 |
| Disagreement AUROC, Dice < 0.5 | 0.991837 for WT, 0.541485 for reference-positive ET |
| Reference-positive ET cases omitted by all three pediatric seeds | 21 of 171, including 11 with ensemble WT Dice ≥ 0.90 |
| Exploratory PCoA, first and second axes | 61.53% and 32.45% |

The software averages the three single-seed Dice scores within each examination for the paired transfer analysis. The ensemble analysis instead scores the segmentation obtained from averaged voxel probabilities.

## Inputs and figure coverage

| Material | How to reproduce it |
|---|---|
| Native-space Dice, HD95 in mm, NSD at 1 mm and lesion measures | `scripts/evaluate_masks.py` reads local reference and prediction NIfTIs, checks geometry, and writes the case-level CSV and an aggregate report. The published numeric CSVs are in `data/eval/`. |
| Figure 1 | `editable_figures/Figure1_framework.svg` is the original study schematic. Its image panels are synthetic illustrations. |
| Figure 2 | `figure02.py` redraws the regional distributions, means and intervals from archived case points and summaries in `figure_data/Figure2.json`. The ensemble input tables also permit independent calculation of these metrics. |
| Figures 3–5 | `reproduce.py` computes paired gains, source-by-seed results, lesion, volume and absence measures from the 18 single-seed evaluation tables. `figures03_05.py` and `figure04.py` draw those results. |
| Figure 6 | `figure06.py` computes ROC and risk–coverage curves from ensemble Dice and case-level uncertainty. The full reliability analysis recomputes the intervals and the source-specific results. |
| Figure 7 | `planning_metrics.py` recomputes paired historical/development-only ensemble differences and stratified intervals. `figures07_s1.py` plots those results. |
| Figures 8–9 | `figures08_09.py` uses the verified omission membership records and archived AUROC intervals. `verify_results.py` checks per-seed ET presence and Dice and ensemble WT/ET Dice against the supplied evaluation tables. The full reliability analysis regenerates the intervals. |
| Figure 10 | `figure10_authorized.py` performs the original case selection and draws the matched MRI crops only with separately obtained image data and predictions. See below. |
| Supplementary Figure S1 | `figures07_s1.py` draws the corrected native-space P1 estimates and intervals from frozen aggregate results. The corrected case-level P1 CSV is not in this directory. After generating it with `../replication/`, `scripts/p1_statistics.py --cases FILE --output FILE` recalculates the numerical records. The earlier crop-space CSV is unsuitable for this calculation. |
| Supplementary Figure S2 | `figureS2.py` recomputes pooled standardization, the distance matrix, principal coordinates and source contours from the three ensemble evaluation tables; it writes case coordinates and explained variance. |

The figure scripts listed in the table are under `scripts/`. `editable_figures/` contains accepted-version artwork for the schematic and quantitative figures. The plotting scripts use Matplotlib’s bundled DejaVu Serif for portability. The final proof axis-label additions for Figures 2–5 and 7 are in `../publication_figures/`; its script operates on the accepted artwork.

## Working with separately authorized imaging

Obtain BraTS-PEDs access under the terms of its data provider. The numeric tables are sufficient for `run.py`. To score new predictions, prepare a local CSV with columns `case_id,source,seg`, where `seg` points to each reference NIfTI. Place the prediction for each validation case in a folder as `<case_id>.nii.gz`. For example, from the repository root:

```sh
python analysis/scripts/evaluate_masks.py \
  --manifest private/manifest.csv --prediction-dir private/fold0_predictions \
  --splits analysis/data/folds.json --fold 0 \
  --output analysis/output/fold0_summary.json \
  --per-case-output analysis/output/fold0_per_case.csv
```

For Figure 10, set `PEDGLIO_BRATS_TRAINING` to the authorized Training directory with `<case_id>/<case_id>-t1c.nii.gz` and `<case_id>/<case_id>-seg.nii.gz`. Set `PEDGLIO_PREDICTIONS` to a directory containing `{b0,adult_init}/fold<F>_seed<S>/<case_id>.nii.gz` for the selected cases and all three seeds. Run `python analysis/scripts/figure10_authorized.py`. The program checks 18 historical prediction file hashes, image geometry, region volumes and Dice before rendering. Independently regenerated or recompressed files may have different hashes. The numeric analyses above require neither those files nor model weights.

## Data and validation notes

`data/folds.json` contains validation membership and held-out source. `../cohort/` carries the training/evaluation split records. `data/eval_source_planned/` and `data/uncertainty_source_planned/` contain the available changed-fold records. `figure_data/` contains the plotted numeric inputs, including the reported P1 aggregate and archived checks for reliability. Input tables use public case identifiers and numeric metrics, without image voxels or segmentation masks.

Release checks reproduced all six sections of the archived paired-transfer analysis, all 120 gain-decomposition records, the four reported planning contrasts and the historical discrimination, risk–coverage and omission results. Development-planning discrimination and risk–coverage results were independently compared with their frozen archive. Figure 10’s conditional rendering requires the restricted imaging inputs.
