# Replicating the trained models in manuscript 2009806

The accepted v15 manuscript used a standard nnU-Net v2 residual encoder (ResEnc M). The primary comparison matched a pediatric-only baseline (B0) to pediatric fine-tuning from one fixed adult checkpoint (AdultInit). Both used the same pediatric data, seeds, geometry, schedule and held-out source. The auxiliary P1 model is a separate network.

## What is included

| Files | Role |
| --- | --- |
| `scripts/prepare_brats.py`, `build_formal_splits.py`, `prepare_nnunet_peds.py`, `prepare_nnunet_adult.py` | Build local manifests and splits and convert both datasets to nnU-Net conventions. |
| `scripts/run_nnunet_seeded.py`, `run_nnunet_formal.sh`, `nnunet_env.sh` | Original training, adult initialization and held-out validation. |
| `scripts/run_adult_transfer_replication.py` | Historical worker for the six additional fine-tuning runs with seeds 2027 and 2028. |
| `scripts/ensemble_uncertainty.py`, `evaluate_formal_segmentation.py` | Cross-seed B0 probability averaging, native-space region measurements and quality analyses. |
| `scripts/audit_source_only_planning.py`, `run_source_planning_sensitivity.py` | Reconstruct development-only planning and train the two changed folds. |
| `scripts/run_p1_all_v2.sh`, `joint_et_refiner.py`, `joint_et_infer.py`, `p1_outer_eval_corrected.py` and imported local modules | Auxiliary two-stage P1 training and corrected held-out evaluation. |
| `configs/Dataset501_BraTSPEDs/` | Dataset schema and a local pediatric ResEnc M reference plan with the reported geometry. The plan contains aggregate normalization/geometry settings. |
| `configs/formal_experiment_v1.json`, `FORMAL_EXPERIMENT_PROTOCOL_v1.md` | Early planning documents. Their SHA-256 values match the original fold definition. The final analysis scope, conditions and interpretations are specified by the accepted v15 manuscript. |
| `external_trainers/pedglio_trainers.py` | Release helper that exposes the checkpoint trainer names to an independent nnU-Net prediction process. |

`SOURCE_MAP.tsv` records the original project-relative path and SHA-256 of every copied file. The external-trainer helper, requirements and this guide were written for the release. Study methods were checked against `output/revision_v15_20261023/editable_source/main_submission_en_v15.tex`. The adult preprocessing and launch examples below follow the original `scripts/cluster_setup.sh` and `scripts/run_cluster_queue.sh`. The original cluster setup is unnecessary for these commands.

## Environment

Run commands from this directory in Bash on a Linux GPU host. The manuscript reports nnU-Net 2.8.1, PyTorch 2.11.0+cu128, NumPy 2.2.6, SciPy 1.15.3, nibabel 5.4.2 and NVIDIA L20X hardware. The cluster runbook records CUDA 12.8 wheel selection and driver 570.133.20. Use a Python version compatible with these wheels, for example Python 3.11. The exact historical Python patch level and a complete cluster package lock could not be verified. Unpinned package versions may affect reproduction.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install 'torch==2.11.0+cu128' --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
source scripts/nnunet_env.sh
export nnUNet_extTrainer="$PWD/external_trainers"
```

## Prepare the authorized data

Obtain the BraTS-PEDs data and BraTS 2021 adult data from their authorized distributors. Supply these locally under `data/raw/brats_peds` and `data/raw/brats_adult_2021`, or provide the directory and metadata arguments to `prepare_brats.py`. The source scans recursively and uses the released file suffixes and pediatric metadata to identify subjects and sources. Regenerate the manifest on the machine holding the data because it contains machine-specific paths. Keep generated manifests, images, model outputs and run logs local.

```bash
python scripts/prepare_brats.py --datasets both --inspect-labels
cp ../cohort/formal_splits_v1.json data/processed/formal_splits_v1.json
python scripts/prepare_nnunet_peds.py --skip-blind
nnUNetv2_plan_and_preprocess -d 501 -pl nnUNetPlannerResEncM -c 3d_fullres -np 24 --verify_dataset_integrity
cp data/nnunet/splits_final_501.json "$nnUNet_preprocessed/Dataset501_BraTSPEDs/splits_final.json"

python scripts/prepare_nnunet_adult.py --workers 24
nnUNetv2_extract_fingerprint -d 502 -np 24 --verify_dataset_integrity
nnUNetv2_move_plans_between_datasets -s 501 -t 502 -sp nnUNetResEncUNetMPlans -tp nnUNetResEncUNetMPlansFrom501
nnUNetv2_preprocess -d 502 -plans_name nnUNetResEncUNetMPlansFrom501 -c 3d_fullres -np 24
```

The commands use the released frozen assignments. To rebuild them from a complete local manifest, run `python scripts/build_formal_splits.py` instead of copying the released file. The historical split-builder checks for 257 labeled and 91 unlabeled pediatric examinations. With the original formal split definition and a complete 257-case local training manifest, `prepare_nnunet_peds.py --skip-blind` can omit linking blind-validation scans. The 257-case findings use the labeled cohort only. Confirm the recreated assignments against the release's cohort records. Held-out folds 0, 1 and 2 correspond to CBTN (116), DFCI-BCH-BWH-PEDs-HGG (61) and DIPGr (80). The four input channels, in order, are T1, T1ce, T2 and FLAIR. The pediatric output schema is 0 background, 1 ET, 2 NET, 3 CC, 4 ED. Adult label 4 is mapped to 3 in the four-class adult output.

The locally available pediatric plan is a reference for inspecting geometry. Its file digest differs from the plan used on the historical training cluster (see the inventory below). For exact reconstruction, recover the original plan and check its full digest before use. The reported historical settings include 1-mm spacing, a 96 × 160 × 160 patch and batch size two. Historical geometric planning used the entire pediatric cohort before model fitting.

## Train and validate the primary models

The example pretrains once on the adult cohort and then trains each matched pediatric fold and seed. The official nnU-Net initialization utility loads compatible weights and excludes the output head. Each pediatric run performs native-space held-out validation and saves class probabilities; the adult pretraining run uses all adult cases and skips held-out validation.

```bash
python scripts/run_nnunet_seeded.py --dataset-folder Dataset502_BraTSAdult \
  --plans nnUNetResEncUNetMPlansFrom501 --fold all --seed 2026 \
  --trainer-suffix AdultPre --num-epochs 500 --skip-validation --data-processes 12

adult_checkpoint="$nnUNet_results/Dataset502_BraTSAdult/nnUNetTrainerPedGlioSeed2026AdultPre__nnUNetResEncUNetMPlansFrom501__3d_fullres/fold_all/checkpoint_final.pth"
for fold in 0 1 2; do
  for seed in 2026 2027 2028; do
    python scripts/run_nnunet_seeded.py --fold "$fold" --seed "$seed" \
      --num-epochs 1000 --save-probabilities --data-processes 12
    python scripts/run_nnunet_seeded.py --fold "$fold" --seed "$seed" \
      --trainer-suffix AdultInit --pretrained-weights "$adult_checkpoint" \
      --num-epochs 1000 --save-probabilities --data-processes 12
  done
  python scripts/ensemble_uncertainty.py --fold "$fold"
done
```

Use the original fixed adult checkpoint when matching the study's exact initialization. Training an adult model again replicates the recipe and can produce a different checkpoint. The adult effect in the paper averages per-case regional Dice across matched pediatric seeds. Averaging predictions first produces a different quantity. The B0 segmentation and quality analyses use the probability ensemble. GPU execution can retain nondeterminism even with the original seeds.

## Apply a trained model in a new process

A model directory must contain `plans.json`, `dataset.json` and `fold_N/checkpoint_final.pth` for the selected fold. Set `nnUNet_extTrainer` as above. The archived training entry point creates subclasses with checkpoint-specific names and inherits every nnU-Net model method. The release helper reproduces these names for standard inference. Provide your own local `input_images` in nnU-Net format with four files ending in `_0000.nii.gz` to `_0003.nii.gz` per examination.

```bash
nnUNetv2_predict -i input_images -o predictions/b0_fold0_seed2026 \
  -d 501 -c 3d_fullres -p nnUNetResEncUNetMPlans \
  -tr nnUNetTrainerPedGlioSeed2026 -f 0 --save_probabilities
```

Use the selected source's held-out fold for manuscript comparisons. Repeat with each seed-specific trainer when recreating the B0 ensemble. For adult-initialized models, append `AdultInit` to the trainer name. The defaults in nnU-Net 2.8.1 apply sliding windows with step 0.5, Gaussian weighting and mirrored inference. The B0 probability ensemble of the original held-out validation runs can be recreated using the previous section's ensemble command.

## Sensitivity and supplementary P1

For development-source-only geometry, first run `python scripts/audit_source_only_planning.py` after historical preprocessing to regenerate per-fold plans. Then run `python scripts/run_source_planning_sensitivity.py --fold 0 --gpu 0` and the same command with `--fold 2`. The historical DFCI models supply fold 1, whose material plan did not change. The two changed windows were 112 × 160 × 128. The planning worker uses the local `.venv/bin/python` and regenerates six training runs.

For the supplementary P1 experiment, `bash scripts/run_p1_all_v2.sh` trains the three pediatric seeds on each source split in two stages. The original scheduler uses three GPU lanes; set `LANE_GPUS` to the available GPU indices, taking concurrent memory use into account. The generic defaults in `joint_et_refiner.py` alone do not encode this final schedule. The archived corrected evaluator expects the three B0 ensemble masks and per-case evaluation CSVs, P1 run records, nine stage-B checkpoints, and the historical comparison CSV produced by `python scripts/p1_outer_eval.py`. After generating these, run `python scripts/p1_outer_eval_corrected.py` for the reported corrected analysis. The helper modules `et_presence_gate.py`, `et_small_target_refinement.py`, `full_transfer_train.py` and `pilot_transfer.py` provide imported implementation components.

## Verification, scope and licensing

This directory contains only text source and configuration. Captured checkpoints, private acquisition or storage paths, patient images, masks, per-case intensity records, credential files, and the third-party framework implementation are excluded. Data must be obtained under each dataset's terms. A full end-to-end CUDA training run and score reproduction were not performed during release preparation. The historical adult plan and development-only plans can be regenerated using the commands above. The pediatric plans have different file contents. Only the local reference file is included here. The archived cluster inventory records the historical plan's byte count and SHA-256. The available records do not establish which fields account for the difference.

| Pediatric ResEnc M plan | Bytes | SHA-256 |
| --- | ---: | --- |
| Included local reference | 11,779 | `61caca2d4a84ee630b7c3c289bf0d5d130232c2e1936ba9c4cee10256c7ce9d9` |
| Historical cluster file | 11,774 | `32fb8abbfedde74c37acdcb4864ada8776e16682e2a6171e9d76d6a7e0b2c423` |

Runtime validation of the independent prediction helper used the actual nnU-Net 2.8.1 trainer discovery on macOS with local PyTorch 2.13.0. All ten expected checkpoint names resolved to direct subclasses of the standard trainer. This check did not load a study checkpoint, predict an MRI, or reproduce the study's Linux/CUDA software environment.

The installed nnU-Net 2.8.1 distribution identifies its upstream license as Apache-2.0 and its official source as [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet). Other third-party packages retain their own licenses. No project-wide license file was located with the archived study code; the authors should select any public grant for their own code in the release root. That grant cannot authorize redistribution of the MRI datasets or third-party software.
