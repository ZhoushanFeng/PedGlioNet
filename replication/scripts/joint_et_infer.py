#!/usr/bin/env python3
"""Deployment-honest P1 inference for one outer fold and seed.

For every held-out case the ET ROI is localised from the B0 model's own
predicted whole tumor (never the reference), the joint refiner segments ET
inside the 1 mm native-resolution ROI, the presence head gates empty-ET
calls with the threshold frozen during development, and the refined ET is
composed back into the B0 multiclass prediction:

- voxels the refiner marks as ET become label 1;
- voxels B0 called ET but the refiner rejects fall back to NET (label 2);
- everything else keeps the B0 label.

The composed segmentations can be scored by evaluate_formal_segmentation.py
exactly like any other model. An oracle-ROI mode (reference WT
localisation) exists purely as an upper-bound diagnostic.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from et_small_target_refinement import (
    extract_patch,
    mask_center,
    paste_patch,
    patch_spec,
)
from joint_et_refiner import JointETRefinerUNet3D
from pilot_transfer import native_to_common, nonzero_bbox, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, choices=(0, 1, 2))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--b0-prediction-dir",
        type=Path,
        required=True,
        help="Directory with the B0 validation NIfTI predictions.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Joint refiner checkpoint (joint_et_best.pt).",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/manifest.csv"),
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path("data/processed/formal_splits_v1.json"),
    )
    parser.add_argument(
        "--roi-source",
        choices=("predicted", "oracle"),
        default="predicted",
        help="oracle uses reference WT and is a diagnostic upper bound only.",
    )
    parser.add_argument(
        "--patch-shape", type=int, nargs=3, default=(96, 112, 112)
    )
    parser.add_argument("--base-channels", type=int, default=4)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    parser.add_argument("--mps-memory-fraction", type=float, default=0.30)
    parser.add_argument(
        "--presence-threshold",
        type=float,
        default=None,
        help="Override; by default the frozen threshold in the checkpoint.",
    )
    parser.add_argument("--case-limit", type=int, default=0)
    return parser.parse_args()


def load_case_arrays(row: pd.Series) -> dict[str, Any]:
    """Native-resolution arrays in the (Z, Y, X) layout used for training."""
    modalities = [
        np.asarray(nib.load(str(row[column])).dataobj, dtype=np.float32)
        for column in ("t1n", "t1c", "t2w", "t2f")
    ]
    image = np.stack(modalities, axis=0).transpose(0, 3, 2, 1)
    reference_image = nib.load(str(row["seg"]))
    reference = np.asarray(reference_image.dataobj, dtype=np.int16).transpose(
        2, 1, 0
    )
    spacing = tuple(
        float(value) for value in reference_image.header.get_zooms()[:3]
    )
    return {
        "image": image,
        "reference": reference,
        "affine": reference_image.affine,
        "spacing": spacing,
    }


def normalise(image: np.ndarray) -> np.ndarray:
    foreground = np.any(np.abs(image) > 1e-6, axis=0)
    for channel in range(image.shape[0]):
        values = image[channel][foreground]
        if values.size:
            mean = float(values.mean())
            std = max(float(values.std()), 1e-6)
            image[channel] = np.clip((image[channel] - mean) / std, -5.0, 5.0)
            image[channel][~foreground] = 0.0
    return image


def main() -> None:
    args = parse_args()
    started_at = time.time()
    device = select_device(args.device, args.mps_memory_fraction)

    bundle = torch.load(args.model, map_location="cpu", weights_only=False)
    model = JointETRefinerUNet3D(args.base_channels)
    model.load_state_dict(bundle["model_state"])
    model.to(device)
    model.eval()
    threshold = (
        args.presence_threshold
        if args.presence_threshold is not None
        else float(bundle["metadata"]["presence_threshold"])
    )

    split_definition = json.loads(args.splits.read_text(encoding="utf-8"))
    fold = next(
        item
        for item in split_definition["outer_folds"]
        if int(item["fold"]) == args.fold
    )
    case_ids = list(fold["val"])
    if args.case_limit > 0:
        case_ids = case_ids[: args.case_limit]
    manifest = pd.read_csv(args.manifest).set_index("case_id", drop=False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    patch_shape = tuple(args.patch_shape)
    records: list[dict[str, Any]] = []
    for case_id in case_ids:
        row = manifest.loc[case_id]
        arrays = load_case_arrays(row)
        prediction_path = args.b0_prediction_dir / f"{case_id}.nii.gz"
        if not prediction_path.exists():
            raise SystemExit(f"Missing B0 prediction: {prediction_path}")
        b0_image = nib.load(str(prediction_path))
        b0_native = np.asarray(b0_image.dataobj, dtype=np.uint8)
        b0 = b0_native.transpose(2, 1, 0)

        crop = nonzero_bbox(arrays["image"])
        image = normalise(arrays["image"][(slice(None),) + crop].copy())
        b0_cropped = b0[crop]
        reference_cropped = arrays["reference"][crop]
        spatial_shape = image.shape[1:]

        if args.roi_source == "predicted":
            roi_mask = torch.from_numpy(b0_cropped > 0)
        else:
            common = native_to_common(reference_cropped, "brats_peds")
            roi_mask = torch.from_numpy(common[2] > 0.5)
        roi_fallback = not bool(roi_mask.any())
        center = mask_center(roi_mask)
        spec = patch_spec(center, spatial_shape, patch_shape)

        image_tensor = (
            extract_patch(torch.from_numpy(image), spec)
            .unsqueeze(0)
            .to(device=device, dtype=torch.float32)
        )
        with torch.inference_mode():
            logits, presence_logit = model.forward_joint(image_tensor)
            presence_probability = float(
                torch.sigmoid(presence_logit).cpu()[0]
            )
            et_patch = (
                (torch.sigmoid(logits).cpu().squeeze(0).squeeze(0) >= 0.5)
                .to(torch.uint8)
            )
        gated_off = presence_probability < threshold
        if gated_off:
            et_patch = torch.zeros_like(et_patch)
        refined_cropped = (
            paste_patch(et_patch, spec, spatial_shape).numpy().astype(bool)
        )

        refined_full = np.zeros(b0.shape, dtype=bool)
        refined_full[crop] = refined_cropped
        refined_native = refined_full.transpose(2, 1, 0)

        composed = b0_native.copy()
        composed[(b0_native == 1) & ~refined_native] = 2
        composed[refined_native] = 1
        nib.save(
            nib.Nifti1Image(composed.astype(np.uint8), b0_image.affine),
            str(args.output_dir / f"{case_id}.nii.gz"),
        )

        voxel_volume = float(np.prod(arrays["spacing"]))
        reference_et = arrays["reference"] == 1
        roi_box = np.zeros(b0.shape, dtype=bool)
        roi_box[crop] = (
            paste_patch(
                torch.ones(patch_shape, dtype=torch.uint8), spec, spatial_shape
            )
            .numpy()
            .astype(bool)
        )
        roi_box_native = roi_box.transpose(2, 1, 0)
        reference_et_voxels = int(reference_et.sum())
        records.append(
            {
                "case_id": case_id,
                "source": str(row["source"]),
                "roi_source": args.roi_source,
                "roi_fallback_no_wt": roi_fallback,
                "presence_probability": presence_probability,
                "presence_threshold": threshold,
                "gated_off": bool(gated_off),
                "reference_et_voxels": reference_et_voxels,
                "reference_et_volume_mm3": reference_et_voxels * voxel_volume,
                "roi_coverage_of_reference_et": (
                    float(
                        (reference_et & roi_box_native).sum()
                        / reference_et_voxels
                    )
                    if reference_et_voxels
                    else None
                ),
                "b0_et_voxels": int((b0_native == 1).sum()),
                "refined_et_voxels": int(refined_native.sum()),
                "false_positive_et_volume_mm3": float(
                    ((composed == 1) & ~reference_et).sum() * voxel_volume
                ),
            }
        )

    frame = pd.DataFrame.from_records(records)
    positives = frame.loc[frame["reference_et_voxels"] > 0]
    negatives = frame.loc[frame["reference_et_voxels"] == 0]
    report = {
        "status": "completed",
        "fold": args.fold,
        "seed": args.seed,
        "held_out_source": fold["held_out_source"],
        "roi_source": args.roi_source,
        "model": str(args.model),
        "b0_prediction_dir": str(args.b0_prediction_dir),
        "output_dir": str(args.output_dir),
        "presence_threshold": threshold,
        "cases": len(frame),
        "et_positive_cases": int(len(positives)),
        "et_negative_cases": int(len(negatives)),
        "roi_fallback_cases": int(frame["roi_fallback_no_wt"].sum()),
        "mean_roi_coverage_of_reference_et": (
            float(positives["roi_coverage_of_reference_et"].mean())
            if len(positives)
            else None
        ),
        "gated_off_positive_cases": int(positives["gated_off"].sum()),
        "gated_off_negative_cases": int(negatives["gated_off"].sum()),
        "mean_false_positive_et_volume_mm3_negatives": (
            float(negatives["false_positive_et_volume_mm3"].mean())
            if len(negatives)
            else None
        ),
        "per_case": records,
        "elapsed_seconds": time.time() - started_at,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary = {
        key: value for key, value in report.items() if key != "per_case"
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
