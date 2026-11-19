#!/usr/bin/env python3
"""Evaluate native-space pediatric BraTS segmentations with empty-case audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage


REGIONS = {
    "ET": (1,),
    "NET": (2,),
    "CC": (3,),
    "ED": (4,),
    "TC": (1, 2, 3),
    "WT": (1, 2, 3, 4),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", type=Path, required=True)
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
    parser.add_argument("--fold", type=int, required=True, choices=(0, 1, 2))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/formal_segmentation_metrics.json"),
    )
    parser.add_argument(
        "--per-case-output",
        type=Path,
        default=Path("results/formal_segmentation_per_case.csv"),
    )
    parser.add_argument("--surface-tolerance-mm", type=float, default=1.0)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--case-limit", type=int, default=0)
    return parser.parse_args()


def dice_score(reference: np.ndarray, prediction: np.ndarray) -> float:
    denominator = int(reference.sum() + prediction.sum())
    if denominator == 0:
        return 1.0
    return 2.0 * float((reference & prediction).sum()) / denominator


def surface(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    structure = ndimage.generate_binary_structure(3, 1)
    return mask & ~ndimage.binary_erosion(
        mask, structure=structure, border_value=0
    )


def surface_metrics(
    reference: np.ndarray,
    prediction: np.ndarray,
    spacing: tuple[float, float, float],
    tolerance_mm: float,
) -> tuple[float | None, float | None]:
    reference_nonempty = bool(reference.any())
    prediction_nonempty = bool(prediction.any())
    if not reference_nonempty and not prediction_nonempty:
        return 0.0, 1.0
    if not reference_nonempty or not prediction_nonempty:
        return None, 0.0
    reference_surface = surface(reference)
    prediction_surface = surface(prediction)
    distance_to_reference = ndimage.distance_transform_edt(
        ~reference_surface, sampling=spacing
    )
    distance_to_prediction = ndimage.distance_transform_edt(
        ~prediction_surface, sampling=spacing
    )
    prediction_distances = distance_to_reference[prediction_surface]
    reference_distances = distance_to_prediction[reference_surface]
    all_distances = np.concatenate(
        (prediction_distances, reference_distances)
    )
    hd95 = float(np.percentile(all_distances, 95))
    nsd = float(
        (
            np.count_nonzero(prediction_distances <= tolerance_mm)
            + np.count_nonzero(reference_distances <= tolerance_mm)
        )
        / (
            len(prediction_distances)
            + len(reference_distances)
        )
    )
    return hd95, nsd


def lesion_counts(
    reference: np.ndarray, prediction: np.ndarray
) -> dict[str, int]:
    structure = ndimage.generate_binary_structure(3, 3)
    reference_components, reference_count = ndimage.label(
        reference, structure=structure
    )
    prediction_components, prediction_count = ndimage.label(
        prediction, structure=structure
    )
    detected_reference = 0
    for lesion_id in range(1, reference_count + 1):
        if prediction[reference_components == lesion_id].any():
            detected_reference += 1
    correct_prediction = 0
    for lesion_id in range(1, prediction_count + 1):
        if reference[prediction_components == lesion_id].any():
            correct_prediction += 1
    return {
        "reference_lesions": int(reference_count),
        "detected_reference_lesions": int(detected_reference),
        "predicted_lesions": int(prediction_count),
        "correct_predicted_lesions": int(correct_prediction),
    }


def bootstrap_mean(
    values: list[float], samples: int, generator: np.random.Generator
) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None}
    if samples <= 0:
        return {
            "n": len(array),
            "mean": float(array.mean()),
            "ci_low": None,
            "ci_high": None,
        }
    indices = generator.integers(0, len(array), size=(samples, len(array)))
    distribution = array[indices].mean(axis=1)
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "ci_low": float(np.percentile(distribution, 2.5)),
        "ci_high": float(np.percentile(distribution, 97.5)),
    }


def aggregate(
    frame: pd.DataFrame,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    generator = np.random.default_rng(seed)
    summary: dict[str, Any] = {}
    for region in REGIONS:
        region_rows = frame.loc[frame["region"].eq(region)]
        positive_rows = region_rows.loc[region_rows["reference_nonempty"]]
        negative_rows = region_rows.loc[~region_rows["reference_nonempty"]]
        lesion_reference = int(
            region_rows["reference_lesions"].sum()
        )
        lesion_prediction = int(
            region_rows["predicted_lesions"].sum()
        )
        summary[region] = {
            "all_cases": {
                "dice": bootstrap_mean(
                    region_rows["dice"].tolist(),
                    bootstrap_samples,
                    generator,
                ),
                "hd95_mm": bootstrap_mean(
                    region_rows["hd95_mm"].dropna().tolist(),
                    bootstrap_samples,
                    generator,
                ),
                "nsd_1mm": bootstrap_mean(
                    region_rows["nsd_1mm"].tolist(),
                    bootstrap_samples,
                    generator,
                ),
            },
            "reference_positive_cases": {
                "count": len(positive_rows),
                "dice": bootstrap_mean(
                    positive_rows["dice"].tolist(),
                    bootstrap_samples,
                    generator,
                ),
                "hd95_mm": bootstrap_mean(
                    positive_rows["hd95_mm"].dropna().tolist(),
                    bootstrap_samples,
                    generator,
                ),
                "nsd_1mm": bootstrap_mean(
                    positive_rows["nsd_1mm"].tolist(),
                    bootstrap_samples,
                    generator,
                ),
            },
            "empty_case_audit": {
                "reference_empty": len(negative_rows),
                "prediction_empty": int(
                    (~region_rows["prediction_nonempty"]).sum()
                ),
                "both_empty": int(
                    (
                        ~region_rows["reference_nonempty"]
                        & ~region_rows["prediction_nonempty"]
                    ).sum()
                ),
                "reference_empty_prediction_nonempty": int(
                    (
                        ~region_rows["reference_nonempty"]
                        & region_rows["prediction_nonempty"]
                    ).sum()
                ),
                "reference_nonempty_prediction_empty": int(
                    (
                        region_rows["reference_nonempty"]
                        & ~region_rows["prediction_nonempty"]
                    ).sum()
                ),
            },
            "lesion_level": {
                "reference_lesions": lesion_reference,
                "predicted_lesions": lesion_prediction,
                "sensitivity": (
                    float(
                        region_rows[
                            "detected_reference_lesions"
                        ].sum()
                    )
                    / lesion_reference
                    if lesion_reference
                    else None
                ),
                "ppv": (
                    float(
                        region_rows[
                            "correct_predicted_lesions"
                        ].sum()
                    )
                    / lesion_prediction
                    if lesion_prediction
                    else None
                ),
            },
        }
        if region == "ET":
            specificity = (
                float(
                    (
                        ~negative_rows["prediction_nonempty"]
                    ).mean()
                )
                if len(negative_rows)
                else None
            )
            false_positive_volumes = negative_rows[
                "prediction_volume_mm3"
            ].tolist()
            summary[region]["negative_case_specificity"] = specificity
            summary[region]["negative_case_false_positive_volume_mm3"] = (
                bootstrap_mean(
                    false_positive_volumes,
                    bootstrap_samples,
                    generator,
                )
            )
    return summary


def main() -> None:
    args = parse_args()
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
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for case_id in case_ids:
        reference_path = Path(str(manifest.loc[case_id, "seg"]))
        prediction_path = args.prediction_dir / f"{case_id}.nii.gz"
        if not prediction_path.exists():
            errors.append(f"Missing prediction: {prediction_path}")
            continue
        reference_image = nib.load(reference_path)
        prediction_image = nib.load(prediction_path)
        reference = np.asarray(reference_image.dataobj, dtype=np.uint8)
        prediction = np.asarray(prediction_image.dataobj, dtype=np.uint8)
        if reference.shape != prediction.shape:
            errors.append(
                f"{case_id}: shape mismatch {reference.shape} vs "
                f"{prediction.shape}"
            )
            continue
        if not np.allclose(
            reference_image.affine, prediction_image.affine, atol=1e-4
        ):
            errors.append(f"{case_id}: affine mismatch")
            continue
        spacing = tuple(
            float(value) for value in reference_image.header.get_zooms()[:3]
        )
        voxel_volume = float(np.prod(spacing))
        for region, labels in REGIONS.items():
            reference_mask = np.isin(reference, labels)
            prediction_mask = np.isin(prediction, labels)
            hd95, nsd = surface_metrics(
                reference_mask,
                prediction_mask,
                spacing,
                args.surface_tolerance_mm,
            )
            lesions = lesion_counts(reference_mask, prediction_mask)
            records.append(
                {
                    "case_id": case_id,
                    "source": str(manifest.loc[case_id, "source"]),
                    "fold": args.fold,
                    "region": region,
                    "reference_nonempty": bool(reference_mask.any()),
                    "prediction_nonempty": bool(prediction_mask.any()),
                    "reference_volume_mm3": (
                        float(reference_mask.sum()) * voxel_volume
                    ),
                    "prediction_volume_mm3": (
                        float(prediction_mask.sum()) * voxel_volume
                    ),
                    "dice": dice_score(reference_mask, prediction_mask),
                    "hd95_mm": hd95,
                    "nsd_1mm": nsd,
                    **lesions,
                }
            )
    if errors:
        raise SystemExit("\n".join(errors[:20]))
    frame = pd.DataFrame.from_records(records)
    args.per_case_output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.per_case_output, index=False)
    result = {
        "status": "completed",
        "prediction_dir": str(args.prediction_dir),
        "fold": args.fold,
        "held_out_source": fold["held_out_source"],
        "evaluated_cases": len(case_ids),
        "surface_tolerance_mm": args.surface_tolerance_mm,
        "empty_case_policy": {
            "both_empty": "Dice=1, HD95=0, NSD=1",
            "one_empty": "Dice=0, HD95=null, NSD=0",
            "hd95_summary": "finite values only; empty counts reported",
        },
        "metrics": aggregate(
            frame, args.bootstrap_samples, args.seed
        ),
        "per_case_output": str(args.per_case_output),
    }
    write_path = args.output
    write_path.parent.mkdir(parents=True, exist_ok=True)
    write_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
