#!/usr/bin/env python3
"""Single-shot outer-fold evaluation of the P1 joint ET refiner.

Implements FORMAL_P1_OUTER_EVAL_LOCK.md: per fold, the three v2 stage-B
seed models are probability-averaged; the ROI is centred on the WT
predicted by the frozen B0 3-seed ensemble (no reference information);
paired ET metrics are reported against the same B0 ensemble's ET channel.

Smoke mode (--smoke N) only checks the code path on fit cases using a
reference-WT ROI and produces no formal numbers.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))

from et_small_target_refinement import (  # noqa: E402
    extract_patch,
    load_native_case,
    mask_center,
    nonzero_bbox,
    paste_patch,
    patch_spec,
)
from evaluate_formal_segmentation import dice_score, lesion_counts  # noqa: E402
from joint_et_refiner import JointETRefinerUNet3D  # noqa: E402

PATCH_SHAPE = (96, 112, 112)
SEEDS = (2026, 2027, 2028)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/processed/manifest.csv")
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path("data/processed/formal_splits_v1.json"),
    )
    parser.add_argument(
        "--ensemble-dir", type=Path, default=Path("results/ensemble")
    )
    parser.add_argument(
        "--checkpoints-root", type=Path, default=Path("results/checkpoints")
    )
    parser.add_argument(
        "--p1-runs-dir", type=Path, default=Path("results/p1")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/p1_outer")
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument(
        "--smoke",
        type=int,
        default=0,
        help="Mechanics check on N fit cases (reference ROI, no outer data).",
    )
    return parser.parse_args()


def load_models(
    checkpoints_root: Path, fold: int, device: torch.device
) -> list[JointETRefinerUNet3D]:
    models = []
    for seed in SEEDS:
        path = (
            checkpoints_root
            / f"joint_et_v2_fold{fold}_seed{seed}"
            / "stageB"
            / "joint_et_best.pt"
        )
        if not path.is_file():
            raise SystemExit(f"Missing checkpoint: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = JointETRefinerUNet3D(4).to(device)
        model.load_state_dict(payload["model_state"])
        model.eval()
        models.append(model)
    return models


def dev_presence_thresholds(p1_runs_dir: Path, fold: int) -> float:
    thresholds = []
    for seed in SEEDS:
        run = json.loads(
            (p1_runs_dir / f"joint_et_v2_fold{fold}_seed{seed}.json")
            .read_text(encoding="utf-8")
        )
        thresholds.append(
            float(run["final_internal_calibration"]["presence_threshold"])
        )
    return float(np.mean(thresholds))


def load_b0_labels_cropped(
    ensemble_dir: Path, fold: int, row: pd.Series
) -> np.ndarray:
    """B0 ensemble labels transposed and cropped exactly like the image."""
    prediction_path = ensemble_dir / f"b0_fold{fold}" / f"{row['case_id']}.nii.gz"
    if not prediction_path.is_file():
        raise SystemExit(f"Missing B0 ensemble prediction: {prediction_path}")
    labels = np.asarray(
        nib.load(prediction_path).dataobj, dtype=np.uint8
    ).transpose(2, 1, 0)
    arrays = [
        np.asarray(nib.load(str(row[column])).dataobj, dtype=np.float32)
        for column in ("t1n", "t1c", "t2w", "t2f")
    ]
    image = np.stack(arrays, axis=0).transpose(0, 3, 2, 1)
    crop = nonzero_bbox(image)
    return labels[crop]


@torch.no_grad()
def predict_case(
    models: list[JointETRefinerUNet3D],
    image: torch.Tensor,
    center: tuple[int, int, int],
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    spec = patch_spec(center, tuple(image.shape[1:]), PATCH_SHAPE)
    patch = extract_patch(image, spec).unsqueeze(0).float().to(device)
    probability_sum = torch.zeros(
        (1, 1, *PATCH_SHAPE), dtype=torch.float32, device=device
    )
    presence_probabilities = []
    for model in models:
        logits, presence_logit = model.forward_joint(patch)
        probability_sum += torch.sigmoid(logits)
        presence_probabilities.append(
            float(torch.sigmoid(presence_logit).cpu()[0])
        )
    mean_probability = probability_sum / len(models)
    patch_mask = (mean_probability >= 0.5).cpu().squeeze(0).squeeze(0)
    full_mask = paste_patch(
        patch_mask.to(torch.uint8), spec, tuple(image.shape[1:])
    )
    return full_mask.bool(), float(np.mean(presence_probabilities))


def case_record(
    reference: np.ndarray, prediction: np.ndarray, voxel_volume: float
) -> dict:
    lesions = lesion_counts(reference, prediction)
    return {
        "dice": dice_score(reference, prediction),
        "reference_nonempty": bool(reference.any()),
        "prediction_nonempty": bool(prediction.any()),
        "prediction_volume_mm3": float(prediction.sum()) * voxel_volume,
        **lesions,
    }


def paired_bootstrap(
    first: np.ndarray,
    second: np.ndarray,
    samples: int,
    seed: int,
) -> dict:
    generator = np.random.default_rng(seed)
    differences = second - first
    if differences.size == 0:
        return {"mean_difference": None, "ci_low": None, "ci_high": None}
    means = []
    for _ in range(samples):
        index = generator.integers(0, differences.size, differences.size)
        means.append(float(differences[index].mean()))
    return {
        "mean_difference": float(differences.mean()),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
    }


def evaluate_fold(
    args: argparse.Namespace,
    fold_definition: dict,
    manifest: pd.DataFrame,
    device: torch.device,
) -> tuple[list[dict], dict]:
    fold = int(fold_definition["fold"])
    models = load_models(args.checkpoints_root, fold, device)
    gate_threshold = dev_presence_thresholds(args.p1_runs_dir, fold)

    if args.smoke > 0:
        case_ids = list(fold_definition["train"])[: args.smoke]
    else:
        case_ids = list(fold_definition["val"])

    records = []
    for position, case_id in enumerate(case_ids, start=1):
        row = manifest.loc[case_id]
        native = load_native_case(row)
        image = native["image"]
        target = native["target"]
        reference_et = target[0].bool().numpy()
        spacing_text = str(row.get("spacing", "(1.0, 1.0, 1.0)"))
        spacing_values = [
            float(value) for value in re.findall(r"\d+\.?\d*", spacing_text)
        ][:3]
        if len(spacing_values) != 3:
            raise SystemExit(f"{case_id}: cannot parse spacing {spacing_text}")
        voxel_volume = float(np.prod(spacing_values))

        if args.smoke > 0:
            roi_source = target[2].bool()
            b0_et = np.zeros_like(reference_et)
        else:
            b0_labels = load_b0_labels_cropped(args.ensemble_dir, fold, row)
            if b0_labels.shape != reference_et.shape:
                raise SystemExit(
                    f"{case_id}: cropped B0 shape {b0_labels.shape} != "
                    f"target {reference_et.shape}"
                )
            roi_source = torch.from_numpy(b0_labels > 0)
            b0_et = b0_labels == 1

        center = mask_center(roi_source)
        p1_mask, presence_probability = predict_case(
            models, image, center, device
        )
        p1_et = p1_mask.numpy()
        gated_et = (
            p1_et
            if presence_probability >= gate_threshold
            else np.zeros_like(p1_et)
        )

        entry: dict = {
            "case_id": case_id,
            "fold": fold,
            "source": str(row["source"]),
            "presence_probability": presence_probability,
            "gate_threshold": gate_threshold,
            "roi_nonempty": bool(roi_source.any()),
        }
        for method, mask in (
            ("b0_ensemble", b0_et),
            ("p1_ungated", p1_et),
            ("p1_gated", gated_et),
        ):
            method_record = case_record(reference_et, mask, voxel_volume)
            entry.update(
                {f"{method}_{key}": value for key, value in method_record.items()}
            )
        records.append(entry)
        if position == 1 or position % 20 == 0 or position == len(case_ids):
            print(f"fold{fold}: {position}/{len(case_ids)}", flush=True)

    frame = pd.DataFrame.from_records(records)
    summary: dict = {
        "fold": fold,
        "held_out_source": fold_definition["held_out_source"],
        "n_cases": int(len(frame)),
        "gate_threshold": gate_threshold,
        "smoke": bool(args.smoke),
    }
    positives = frame[frame["b0_ensemble_reference_nonempty"]]
    negatives = frame[~frame["b0_ensemble_reference_nonempty"]]
    for method in ("b0_ensemble", "p1_ungated", "p1_gated"):
        positive_dice = positives[f"{method}_dice"].to_numpy(dtype=float)
        summary[method] = {
            "positive_case_dice": (
                float(positive_dice.mean()) if positive_dice.size else None
            ),
            "all_case_dice": float(
                frame[f"{method}_dice"].to_numpy(dtype=float).mean()
            ),
            "negative_specificity": (
                float((~negatives[f"{method}_prediction_nonempty"]).mean())
                if len(negatives)
                else None
            ),
            "negative_fp_volume_mm3": (
                float(negatives[f"{method}_prediction_volume_mm3"].mean())
                if len(negatives)
                else None
            ),
            "lesion_sensitivity": (
                float(positives[f"{method}_detected_reference_lesions"].sum())
                / max(1, int(positives[f"{method}_reference_lesions"].sum()))
            ),
        }
    for method in ("p1_ungated", "p1_gated"):
        summary[f"{method}_vs_b0_positive_dice"] = paired_bootstrap(
            positives["b0_ensemble_dice"].to_numpy(dtype=float),
            positives[f"{method}_dice"].to_numpy(dtype=float),
            args.bootstrap_samples,
            args.bootstrap_seed,
        )
    return records, summary


def main() -> None:
    args = parse_args()
    started = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = json.loads(args.splits.read_text(encoding="utf-8"))
    manifest = pd.read_csv(args.manifest).set_index("case_id", drop=False)

    all_records: list[dict] = []
    summaries = []
    for fold_definition in splits["outer_folds"]:
        if int(fold_definition["fold"]) not in args.folds:
            continue
        records, summary = evaluate_fold(
            args, fold_definition, manifest, device
        )
        all_records.extend(records)
        summaries.append(summary)
        print(json.dumps(summary, indent=2), flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if args.smoke else ""
    frame = pd.DataFrame.from_records(all_records)
    frame.to_csv(
        args.output_dir / f"p1_outer_per_case{suffix}.csv", index=False
    )
    payload = {
        "protocol": "FORMAL_P1_OUTER_EVAL_LOCK.md v1",
        "smoke": bool(args.smoke),
        "seeds": list(SEEDS),
        "patch_shape": list(PATCH_SHAPE),
        "elapsed_seconds": round(time.time() - started, 1),
        "folds": summaries,
    }
    (args.output_dir / f"p1_outer_eval{suffix}.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print("P1_OUTER_EVAL_DONE" + ("_SMOKE" if args.smoke else ""), flush=True)


if __name__ == "__main__":
    main()
