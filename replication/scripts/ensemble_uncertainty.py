#!/usr/bin/env python3
"""Seed-ensemble B0 validation predictions and quantify uncertainty.

For one outer fold this script:
1. averages the per-seed nnU-Net softmax probabilities (.npz saved during
   validation) into an ensemble segmentation per case;
2. writes native-space ensemble predictions that the frozen evaluator
   (evaluate_formal_segmentation.py) scores exactly like single runs;
3. derives per-case uncertainty scalars (normalised entropy, inter-seed
   disagreement, pairwise seed Dice, volume CV);
4. joins uncertainty with ensemble per-case Dice to report failure-detection
   AUROC, Spearman rank correlation, and risk-coverage curves.

Only outer-fold validation outputs are touched; no model is retrained, so
this stays within the frozen protocol (post-hoc analysis of B0 runs).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

NUM_CLASSES = 5
WT_LABELS = (1, 2, 3, 4)
ET_LABELS = (1,)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, choices=(0, 1, 2))
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[2026, 2027, 2028]
    )
    parser.add_argument(
        "--validation-template",
        default=(
            "data/nnunet/results/Dataset501_BraTSPEDs/"
            "nnUNetTrainerPedGlioSeed{seed}__nnUNetResEncUNetMPlans__"
            "3d_fullres/fold_{fold}/validation"
        ),
    )
    parser.add_argument(
        "--ensemble-dir", type=Path, default=Path("results/ensemble")
    )
    parser.add_argument("--eval-dir", type=Path, default=Path("results/eval"))
    parser.add_argument(
        "--analysis-dir", type=Path, default=Path("results/uncertainty")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/processed/manifest.csv")
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path("data/processed/formal_splits_v1.json"),
    )
    parser.add_argument(
        "--evaluator",
        type=Path,
        default=Path("scripts/evaluate_formal_segmentation.py"),
    )
    parser.add_argument("--dice-failure-threshold", type=float, default=0.5)
    parser.add_argument(
        "--skip-ensemble",
        action="store_true",
        help="Reuse existing ensemble predictions and evaluation.",
    )
    return parser.parse_args()


def load_probabilities(npz_path: Path, expected_shape: tuple) -> np.ndarray:
    payload = np.load(npz_path)
    key = "probabilities" if "probabilities" in payload else payload.files[0]
    probabilities = payload[key].astype(np.float32)
    if probabilities.shape[0] != NUM_CLASSES:
        raise SystemExit(
            f"{npz_path}: expected {NUM_CLASSES} classes, got "
            f"{probabilities.shape}"
        )
    if probabilities.shape[1:] == expected_shape:
        return probabilities
    transposed = probabilities.transpose(0, 3, 2, 1)
    if transposed.shape[1:] == expected_shape:
        return transposed
    raise SystemExit(
        f"{npz_path}: probability shape {probabilities.shape[1:]} does not "
        f"match reference {expected_shape}"
    )


def entropy_map(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-8, 1.0)
    entropy = -(clipped * np.log(clipped)).sum(axis=0)
    return entropy / np.log(NUM_CLASSES)


def pairwise_dice(masks: list[np.ndarray]) -> float:
    scores = []
    for first in range(len(masks)):
        for second in range(first + 1, len(masks)):
            denominator = int(masks[first].sum() + masks[second].sum())
            if denominator == 0:
                scores.append(1.0)
            else:
                intersection = int(
                    np.logical_and(masks[first], masks[second]).sum()
                )
                scores.append(2.0 * intersection / denominator)
    return float(np.mean(scores))


def volume_cv(volumes: list[float]) -> float:
    mean = float(np.mean(volumes))
    if mean <= 0:
        return 0.0
    return float(np.std(volumes) / mean)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    positives = scores[labels]
    negatives = scores[~labels]
    if positives.size == 0 or negatives.size == 0:
        return None
    order = np.argsort(np.concatenate([positives, negatives]), kind="stable")
    ranks = np.empty(order.size, dtype=np.float64)
    ranks[order] = np.arange(1, order.size + 1)
    combined = np.concatenate([positives, negatives])
    sorted_values = np.sort(combined)
    for value in np.unique(sorted_values):
        matches = combined == value
        if matches.sum() > 1:
            ranks[matches] = ranks[matches].mean()
    rank_sum = ranks[: positives.size].sum()
    u_statistic = rank_sum - positives.size * (positives.size + 1) / 2
    return float(u_statistic / (positives.size * negatives.size))


def spearman(first: np.ndarray, second: np.ndarray) -> float | None:
    if first.size < 3:
        return None
    first_ranks = pd.Series(first).rank().to_numpy()
    second_ranks = pd.Series(second).rank().to_numpy()
    if np.std(first_ranks) == 0 or np.std(second_ranks) == 0:
        return None
    return float(np.corrcoef(first_ranks, second_ranks)[0, 1])


def risk_coverage(
    uncertainty: np.ndarray, dice: np.ndarray
) -> list[dict[str, float]]:
    order = np.argsort(uncertainty, kind="stable")
    sorted_dice = dice[order]
    curve = []
    for coverage in range(10, 101, 10):
        keep = max(1, int(round(sorted_dice.size * coverage / 100)))
        retained = sorted_dice[:keep]
        curve.append(
            {
                "coverage_pct": coverage,
                "retained_cases": int(keep),
                "mean_dice": float(retained.mean()),
                "failure_rate": float((retained < 0.5).mean()),
            }
        )
    return curve


def build_ensemble(args: argparse.Namespace) -> tuple[Path, Path]:
    seed_dirs = [
        Path(args.validation_template.format(seed=seed, fold=args.fold))
        for seed in args.seeds
    ]
    for directory in seed_dirs:
        if not directory.is_dir():
            raise SystemExit(f"Missing validation dir: {directory}")
    case_sets = [
        {path.name[: -len(".nii.gz")] for path in d.glob("*.nii.gz")}
        for d in seed_dirs
    ]
    case_ids = sorted(set.intersection(*case_sets))
    if not case_ids:
        raise SystemExit("No shared validation cases across seeds")
    if len(set(map(len, case_sets))) != 1:
        raise SystemExit(
            f"Seed case counts differ: {[len(s) for s in case_sets]}"
        )

    prediction_dir = args.ensemble_dir / f"b0_fold{args.fold}"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    uncertainty_rows = []
    for index, case_id in enumerate(case_ids, start=1):
        reference_image = nib.load(seed_dirs[0] / f"{case_id}.nii.gz")
        shape = tuple(reference_image.shape)
        probability_sum = np.zeros((NUM_CLASSES, *shape), dtype=np.float32)
        seed_argmaxes = []
        for directory in seed_dirs:
            probabilities = load_probabilities(
                directory / f"{case_id}.npz", shape
            )
            probability_sum += probabilities
            seed_argmaxes.append(
                probabilities.argmax(axis=0).astype(np.uint8)
            )
        mean_probabilities = probability_sum / len(seed_dirs)
        ensemble_segmentation = mean_probabilities.argmax(axis=0).astype(
            np.uint8
        )
        nib.save(
            nib.Nifti1Image(
                ensemble_segmentation,
                reference_image.affine,
                reference_image.header,
            ),
            prediction_dir / f"{case_id}.nii.gz",
        )

        union = ensemble_segmentation > 0
        for argmax in seed_argmaxes:
            union |= argmax > 0
        union |= (1.0 - mean_probabilities[0]) > 0.1
        if union.any():
            entropy = entropy_map(mean_probabilities)
            mean_entropy = float(entropy[union].mean())
            stacked = np.stack([a[union] for a in seed_argmaxes])
            disagreement = float(
                (stacked != stacked[0]).any(axis=0).mean()
            )
        else:
            mean_entropy = 0.0
            disagreement = 0.0
        wt_masks = [np.isin(a, WT_LABELS) for a in seed_argmaxes]
        et_masks = [np.isin(a, ET_LABELS) for a in seed_argmaxes]
        uncertainty_rows.append(
            {
                "case_id": case_id,
                "fold": args.fold,
                "mean_entropy": mean_entropy,
                "seed_disagreement": disagreement,
                "wt_pairwise_dice": pairwise_dice(wt_masks),
                "et_pairwise_dice": pairwise_dice(et_masks),
                "wt_volume_cv": volume_cv(
                    [float(mask.sum()) for mask in wt_masks]
                ),
                "union_voxels": int(union.sum()),
            }
        )
        if index % 20 == 0 or index == len(case_ids):
            print(f"ensembled {index}/{len(case_ids)}", flush=True)

    uncertainty_path = (
        args.analysis_dir / f"b0_fold{args.fold}_uncertainty_per_case.csv"
    )
    uncertainty_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(uncertainty_rows).to_csv(uncertainty_path, index=False)
    return prediction_dir, uncertainty_path


def analyse(
    args: argparse.Namespace,
    uncertainty_path: Path,
    per_case_eval: Path,
) -> dict:
    uncertainty = (
        pd.read_csv(uncertainty_path)
        .drop(columns=["fold"])
        .set_index("case_id")
    )
    evaluation = pd.read_csv(per_case_eval)
    scalars = [
        "mean_entropy",
        "seed_disagreement",
        "wt_volume_cv",
    ]
    inverted = ["wt_pairwise_dice", "et_pairwise_dice"]
    analysis: dict = {
        "fold": args.fold,
        "seeds": args.seeds,
        "dice_failure_threshold": args.dice_failure_threshold,
        "targets": {},
    }
    targets = {
        "WT": evaluation[evaluation["region"] == "WT"],
        "ET_positive": evaluation[
            (evaluation["region"] == "ET")
            & evaluation["reference_nonempty"]
        ],
    }
    for target_name, frame in targets.items():
        frame = frame.set_index("case_id").join(uncertainty, how="inner")
        dice = frame["dice"].to_numpy()
        failures = dice < args.dice_failure_threshold
        entry: dict = {
            "n_cases": int(frame.shape[0]),
            "n_failures": int(failures.sum()),
            "mean_dice": float(dice.mean()),
            "signals": {},
        }
        for scalar in scalars + inverted:
            values = frame[scalar].to_numpy(dtype=np.float64)
            scores = -values if scalar in inverted else values
            entry["signals"][scalar] = {
                "failure_auroc": auroc(scores, failures),
                "spearman_vs_dice": spearman(scores, dice),
            }
        primary = frame["mean_entropy"].to_numpy(dtype=np.float64)
        entry["risk_coverage_mean_entropy"] = risk_coverage(primary, dice)
        analysis["targets"][target_name] = entry
    return analysis


def main() -> None:
    args = parse_args()
    prediction_dir = args.ensemble_dir / f"b0_fold{args.fold}"
    uncertainty_path = (
        args.analysis_dir / f"b0_fold{args.fold}_uncertainty_per_case.csv"
    )
    if not args.skip_ensemble or not uncertainty_path.exists():
        prediction_dir, uncertainty_path = build_ensemble(args)

    eval_json = args.eval_dir / f"ensemble_b0_fold{args.fold}.json"
    per_case_eval = args.eval_dir / f"ensemble_b0_fold{args.fold}_per_case.csv"
    if not args.skip_ensemble or not per_case_eval.exists():
        subprocess.run(
            [
                sys.executable,
                str(args.evaluator),
                "--prediction-dir",
                str(prediction_dir),
                "--manifest",
                str(args.manifest),
                "--splits",
                str(args.splits),
                "--fold",
                str(args.fold),
                "--output",
                str(eval_json),
                "--per-case-output",
                str(per_case_eval),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )

    analysis = analyse(args, uncertainty_path, per_case_eval)
    analysis_path = (
        args.analysis_dir / f"b0_fold{args.fold}_uncertainty_analysis.json"
    )
    analysis_path.write_text(
        json.dumps(analysis, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
