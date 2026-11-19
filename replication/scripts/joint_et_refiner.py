#!/usr/bin/env python3
"""Joint ET segmentation and presence-head development training.

This script only uses the development portion of one frozen outer fold. It
creates a deterministic source-stratified internal calibration split, trains
the joint model, and selects an ET-presence threshold without touching the
outer held-out source.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from et_presence_gate import choose_threshold
from et_small_target_refinement import (
    ETRefinerUNet3D,
    focal_tversky_loss,
    prepare_training_patches,
    translate_with_zeros,
)
from full_transfer_train import MemoryTracker, cpu_state_dict, write_json
from pilot_transfer import complete_rows, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--outer-fold", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/joint_et_fold2_development.json"),
    )
    parser.add_argument(
        "--status-output",
        type=Path,
        default=Path("results/joint_et_fold2_status.json"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("results/checkpoints/joint_et_fold2"),
    )
    parser.add_argument(
        "--patch-shape", type=int, nargs=3, default=(96, 112, 112)
    )
    parser.add_argument("--base-channels", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--presence-loss-weight", type=float, default=1.0)
    parser.add_argument("--empty-target-weight", type=float, default=1.0)
    parser.add_argument("--empty-target-topk", type=int, default=8192)
    parser.add_argument(
        "--positive-repeat",
        type=int,
        default=1,
        help=(
            "How many times each ET-positive fit case appears per epoch. "
            "The proven stage-A recipe uses 2 so positives escape the "
            "all-background local optimum."
        ),
    )
    parser.add_argument(
        "--initial-model",
        type=Path,
        default=None,
        help="Optional joint_et_best.pt checkpoint to initialise from.",
    )
    parser.add_argument("--translation-voxels", type=int, default=8)
    parser.add_argument("--calibration-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    parser.add_argument("--mps-memory-fraction", type=float, default=0.30)
    parser.add_argument("--inter-step-sleep-ms", type=float, default=10.0)
    parser.add_argument("--fit-limit", type=int, default=0)
    parser.add_argument("--calibration-limit", type=int, default=0)
    return parser.parse_args()


class JointETRefinerUNet3D(ETRefinerUNet3D):
    def __init__(self, base: int = 4) -> None:
        super().__init__(base)
        self.presence_head = nn.Linear(base * 4, 1)

    def forward_joint(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits, features = self.forward_with_features(image)
        presence_logit = self.presence_head(features).squeeze(1)
        return logits, presence_logit

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_joint(image)
        return logits


def update_status(
    path: Path,
    stage: str,
    detail: str,
    started_at: float,
    **extra: Any,
) -> None:
    write_json(
        path,
        {
            "state": "running",
            "stage": stage,
            "detail": detail,
            "elapsed_seconds": round(time.time() - started_at, 3),
            **extra,
        },
    )


def has_et(native_labels: Any) -> bool:
    return "1" in str(native_labels).split(",")


def stratified_calibration_split(
    rows: pd.DataFrame, fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < fraction < 0.5:
        raise ValueError("calibration-fraction must be between 0 and 0.5.")
    table = rows.copy()
    table["_et_positive"] = table["native_labels"].map(has_et)
    generator = np.random.default_rng(seed)
    calibration_indices: list[Any] = []
    for _, group in table.groupby(
        ["source", "_et_positive"], dropna=False, sort=True
    ):
        indices = group.index.to_numpy(copy=True)
        generator.shuffle(indices)
        count = max(1, int(round(len(indices) * fraction)))
        count = min(count, max(len(indices) - 1, 1))
        calibration_indices.extend(indices[:count].tolist())
    calibration = table.loc[sorted(calibration_indices)].drop(
        columns="_et_positive"
    )
    fit = table.drop(index=calibration_indices).drop(columns="_et_positive")
    return fit, calibration


def limited(rows: pd.DataFrame, limit: int) -> pd.DataFrame:
    return rows.iloc[:limit] if limit > 0 else rows


def augment(
    image: torch.Tensor,
    target: torch.Tensor,
    generator: random.Random,
    translation_voxels: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    for dimension in (2, 3, 4):
        if generator.random() < 0.5:
            image = torch.flip(image, dims=(dimension,))
            target = torch.flip(target, dims=(dimension,))
    shifts = tuple(
        generator.randint(-translation_voxels, translation_voxels)
        if translation_voxels > 0
        else 0
        for _ in range(3)
    )
    image, target = translate_with_zeros(image, target, shifts)
    if generator.random() < 0.3:
        scale = 0.9 + 0.2 * generator.random()
        shift = 0.1 * (generator.random() - 0.5)
        image = image * scale + shift
    return image, target


@torch.inference_mode()
def validate(
    model: JointETRefinerUNet3D,
    cases: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    probabilities: list[float] = []
    labels: list[int] = []
    positive_dice: list[float] = []
    negative_empty: list[bool] = []
    overlap_detected: list[bool] = []
    for case in cases:
        image = (
            case["image"]
            .unsqueeze(0)
            .to(device=device, dtype=torch.float32)
        )
        target = case["target"].bool()
        logits, presence_logit = model.forward_joint(image)
        prediction = (torch.sigmoid(logits).cpu().squeeze(0) >= 0.5)
        label = int(case["et_positive"])
        probabilities.append(float(torch.sigmoid(presence_logit).cpu()[0]))
        labels.append(label)
        if label:
            intersection = int((prediction & target).sum())
            denominator = int(prediction.sum() + target.sum())
            positive_dice.append(
                2.0 * intersection / denominator if denominator else 1.0
            )
            overlap_detected.append(intersection > 0)
        else:
            negative_empty.append(not bool(prediction.any()))
    probability_array = np.asarray(probabilities, dtype=np.float32)
    label_array = np.asarray(labels, dtype=np.float32)
    threshold, threshold_metrics = choose_threshold(
        probability_array, label_array
    )
    return {
        "positive_case_dice": (
            float(np.mean(positive_dice)) if positive_dice else None
        ),
        "segmentation_overlap_sensitivity": (
            float(np.mean(overlap_detected)) if overlap_detected else None
        ),
        "ungated_negative_case_specificity": (
            float(np.mean(negative_empty)) if negative_empty else None
        ),
        "presence_threshold": threshold,
        "presence_metrics": threshold_metrics,
        "presence_probabilities": probabilities,
        "presence_labels": labels,
    }


def save_model(
    path: Path,
    model: nn.Module,
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state": cpu_state_dict(model), "metadata": metadata},
        path,
    )


def main() -> None:
    args = parse_args()
    started_at = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = select_device(args.device, args.mps_memory_fraction)
    tracker = MemoryTracker(device)

    split_definition = json.loads(args.splits.read_text(encoding="utf-8"))
    selected = [
        fold
        for fold in split_definition["outer_folds"]
        if int(fold["fold"]) == args.outer_fold
    ]
    if len(selected) != 1:
        raise SystemExit(f"Outer fold {args.outer_fold} not found.")
    outer_fold = selected[0]
    manifest = pd.read_csv(args.manifest)
    pediatric = complete_rows(manifest, "brats_peds")
    pediatric = pediatric.loc[
        pediatric["split"].astype(str).str.lower().eq("training")
        & pediatric["seg"].fillna("").ne("")
    ].set_index("case_id", drop=False)
    development_rows = pediatric.loc[outer_fold["train"]].copy()
    fit_rows, calibration_rows = stratified_calibration_split(
        development_rows,
        args.calibration_fraction,
        args.seed,
    )
    fit_rows = limited(
        fit_rows.sample(frac=1.0, random_state=args.seed), args.fit_limit
    )
    calibration_rows = limited(
        calibration_rows.sample(
            frac=1.0, random_state=args.seed + 1
        ),
        args.calibration_limit,
    )
    outer_overlap = (
        set(fit_rows["case_id"])
        | set(calibration_rows["case_id"])
    ) & set(outer_fold["val"])
    if outer_overlap:
        raise SystemExit(
            f"Outer held-out leakage detected: {len(outer_overlap)} cases."
        )

    config = {
        "status": "development_only_no_outer_test",
        "outer_fold": args.outer_fold,
        "outer_held_out_source": outer_fold["held_out_source"],
        "fit_cases": len(fit_rows),
        "calibration_cases": len(calibration_rows),
        "outer_test_cases_untouched": len(outer_fold["val"]),
        "patch_shape_dhw": tuple(args.patch_shape),
        "base_channels": args.base_channels,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "presence_loss_weight": args.presence_loss_weight,
        "empty_target_weight": args.empty_target_weight,
        "positive_repeat": args.positive_repeat,
        "initial_model": (
            str(args.initial_model) if args.initial_model else None
        ),
        "seed": args.seed,
        "device": str(device),
        "batch_size": 1,
    }
    print(json.dumps(config, indent=2, ensure_ascii=False), flush=True)
    update_status(
        args.status_output,
        "preprocessing",
        "Preparing development-only ET ROIs",
        started_at,
        config=config,
    )
    fit_cases = prepare_training_patches(
        fit_rows,
        tuple(args.patch_shape),
        args.status_output,
        started_at,
    )
    calibration_cases = prepare_training_patches(
        calibration_rows,
        tuple(args.patch_shape),
        args.status_output,
        started_at,
    )

    model = JointETRefinerUNet3D(args.base_channels).to(device)
    if args.initial_model is not None:
        if not args.initial_model.is_file():
            raise SystemExit(
                f"Initial model not found: {args.initial_model}"
            )
        payload = torch.load(
            args.initial_model, map_location="cpu", weights_only=False
        )
        model.load_state_dict(payload["model_state"])
        print(f"Initialised from {args.initial_model}", flush=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    positive_count = sum(int(case["et_positive"]) for case in fit_cases)
    negative_count = len(fit_cases) - positive_count
    positive_weight = torch.tensor(
        max(negative_count, 1) / max(positive_count, 1),
        device=device,
    )
    history: list[dict[str, Any]] = []
    best_score = float("-inf")
    best_path = args.checkpoint_dir / "joint_et_best.pt"
    sleep_seconds = max(args.inter_step_sleep_ms, 0.0) / 1000.0

    repeat = max(1, args.positive_repeat)
    epoch_indices: list[int] = []
    for index, case in enumerate(fit_cases):
        epoch_indices.extend(
            [index] * (repeat if case["et_positive"] else 1)
        )

    for epoch_index in range(args.epochs):
        epoch_started = time.time()
        generator = random.Random(args.seed + epoch_index)
        order = list(epoch_indices)
        generator.shuffle(order)
        model.train()
        total_loss = 0.0
        total_segmentation_loss = 0.0
        total_presence_loss = 0.0
        for case_index in order:
            case = fit_cases[case_index]
            image = (
                case["image"]
                .unsqueeze(0)
                .to(device=device, dtype=torch.float32)
            )
            target = (
                case["target"]
                .unsqueeze(0)
                .to(device=device, dtype=torch.float32)
            )
            image, target = augment(
                image, target, generator, args.translation_voxels
            )
            logits, presence_logit = model.forward_joint(image)
            segmentation_loss = focal_tversky_loss(
                logits,
                target,
                (
                    args.empty_target_weight
                    if not case["et_positive"]
                    else 0.0
                ),
                args.empty_target_topk,
            )
            presence_target = torch.tensor(
                [float(case["et_positive"])], device=device
            )
            presence_loss = (
                torch.nn.functional.binary_cross_entropy_with_logits(
                    presence_logit,
                    presence_target,
                    pos_weight=positive_weight,
                )
            )
            loss = (
                segmentation_loss
                + args.presence_loss_weight * presence_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            total_segmentation_loss += float(
                segmentation_loss.detach().cpu()
            )
            total_presence_loss += float(presence_loss.detach().cpu())
            if sleep_seconds:
                time.sleep(sleep_seconds)

        if device.type == "mps":
            torch.mps.synchronize()
        validation = validate(model, calibration_cases, device)
        dice = validation["positive_case_dice"] or 0.0
        balanced_accuracy = validation["presence_metrics"][
            "balanced_accuracy"
        ]
        score = dice + 0.25 * balanced_accuracy
        epoch_result = {
            "epoch": epoch_index + 1,
            "mean_total_loss": total_loss / len(order),
            "mean_segmentation_loss": (
                total_segmentation_loss / len(order)
            ),
            "mean_presence_loss": total_presence_loss / len(order),
            "calibration": {
                key: value
                for key, value in validation.items()
                if key not in (
                    "presence_probabilities",
                    "presence_labels",
                )
            },
            "selection_score": score,
            "seconds": time.time() - epoch_started,
        }
        history.append(epoch_result)
        if score > best_score:
            best_score = score
            save_model(
                best_path,
                model,
                {
                    **config,
                    "best_epoch": epoch_index + 1,
                    "presence_threshold": validation[
                        "presence_threshold"
                    ],
                    "selection_score": score,
                },
            )
        detail = (
            f"joint ET epoch {epoch_index + 1}/{args.epochs}, "
            f"loss={epoch_result['mean_total_loss']:.4f}, "
            f"calibration Dice={dice:.4f}, "
            f"presence BA={balanced_accuracy:.4f}"
        )
        print(detail, flush=True)
        tracker.sample()
        update_status(
            args.status_output,
            "training",
            detail,
            started_at,
            completed_epochs=epoch_index + 1,
            total_epochs=args.epochs,
            resource_usage=tracker.as_dict(),
        )

    best_bundle = torch.load(
        best_path, map_location="cpu", weights_only=False
    )
    model.load_state_dict(best_bundle["model_state"])
    model.to(device)
    final_calibration = validate(model, calibration_cases, device)
    result = {
        "status": "completed_development_only_no_outer_test",
        "config": config,
        "split": {
            "fit_case_ids": [
                str(case["case_id"]) for case in fit_cases
            ],
            "calibration_case_ids": [
                str(case["case_id"]) for case in calibration_cases
            ],
            "outer_test_case_ids_sha256": outer_fold["val_sha256"],
            "outer_test_source": outer_fold["held_out_source"],
            "outer_test_evaluated": False,
        },
        "history": history,
        "best_model": best_bundle["metadata"],
        "final_internal_calibration": final_calibration,
        "resource_usage": {
            "device": str(device),
            "torch_threads": args.threads,
            **tracker.as_dict(),
        },
        "elapsed_seconds": time.time() - started_at,
        "artifacts": {
            "best_model": str(best_path),
        },
    }
    write_json(args.output, result)
    write_json(
        args.status_output,
        {
            "state": "completed",
            "stage": "done",
            "detail": f"Development result written to {args.output}",
            "outer_test_evaluated": False,
            "output": str(args.output),
        },
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "config": config,
                "best_model": result["best_model"],
                "final_internal_calibration": final_calibration,
                "resource_usage": result["resource_usage"],
                "elapsed_seconds": result["elapsed_seconds"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
