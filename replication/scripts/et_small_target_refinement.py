#!/usr/bin/env python3
"""Two-stage native-resolution ET small-target refinement experiment.

The ET refiner is fitted only on CBTN and DFCI pediatric cases. During fitting,
the crop is centered on the reference WT. During held-out DIPGr inference, the
crop is centered only on WT predicted by the frozen pediatric-only coarse model.
An oracle-WT crop is reported strictly as an upper-bound diagnostic.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from full_transfer_train import (
    MemoryTracker,
    cpu_state_dict,
    move_optimizer_state,
    nested_prediction,
    save_checkpoint,
    write_json,
)
from pilot_transfer import (
    ConvBlock,
    TinyUNet3D,
    complete_rows,
    native_to_common,
    nonzero_bbox,
    select_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/processed/manifest.csv")
    )
    parser.add_argument(
        "--coarse-model",
        type=Path,
        default=Path(
            "results/checkpoints/full_transfer/pediatric_only_model.pt"
        ),
    )
    parser.add_argument(
        "--initial-model",
        type=Path,
        default=None,
        help="Optional trained ET refiner used to initialize a new run.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/et_small_target_refinement.json"),
    )
    parser.add_argument(
        "--status-output",
        type=Path,
        default=Path("results/et_small_target_status.json"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("results/checkpoints/et_small_target"),
    )
    parser.add_argument(
        "--patch-shape", type=int, nargs=3, default=(96, 112, 112)
    )
    parser.add_argument("--base-channels", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--positive-repeat",
        type=int,
        default=2,
        help="Number of appearances per ET-positive case in each epoch.",
    )
    parser.add_argument("--translation-voxels", type=int, default=8)
    parser.add_argument(
        "--empty-target-weight",
        type=float,
        default=0.0,
        help="Extra top-probability and BCE penalty for ET-negative cases.",
    )
    parser.add_argument(
        "--empty-target-topk",
        type=int,
        default=8192,
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--peds-train-sources",
        nargs="+",
        default=["CBTN", "DFCI-BCH-BWH-PEDs-HGG"],
    )
    parser.add_argument(
        "--peds-eval-sources", nargs="+", default=["DIPGr"]
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    parser.add_argument("--mps-memory-fraction", type=float, default=0.30)
    parser.add_argument("--inter-step-sleep-ms", type=float, default=10.0)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--eval-limit", type=int, default=0)
    return parser.parse_args()


class ETRefinerUNet3D(nn.Module):
    def __init__(self, base: int = 4) -> None:
        super().__init__()
        self.enc1 = ConvBlock(4, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.bottleneck = ConvBlock(base * 2, base * 4)
        self.dec2 = ConvBlock(base * 6, base * 2)
        self.dec1 = ConvBlock(base * 3, base)
        self.out = nn.Conv3d(base, 1, kernel_size=1)

    def forward_with_features(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        enc1 = self.enc1(image)
        enc2 = self.enc2(F.max_pool3d(enc1, kernel_size=2))
        bottleneck = self.bottleneck(
            F.max_pool3d(enc2, kernel_size=2)
        )
        up2 = F.interpolate(
            bottleneck,
            size=enc2.shape[2:],
            mode="trilinear",
            align_corners=False,
        )
        dec2 = self.dec2(torch.cat((up2, enc2), dim=1))
        up1 = F.interpolate(
            dec2,
            size=enc1.shape[2:],
            mode="trilinear",
            align_corners=False,
        )
        logits = self.out(self.dec1(torch.cat((up1, enc1), dim=1)))
        pooled_features = F.adaptive_avg_pool3d(
            bottleneck, output_size=1
        ).flatten(1)
        return logits, pooled_features

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_features(image)
        return logits


@dataclass(frozen=True)
class PatchSpec:
    source: tuple[slice, slice, slice]
    destination: tuple[slice, slice, slice]
    patch_shape: tuple[int, int, int]


def limited(rows: pd.DataFrame, limit: int) -> pd.DataFrame:
    return rows.iloc[:limit] if limit > 0 else rows


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


def load_native_case(row: pd.Series) -> dict[str, Any]:
    arrays = [
        np.asarray(nib.load(str(row[column])).dataobj, dtype=np.float32)
        for column in ("t1n", "t1c", "t2w", "t2f")
    ]
    image = np.stack(arrays, axis=0).transpose(0, 3, 2, 1)
    segmentation = np.asarray(
        nib.load(str(row["seg"])).dataobj, dtype=np.int16
    ).transpose(2, 1, 0)
    crop = nonzero_bbox(image)
    image = image[(slice(None),) + crop]
    segmentation = segmentation[crop]

    foreground = np.any(np.abs(image) > 1e-6, axis=0)
    for channel in range(image.shape[0]):
        values = image[channel][foreground]
        if values.size:
            mean = float(values.mean())
            std = max(float(values.std()), 1e-6)
            image[channel] = np.clip(
                (image[channel] - mean) / std, -5.0, 5.0
            )
            image[channel][~foreground] = 0.0

    target = native_to_common(segmentation, "brats_peds")
    return {
        "case_id": str(row["case_id"]),
        "image": torch.from_numpy(image).contiguous(),
        "target": torch.from_numpy(target).to(torch.uint8).contiguous(),
    }


def mask_center(mask: torch.Tensor) -> tuple[int, int, int]:
    coordinates = torch.nonzero(mask, as_tuple=False)
    if not len(coordinates):
        return tuple(int(size // 2) for size in mask.shape)
    lower = coordinates.min(dim=0).values
    upper = coordinates.max(dim=0).values
    return tuple(
        int(value)
        for value in torch.div(
            lower + upper, 2, rounding_mode="floor"
        ).tolist()
    )


def patch_spec(
    center: tuple[int, int, int],
    spatial_shape: tuple[int, int, int],
    patch_shape: tuple[int, int, int],
) -> PatchSpec:
    source: list[slice] = []
    destination: list[slice] = []
    for coordinate, size, requested in zip(
        center, spatial_shape, patch_shape
    ):
        nominal_start = coordinate - requested // 2
        nominal_end = nominal_start + requested
        source_start = max(nominal_start, 0)
        source_end = min(nominal_end, size)
        destination_start = source_start - nominal_start
        destination_end = destination_start + (source_end - source_start)
        source.append(slice(source_start, source_end))
        destination.append(slice(destination_start, destination_end))
    return PatchSpec(
        source=tuple(source),  # type: ignore[arg-type]
        destination=tuple(destination),  # type: ignore[arg-type]
        patch_shape=patch_shape,
    )


def extract_patch(tensor: torch.Tensor, spec: PatchSpec) -> torch.Tensor:
    channels = tensor.ndim == 4
    output_shape = (
        (tensor.shape[0],) + spec.patch_shape
        if channels
        else spec.patch_shape
    )
    output = torch.zeros(output_shape, dtype=tensor.dtype)
    if channels:
        output[(slice(None),) + spec.destination] = tensor[
            (slice(None),) + spec.source
        ]
    else:
        output[spec.destination] = tensor[spec.source]
    return output


def paste_patch(
    patch: torch.Tensor,
    spec: PatchSpec,
    spatial_shape: tuple[int, int, int],
) -> torch.Tensor:
    output = torch.zeros(spatial_shape, dtype=patch.dtype)
    output[spec.source] = patch[spec.destination]
    return output


def prepare_training_patches(
    rows: pd.DataFrame,
    patch_shape: tuple[int, int, int],
    status_path: Path,
    started_at: float,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    total = len(rows)
    stage_started = time.time()
    for position, (_, row) in enumerate(rows.iterrows(), 1):
        native = load_native_case(row)
        image: torch.Tensor = native["image"]
        target: torch.Tensor = native["target"]
        center = mask_center(target[2].bool())
        spec = patch_spec(center, tuple(target.shape[1:]), patch_shape)
        et_patch = extract_patch(target[0], spec)
        cases.append(
            {
                "case_id": native["case_id"],
                "image": extract_patch(image, spec).to(torch.float16),
                "target": et_patch.unsqueeze(0).to(torch.uint8),
                "et_positive": bool(et_patch.any()),
            }
        )
        if position == 1 or position % 25 == 0 or position == total:
            detail = f"training ROI cache: {position}/{total}"
            print(detail, flush=True)
            update_status(
                status_path,
                "preprocessing",
                detail,
                started_at,
                cached_cases=position,
                total_cases=total,
            )
    print(
        "training ROI cache completed in "
        f"{time.time() - stage_started:.1f}s",
        flush=True,
    )
    return cases


def translate_with_zeros(
    image: torch.Tensor,
    target: torch.Tensor,
    shifts: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    dimensions = (2, 3, 4)
    image = torch.roll(image, shifts=shifts, dims=dimensions)
    target = torch.roll(target, shifts=shifts, dims=dimensions)
    for dimension, shift in zip(dimensions, shifts):
        if shift == 0:
            continue
        index = [slice(None)] * image.ndim
        target_index = [slice(None)] * target.ndim
        if shift > 0:
            index[dimension] = slice(0, shift)
            target_index[dimension] = slice(0, shift)
        else:
            index[dimension] = slice(shift, None)
            target_index[dimension] = slice(shift, None)
        image[tuple(index)] = 0
        target[tuple(target_index)] = 0
    return image, target


def focal_tversky_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    empty_target_weight: float = 0.0,
    empty_target_topk: int = 8192,
) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    axes = tuple(range(2, logits.ndim))
    true_positive = (probability * target).sum(dim=axes)
    false_positive = (probability * (1.0 - target)).sum(dim=axes)
    false_negative = ((1.0 - probability) * target).sum(dim=axes)
    tversky = (true_positive + 1.0) / (
        true_positive + 0.3 * false_positive + 0.7 * false_negative + 1.0
    )

    bce = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none"
    )
    target_probability = probability * target + (1.0 - probability) * (
        1.0 - target
    )
    alpha = 0.75 * target + 0.25 * (1.0 - target)
    focal = (alpha * (1.0 - target_probability).pow(2.0) * bce).mean()
    loss = (1.0 - tversky).mean() + focal
    if empty_target_weight > 0:
        flattened = probability.flatten()
        top_count = min(max(empty_target_topk, 1), flattened.numel())
        false_positive_penalty = flattened.topk(top_count).values.mean()
        false_positive_penalty = false_positive_penalty + bce.mean()
        loss = loss + empty_target_weight * false_positive_penalty
    return loss


def train_refiner(
    model: nn.Module,
    cases: list[dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
    tracker: MemoryTracker,
    run_started_at: float,
) -> tuple[list[dict[str, float]], float]:
    checkpoint_path = args.checkpoint_dir / "latest.pt"
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    history: list[dict[str, float]] = []
    start_epoch = 0
    if args.resume and checkpoint_path.exists():
        saved = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        move_optimizer_state(optimizer, device)
        history = list(saved.get("history", []))
        start_epoch = int(saved["completed_epochs"])
        print(f"resumed ET refiner at epoch {start_epoch}", flush=True)

    positive_indices = [
        index for index, case in enumerate(cases) if case["et_positive"]
    ]
    base_indices = list(range(len(cases)))
    stage_started = time.time()
    sleep_seconds = max(args.inter_step_sleep_ms, 0.0) / 1000.0

    for epoch_index in range(start_epoch, args.epochs):
        epoch_started = time.time()
        generator = random.Random(args.seed + epoch_index)
        order = base_indices + positive_indices * max(
            args.positive_repeat - 1, 0
        )
        generator.shuffle(order)
        model.train()
        total_loss = 0.0
        for step_index, case_index in enumerate(order, 1):
            case = cases[case_index]
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
            for dimension in (2, 3, 4):
                if generator.random() < 0.5:
                    image = torch.flip(image, dims=(dimension,))
                    target = torch.flip(target, dims=(dimension,))
            maximum_shift = max(args.translation_voxels, 0)
            shifts = tuple(
                generator.randint(-maximum_shift, maximum_shift)
                if maximum_shift
                else 0
                for _ in range(3)
            )
            image, target = translate_with_zeros(image, target, shifts)
            if generator.random() < 0.3:
                scale = 0.9 + 0.2 * generator.random()
                shift = 0.1 * (generator.random() - 0.5)
                image = image * scale + shift

            optimizer.zero_grad(set_to_none=True)
            loss = focal_tversky_loss(
                model(image),
                target,
                (
                    args.empty_target_weight
                    if not case["et_positive"]
                    else 0.0
                ),
                args.empty_target_topk,
            )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            if step_index % 50 == 0:
                tracker.sample()
            if sleep_seconds:
                time.sleep(sleep_seconds)

        if device.type == "mps":
            torch.mps.synchronize()
        epoch_seconds = time.time() - epoch_started
        mean_loss = total_loss / len(order)
        history.append(
            {
                "epoch": float(epoch_index + 1),
                "mean_loss": mean_loss,
                "seconds": epoch_seconds,
                "updates": float(len(order)),
            }
        )
        save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            epoch_index + 1,
            history,
            "et_small_target_refinement",
        )
        detail = (
            f"ET refiner epoch {epoch_index + 1}/{args.epochs}, "
            f"loss={mean_loss:.4f}, {epoch_seconds:.1f}s"
        )
        print(detail, flush=True)
        update_status(
            args.status_output,
            "training",
            detail,
            run_started_at,
            completed_epochs=epoch_index + 1,
            total_epochs=args.epochs,
            resource_usage=tracker.as_dict(),
        )
    return history, time.time() - stage_started


@torch.inference_mode()
def coarse_native_prediction(
    model: nn.Module,
    image: torch.Tensor,
    coarse_shape: tuple[int, int, int],
    device: torch.device,
) -> torch.Tensor:
    resized = F.interpolate(
        image.unsqueeze(0),
        size=coarse_shape,
        mode="trilinear",
        align_corners=False,
    ).to(device)
    prediction = nested_prediction(torch.sigmoid(model(resized))).float()
    native = F.interpolate(
        prediction,
        size=tuple(image.shape[1:]),
        mode="nearest",
    )
    return native.squeeze(0).bool().cpu()


@torch.inference_mode()
def refine_from_center(
    model: nn.Module,
    image: torch.Tensor,
    center_mask: torch.Tensor,
    patch_shape: tuple[int, int, int],
    device: torch.device,
) -> tuple[torch.Tensor, PatchSpec]:
    center = mask_center(center_mask.bool())
    spec = patch_spec(center, tuple(image.shape[1:]), patch_shape)
    image_patch = extract_patch(image, spec).unsqueeze(0).to(device)
    prediction_patch = (
        torch.sigmoid(model(image_patch)) >= 0.5
    ).squeeze(0).squeeze(0).cpu()
    return paste_patch(
        prediction_patch, spec, tuple(image.shape[1:])
    ), spec


def case_metrics(
    prediction: torch.Tensor, target: torch.Tensor
) -> dict[str, Any]:
    prediction = prediction.bool()
    target = target.bool()
    intersection = int((prediction & target).sum())
    prediction_voxels = int(prediction.sum())
    target_voxels = int(target.sum())
    denominator = prediction_voxels + target_voxels
    dice = 2.0 * intersection / denominator if denominator else 1.0
    return {
        "dice": float(dice),
        "intersection_voxels": intersection,
        "prediction_voxels": prediction_voxels,
        "target_voxels": target_voxels,
        "overlap_detected": bool(intersection > 0),
    }


def bootstrap_mean(
    values: list[float], samples: int, seed: int
) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "ci95_low": None, "ci95_high": None}
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0, len(array), size=(samples, len(array)), endpoint=False
    )
    means = array[indices].mean(axis=1)
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def aggregate_method(
    cases: list[dict[str, Any]],
    method: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    positive = [
        case[method]["dice"]
        for case in cases
        if case[method]["target_voxels"] > 0
    ]
    negative = [
        case[method]
        for case in cases
        if case[method]["target_voxels"] == 0
    ]
    positive_cases = [
        case[method]
        for case in cases
        if case[method]["target_voxels"] > 0
    ]
    return {
        "hard_dice_reference_positive": bootstrap_mean(
            positive, samples, seed
        ),
        "lesion_overlap_sensitivity": (
            float(
                np.mean(
                    [case["overlap_detected"] for case in positive_cases]
                )
            )
            if positive_cases
            else None
        ),
        "negative_case_specificity": (
            float(
                np.mean(
                    [case["prediction_voxels"] == 0 for case in negative]
                )
            )
            if negative
            else None
        ),
        "positive_cases": len(positive_cases),
        "negative_cases": len(negative),
    }


def paired_difference(
    cases: list[dict[str, Any]],
    first: str,
    second: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    values = np.asarray(
        [
            case[first]["dice"] - case[second]["dice"]
            for case in cases
            if case[first]["target_voxels"] > 0
        ],
        dtype=np.float64,
    )
    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0, len(values), size=(samples, len(values)), endpoint=False
    )
    means = values[indices].mean(axis=1)
    return {
        "contrast": f"{first} minus {second}",
        "n": int(len(values)),
        "mean_difference": float(values.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


@torch.inference_mode()
def evaluate(
    refiner: nn.Module,
    coarse_model: nn.Module,
    rows: pd.DataFrame,
    patch_shape: tuple[int, int, int],
    coarse_shape: tuple[int, int, int],
    device: torch.device,
    args: argparse.Namespace,
    tracker: MemoryTracker,
    run_started_at: float,
) -> dict[str, Any]:
    refiner.to(device).eval()
    coarse_model.to(device).eval()
    per_case: list[dict[str, Any]] = []
    total = len(rows)
    for position, (_, row) in enumerate(rows.iterrows(), 1):
        native = load_native_case(row)
        image: torch.Tensor = native["image"]
        target: torch.Tensor = native["target"]
        coarse = coarse_native_prediction(
            coarse_model, image, coarse_shape, device
        )
        predicted_center_mask = coarse[2]
        if not predicted_center_mask.any():
            predicted_center_mask = coarse[1]
        predicted_refinement, predicted_spec = refine_from_center(
            refiner,
            image,
            predicted_center_mask,
            patch_shape,
            device,
        )
        oracle_refinement, oracle_spec = refine_from_center(
            refiner,
            image,
            target[2],
            patch_shape,
            device,
        )
        et_target = target[0].bool()
        target_voxels = max(int(et_target.sum()), 1)
        predicted_roi_coverage = float(
            et_target[predicted_spec.source].sum() / target_voxels
        )
        oracle_roi_coverage = float(
            et_target[oracle_spec.source].sum() / target_voxels
        )
        per_case.append(
            {
                "case_id": native["case_id"],
                "coarse_baseline": case_metrics(coarse[0], et_target),
                "predicted_wt_roi_refiner": case_metrics(
                    predicted_refinement, et_target
                ),
                "oracle_wt_roi_refiner": case_metrics(
                    oracle_refinement, et_target
                ),
                "predicted_roi_et_coverage": predicted_roi_coverage,
                "oracle_roi_et_coverage": oracle_roi_coverage,
            }
        )
        if position % 10 == 0 or position == total:
            detail = f"native ET evaluation: {position}/{total}"
            print(detail, flush=True)
            tracker.sample()
            update_status(
                args.status_output,
                "evaluation",
                detail,
                run_started_at,
                evaluated_cases=position,
                total_cases=total,
                resource_usage=tracker.as_dict(),
            )

    methods = (
        "coarse_baseline",
        "predicted_wt_roi_refiner",
        "oracle_wt_roi_refiner",
    )
    aggregate = {
        method: aggregate_method(
            per_case,
            method,
            args.bootstrap_samples,
            args.seed + 100 + index,
        )
        for index, method in enumerate(methods)
    }
    positive_cases = [
        case
        for case in per_case
        if case["coarse_baseline"]["target_voxels"] > 0
    ]
    aggregate["roi_coverage"] = {
        "predicted_wt_roi_mean_et_coverage": float(
            np.mean(
                [case["predicted_roi_et_coverage"] for case in positive_cases]
            )
        ),
        "predicted_wt_roi_full_coverage_rate": float(
            np.mean(
                [
                    case["predicted_roi_et_coverage"] >= 0.999
                    for case in positive_cases
                ]
            )
        ),
        "oracle_wt_roi_mean_et_coverage": float(
            np.mean(
                [case["oracle_roi_et_coverage"] for case in positive_cases]
            )
        ),
    }
    aggregate["paired_differences"] = {
        "predicted_refiner_vs_coarse": paired_difference(
            per_case,
            "predicted_wt_roi_refiner",
            "coarse_baseline",
            args.bootstrap_samples,
            args.seed + 200,
        ),
        "oracle_refiner_vs_coarse": paired_difference(
            per_case,
            "oracle_wt_roi_refiner",
            "coarse_baseline",
            args.bootstrap_samples,
            args.seed + 201,
        ),
    }
    return {"aggregate": aggregate, "per_case": per_case}


def main() -> None:
    args = parse_args()
    run_started_at = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = select_device(args.device, args.mps_memory_fraction)
    tracker = MemoryTracker(device)
    patch_shape = tuple(args.patch_shape)

    overlap = set(args.peds_train_sources) & set(args.peds_eval_sources)
    if overlap:
        raise SystemExit(f"Training and evaluation sources overlap: {overlap}")

    manifest = pd.read_csv(args.manifest)
    pediatric = complete_rows(manifest, "brats_peds")
    pediatric = pediatric.loc[
        pediatric["split"].astype(str).str.lower().eq("training")
        & pediatric["seg"].fillna("").ne("")
    ]
    train_rows = pediatric.loc[
        pediatric["source"].isin(args.peds_train_sources)
    ].sample(frac=1.0, random_state=args.seed)
    eval_rows = pediatric.loc[
        pediatric["source"].isin(args.peds_eval_sources)
    ].sample(frac=1.0, random_state=args.seed + 1)
    train_rows = limited(train_rows, args.train_limit)
    eval_rows = limited(eval_rows, args.eval_limit)
    if not len(train_rows) or not len(eval_rows):
        raise SystemExit("Training and evaluation rows must be non-empty.")

    config = {
        "device": str(device),
        "batch_size": 1,
        "mps_memory_fraction": args.mps_memory_fraction,
        "patch_shape_dhw": patch_shape,
        "native_voxel_spacing_mm": 1,
        "base_channels": args.base_channels,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "positive_repeat": args.positive_repeat,
        "translation_voxels": args.translation_voxels,
        "empty_target_weight": args.empty_target_weight,
        "empty_target_topk": args.empty_target_topk,
        "initial_model": (
            str(args.initial_model) if args.initial_model else None
        ),
        "training_cases": len(train_rows),
        "evaluation_cases": len(eval_rows),
        "training_sources": args.peds_train_sources,
        "evaluation_sources": args.peds_eval_sources,
        "coarse_model": str(args.coarse_model),
        "roi_policy": {
            "training": "reference WT center",
            "held_out_inference": "frozen coarse-model predicted WT center",
            "oracle_diagnostic": "reference WT center; not deployable",
        },
        "threshold": 0.5,
    }
    print(json.dumps(config, indent=2, ensure_ascii=False), flush=True)
    update_status(
        args.status_output,
        "initializing",
        "Validated source-held-out ET refinement split",
        run_started_at,
        config=config,
    )

    preprocessing_started = time.time()
    training_cases = prepare_training_patches(
        train_rows,
        patch_shape,
        args.status_output,
        run_started_at,
    )
    preprocessing_seconds = time.time() - preprocessing_started
    et_positive_training_cases = sum(
        int(case["et_positive"]) for case in training_cases
    )

    refiner = ETRefinerUNet3D(args.base_channels)
    if args.initial_model is not None:
        initial_bundle = torch.load(
            args.initial_model, map_location="cpu", weights_only=False
        )
        refiner.load_state_dict(initial_bundle["model_state"])
        print(f"initialized refiner from {args.initial_model}", flush=True)
    history, training_seconds_current_invocation = train_refiner(
        refiner,
        training_cases,
        args,
        device,
        tracker,
        run_started_at,
    )
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": cpu_state_dict(refiner),
            "metadata": {
                **config,
                "et_positive_training_cases": et_positive_training_cases,
            },
        },
        args.checkpoint_dir / "et_refiner_model.pt",
    )

    coarse_bundle = torch.load(
        args.coarse_model, map_location="cpu", weights_only=False
    )
    coarse_metadata = coarse_bundle["metadata"]
    coarse_model = TinyUNet3D(coarse_metadata["base_channels"])
    coarse_model.load_state_dict(coarse_bundle["model_state"])
    coarse_shape = tuple(coarse_metadata["shape_dhw"])
    evaluation_started = time.time()
    evaluation = evaluate(
        refiner,
        coarse_model,
        eval_rows,
        patch_shape,
        coarse_shape,
        device,
        args,
        tracker,
        run_started_at,
    )
    evaluation_seconds = time.time() - evaluation_started

    cumulative_training_seconds = float(
        sum(epoch["seconds"] for epoch in history)
    )
    result = {
        "status": "completed_native_resolution_et_refinement",
        "config": {
            **config,
            "coarse_shape_dhw": coarse_shape,
            "et_positive_training_cases": et_positive_training_cases,
        },
        "training_history": history,
        "evaluation": evaluation,
        "timing_seconds": {
            "current_invocation_preprocessing": preprocessing_seconds,
            "current_invocation_training": training_seconds_current_invocation,
            "current_invocation_evaluation": evaluation_seconds,
            "current_invocation_total": time.time() - run_started_at,
            "cumulative_epoch_training": cumulative_training_seconds,
        },
        "resource_usage": {
            "device": str(device),
            "torch_threads": args.threads,
            **tracker.as_dict(),
        },
        "artifacts": {
            "model": str(args.checkpoint_dir / "et_refiner_model.pt"),
            "latest_checkpoint": str(args.checkpoint_dir / "latest.pt"),
        },
        "limitations": [
            "Single fixed leave-source-out experiment.",
            "Training ROIs use reference WT while inference ROIs use predicted WT.",
            "Oracle-WT ROI results are diagnostic and must not be presented as deployable performance.",
            "No threshold or post-processing was tuned on DIPGr.",
        ],
    }
    write_json(args.output, result)
    write_json(
        args.status_output,
        {
            "state": "completed",
            "stage": "done",
            "detail": f"Results written to {args.output}",
            "current_invocation_elapsed_seconds": (
                time.time() - run_started_at
            ),
            "cumulative_epoch_training_seconds": cumulative_training_seconds,
            "resource_usage": result["resource_usage"],
            "output": str(args.output),
        },
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "config": result["config"],
                "aggregate": evaluation["aggregate"],
                "timing_seconds": result["timing_seconds"],
                "resource_usage": result["resource_usage"],
                "artifacts": result["artifacts"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
