#!/usr/bin/env python3
"""Low-resource adult-to-pediatric BraTS transfer smoke experiment.

This is deliberately a small feasibility run, not a manuscript-quality model.
It trains a tiny 3D U-Net at reduced resolution and compares:
1) adult-trained zero-shot pediatric inference,
2) adult initialization followed by pediatric fine-tuning,
3) pediatric-only training.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


REGIONS = ("ET", "TC", "WT")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/processed/manifest.csv")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/pilot_transfer.json")
    )
    parser.add_argument("--train-cases", type=int, default=8)
    parser.add_argument("--eval-cases", type=int, default=3)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--finetune-steps", type=int, default=6)
    parser.add_argument("--shape", type=int, nargs=3, default=(48, 64, 64))
    parser.add_argument("--base-channels", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--peds-train-sources",
        nargs="+",
        default=None,
        help="Optional source names used for pediatric fitting.",
    )
    parser.add_argument(
        "--peds-eval-sources",
        nargs="+",
        default=None,
        help="Optional disjoint source names used for pediatric evaluation.",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    parser.add_argument(
        "--mps-memory-fraction",
        type=float,
        default=0.30,
        help="Fraction of the recommended MPS working-set size available to this run.",
    )
    return parser.parse_args()


def select_device(requested: str, memory_fraction: float) -> torch.device:
    if requested == "cuda" or (
        requested == "auto" and torch.cuda.is_available()
    ):
        if not torch.cuda.is_available():
            raise SystemExit("CUDA was requested but is unavailable.")
        return torch.device("cuda")
    use_mps = requested == "mps" or (
        requested == "auto" and torch.backends.mps.is_available()
    )
    if use_mps:
        device = torch.device("mps")
        try:
            torch.mps.set_per_process_memory_fraction(memory_fraction)
        except (AttributeError, RuntimeError):
            pass
        return device
    return torch.device("cpu")


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = min(4, out_channels)
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TinyUNet3D(nn.Module):
    def __init__(self, base: int = 4) -> None:
        super().__init__()
        self.enc1 = ConvBlock(4, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.bottleneck = ConvBlock(base * 2, base * 4)
        self.dec2 = ConvBlock(base * 6, base * 2)
        self.dec1 = ConvBlock(base * 3, base)
        self.out = nn.Conv3d(base, 3, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool3d(e1, kernel_size=2))
        bottleneck = self.bottleneck(F.max_pool3d(e2, kernel_size=2))
        up2 = F.interpolate(
            bottleneck, size=e2.shape[2:], mode="trilinear", align_corners=False
        )
        d2 = self.dec2(torch.cat((up2, e2), dim=1))
        up1 = F.interpolate(
            d2, size=e1.shape[2:], mode="trilinear", align_corners=False
        )
        return self.out(self.dec1(torch.cat((up1, e1), dim=1)))


def native_to_common(segmentation: np.ndarray, dataset: str) -> np.ndarray:
    if dataset == "brats_peds":
        et = segmentation == 1
        tc = np.isin(segmentation, (1, 2, 3))
        wt = np.isin(segmentation, (1, 2, 3, 4))
    elif dataset == "brats_adult_2021":
        et = segmentation == 4
        tc = np.isin(segmentation, (1, 4))
        wt = np.isin(segmentation, (1, 2, 4))
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return np.stack((et, tc, wt), axis=0).astype(np.float32)


def nonzero_bbox(images: np.ndarray, margin: int = 4) -> tuple[slice, slice, slice]:
    foreground = np.any(np.abs(images) > 1e-6, axis=0)
    coordinates = np.argwhere(foreground)
    if coordinates.size == 0:
        return tuple(slice(0, size) for size in foreground.shape)  # type: ignore[return-value]
    lower = np.maximum(coordinates.min(axis=0) - margin, 0)
    upper = np.minimum(coordinates.max(axis=0) + margin + 1, foreground.shape)
    return tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))  # type: ignore[return-value]


def load_case(row: pd.Series, shape: tuple[int, int, int]) -> dict[str, object]:
    arrays = [
        np.asarray(nib.load(str(row[column])).dataobj, dtype=np.float32)
        for column in ("t1n", "t1c", "t2w", "t2f")
    ]
    image = np.stack(arrays, axis=0)
    segmentation = np.asarray(
        nib.load(str(row["seg"])).dataobj, dtype=np.int16
    )

    # NIfTI uses X,Y,Z; PyTorch convolutions use D,H,W.
    image = image.transpose(0, 3, 2, 1)
    segmentation = segmentation.transpose(2, 1, 0)
    crop = nonzero_bbox(image)
    image = image[(slice(None),) + crop]
    segmentation = segmentation[crop]

    foreground = np.any(np.abs(image) > 1e-6, axis=0)
    for channel in range(image.shape[0]):
        values = image[channel][foreground]
        if values.size:
            mean = float(values.mean())
            std = max(float(values.std()), 1e-6)
            image[channel] = np.clip((image[channel] - mean) / std, -5.0, 5.0)
            image[channel][~foreground] = 0.0

    target = native_to_common(segmentation, str(row["dataset"]))
    image_tensor = F.interpolate(
        torch.from_numpy(image).unsqueeze(0),
        size=shape,
        mode="trilinear",
        align_corners=False,
    ).squeeze(0)
    target_tensor = F.interpolate(
        torch.from_numpy(target).unsqueeze(0), size=shape, mode="nearest"
    ).squeeze(0)
    return {
        "case_id": str(row["case_id"]),
        "image": image_tensor.contiguous(),
        "target": target_tensor.contiguous(),
    }


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    axes = tuple(range(2, probabilities.ndim))
    intersection = (probabilities * target).sum(dim=axes)
    denominator = probabilities.sum(dim=axes) + target.sum(dim=axes)
    return 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def train(
    model: nn.Module,
    cases: list[dict[str, object]],
    steps: int,
    learning_rate: float,
    device: torch.device,
    seed: int,
) -> list[float]:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    generator = random.Random(seed)
    losses: list[float] = []
    for step in range(steps):
        case = cases[step % len(cases)]
        image = case["image"].unsqueeze(0).to(device)  # type: ignore[union-attr]
        target = case["target"].unsqueeze(0).to(device)  # type: ignore[union-attr]
        for dimension in (2, 3, 4):
            if generator.random() < 0.5:
                image = torch.flip(image, dims=(dimension,))
                target = torch.flip(target, dims=(dimension,))

        optimizer.zero_grad(set_to_none=True)
        logits = model(image)
        loss = F.binary_cross_entropy_with_logits(logits, target) + dice_loss(
            logits, target
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return losses


@torch.inference_mode()
def evaluate(
    model: nn.Module, cases: list[dict[str, object]], device: torch.device
) -> dict[str, object]:
    model.eval()
    per_case: list[dict[str, object]] = []
    for case in cases:
        image = case["image"].unsqueeze(0).to(device)  # type: ignore[union-attr]
        target = case["target"].unsqueeze(0).to(device)  # type: ignore[union-attr]
        probability = torch.sigmoid(model(image))
        prediction = probability >= 0.5
        target_binary = target >= 0.5
        axes = (0, 2, 3, 4)
        intersection = (prediction & target_binary).sum(dim=axes).float()
        denominator = prediction.sum(dim=axes) + target_binary.sum(dim=axes)
        hard_dice = torch.where(
            denominator > 0,
            2.0 * intersection / denominator.clamp_min(1),
            torch.ones_like(intersection),
        )
        soft_intersection = (probability * target).sum(dim=axes)
        soft_denominator = probability.sum(dim=axes) + target.sum(dim=axes)
        soft_dice = (2.0 * soft_intersection + 1.0) / (
            soft_denominator + 1.0
        )
        target_voxels = target_binary.sum(dim=axes)
        prediction_voxels = prediction.sum(dim=axes)
        per_case.append(
            {
                "case_id": case["case_id"],
                "hard_dice": {
                    region: float(value)
                    for region, value in zip(REGIONS, hard_dice.cpu())
                },
                "soft_dice": {
                    region: float(value)
                    for region, value in zip(REGIONS, soft_dice.cpu())
                },
                "target_voxels": {
                    region: int(value)
                    for region, value in zip(REGIONS, target_voxels.cpu())
                },
                "prediction_voxels": {
                    region: int(value)
                    for region, value in zip(REGIONS, prediction_voxels.cpu())
                },
            }
        )

    aggregate = {}
    for metric in ("hard_dice", "soft_dice"):
        aggregate[metric] = {
            region: float(
                np.mean([case_result[metric][region] for case_result in per_case])
            )
            for region in REGIONS
        }
    aggregate["reference_positive_cases"] = {
        region: sum(
            int(case_result["target_voxels"][region] > 0)
            for case_result in per_case
        )
        for region in REGIONS
    }
    aggregate["hard_dice_reference_positive"] = {}
    for region in REGIONS:
        positive_values = [
            case_result["hard_dice"][region]
            for case_result in per_case
            if case_result["target_voxels"][region] > 0
        ]
        aggregate["hard_dice_reference_positive"][region] = (
            float(np.mean(positive_values)) if positive_values else None
        )
    return {"aggregate": aggregate, "per_case": per_case}


def complete_rows(manifest: pd.DataFrame, dataset: str) -> pd.DataFrame:
    rows = manifest.loc[manifest["dataset"] == dataset].copy()
    rows = rows.loc[rows["missing_files"].fillna("") == ""]
    rows = rows.loc[rows["geometry_ok"].astype(str).str.lower() == "true"]
    return rows


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = select_device(args.device, args.mps_memory_fraction)

    manifest = pd.read_csv(args.manifest)
    adult_rows = complete_rows(manifest, "brats_adult_2021").sample(
        frac=1.0, random_state=args.seed
    )
    peds_rows = complete_rows(manifest, "brats_peds")
    peds_rows = peds_rows.loc[
        peds_rows["split"].astype(str).str.lower() == "training"
    ]
    if args.peds_train_sources or args.peds_eval_sources:
        if not (args.peds_train_sources and args.peds_eval_sources):
            raise SystemExit(
                "Provide both --peds-train-sources and --peds-eval-sources."
            )
        overlap = set(args.peds_train_sources) & set(args.peds_eval_sources)
        if overlap:
            raise SystemExit(f"Training and evaluation sources overlap: {overlap}")
        peds_train_rows = peds_rows.loc[
            peds_rows["source"].isin(args.peds_train_sources)
        ].sample(frac=1.0, random_state=args.seed)
        peds_eval_rows = peds_rows.loc[
            peds_rows["source"].isin(args.peds_eval_sources)
        ].sample(frac=1.0, random_state=args.seed + 1)
    else:
        peds_rows = peds_rows.sample(frac=1.0, random_state=args.seed)
        peds_train_rows = peds_rows.iloc[: args.train_cases]
        peds_eval_rows = peds_rows.iloc[
            args.train_cases : args.train_cases + args.eval_cases
        ]

    if (
        len(adult_rows) < args.train_cases
        or len(peds_train_rows) < args.train_cases
        or len(peds_eval_rows) < args.eval_cases
    ):
        raise SystemExit(
            f"Need adult/train-pediatric/eval-pediatric counts of at least "
            f"{args.train_cases}/{args.train_cases}/{args.eval_cases}; found "
            f"{len(adult_rows)}/{len(peds_train_rows)}/{len(peds_eval_rows)}."
        )

    shape = tuple(args.shape)
    adult_train = [
        load_case(row, shape)
        for _, row in adult_rows.iloc[: args.train_cases].iterrows()
    ]
    peds_train = [
        load_case(row, shape)
        for _, row in peds_train_rows.iloc[: args.train_cases].iterrows()
    ]
    peds_eval = [
        load_case(row, shape)
        for _, row in peds_eval_rows.iloc[: args.eval_cases].iterrows()
    ]

    started = time.time()
    adult_model = TinyUNet3D(args.base_channels).to(device)
    adult_loss = train(
        adult_model,
        adult_train,
        args.steps,
        args.learning_rate,
        device,
        args.seed,
    )
    zero_shot = evaluate(adult_model, peds_eval, device)

    fine_tuned_model = copy.deepcopy(adult_model)
    fine_tune_loss = train(
        fine_tuned_model,
        peds_train,
        args.finetune_steps,
        args.learning_rate,
        device,
        args.seed + 1,
    )
    fine_tuned = evaluate(fine_tuned_model, peds_eval, device)

    pediatric_model = TinyUNet3D(args.base_channels).to(device)
    pediatric_loss = train(
        pediatric_model,
        peds_train,
        args.steps,
        args.learning_rate,
        device,
        args.seed + 2,
    )
    pediatric_only = evaluate(pediatric_model, peds_eval, device)

    resource_usage: dict[str, float | int | str] = {
        "torch_threads": args.threads,
        "device": str(device),
    }
    if device.type == "mps":
        torch.mps.synchronize()
        resource_usage.update(
            {
                "mps_current_allocated_mb": round(
                    torch.mps.current_allocated_memory() / (1024**2), 2
                ),
                "mps_driver_allocated_mb": round(
                    torch.mps.driver_allocated_memory() / (1024**2), 2
                ),
            }
        )

    result = {
        "status": "smoke_test_not_for_manuscript_inference",
        "device": str(device),
        "mps_memory_fraction": args.mps_memory_fraction,
        "batch_size": 1,
        "shape_dhw": shape,
        "base_channels": args.base_channels,
        "adult_train_cases": [case["case_id"] for case in adult_train],
        "pediatric_train_cases": [case["case_id"] for case in peds_train],
        "pediatric_eval_cases": [case["case_id"] for case in peds_eval],
        "pediatric_train_sources": args.peds_train_sources,
        "pediatric_eval_sources": args.peds_eval_sources,
        "steps": args.steps,
        "finetune_steps": args.finetune_steps,
        "losses": {
            "adult": adult_loss,
            "fine_tune": fine_tune_loss,
            "pediatric_only": pediatric_loss,
        },
        "experiments": {
            "adult_zero_shot": zero_shot,
            "adult_initialized_pediatric_finetune": fine_tuned,
            "pediatric_only": pediatric_only,
        },
        "resource_usage_at_end": resource_usage,
        "elapsed_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
