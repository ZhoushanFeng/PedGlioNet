#!/usr/bin/env python3
"""Training-only ET-presence gate for the native-resolution ET refiner.

The gate is a regularized logistic model fitted to frozen refiner features.
Its probability threshold is selected from stratified out-of-fold predictions
on CBTN/DFCI only. DIPGr labels are used once for final evaluation.
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

from et_small_target_refinement import (
    ETRefinerUNet3D,
    aggregate_method,
    case_metrics,
    extract_patch,
    load_native_case,
    mask_center,
    paired_difference,
    paste_patch,
    patch_spec,
)
from full_transfer_train import (
    MemoryTracker,
    nested_prediction,
    write_json,
)
from pilot_transfer import TinyUNet3D, complete_rows, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/processed/manifest.csv")
    )
    parser.add_argument(
        "--refiner-model",
        type=Path,
        default=Path(
            "results/checkpoints/et_small_target/et_refiner_model.pt"
        ),
    )
    parser.add_argument(
        "--coarse-model",
        type=Path,
        default=Path(
            "results/checkpoints/full_transfer/pediatric_only_model.pt"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/et_presence_gated_refinement.json"),
    )
    parser.add_argument(
        "--status-output",
        type=Path,
        default=Path("results/et_presence_gate_status.json"),
    )
    parser.add_argument(
        "--gate-model",
        type=Path,
        default=Path(
            "results/checkpoints/et_small_target/et_presence_gate.pt"
        ),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--classifier-steps", type=int, default=1000)
    parser.add_argument("--classifier-learning-rate", type=float, default=0.05)
    parser.add_argument("--classifier-weight-decay", type=float, default=1e-2)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
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
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--eval-limit", type=int, default=0)
    return parser.parse_args()


class PresenceGate(nn.Module):
    def __init__(self, feature_count: int) -> None:
        super().__init__()
        self.linear = nn.Linear(feature_count, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features).squeeze(1)


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


@torch.inference_mode()
def refiner_features_and_prediction(
    model: ETRefinerUNet3D,
    image: torch.Tensor,
    center_mask: torch.Tensor,
    patch_shape: tuple[int, int, int],
    device: torch.device,
) -> tuple[np.ndarray, torch.Tensor]:
    spec = patch_spec(
        mask_center(center_mask.bool()),
        tuple(image.shape[1:]),
        patch_shape,
    )
    image_patch = extract_patch(image, spec).unsqueeze(0).to(device)
    logits, pooled = model.forward_with_features(image_patch)
    probability = torch.sigmoid(logits).flatten()
    top_count = min(2048, probability.numel())
    top_mean = probability.topk(top_count).values.mean()
    statistics = torch.stack(
        (
            probability.max(),
            probability.mean(),
            probability.std(),
            (probability >= 0.25).float().mean(),
            (probability >= 0.50).float().mean(),
            (probability >= 0.75).float().mean(),
            top_mean,
        )
    )
    features = torch.cat((pooled.squeeze(0), statistics)).cpu().numpy()
    prediction_patch = (probability >= 0.5).reshape(patch_shape).cpu()
    prediction = paste_patch(
        prediction_patch, spec, tuple(image.shape[1:])
    )
    return features.astype(np.float32), prediction


@torch.inference_mode()
def coarse_prediction(
    model: TinyUNet3D,
    image: torch.Tensor,
    coarse_shape: tuple[int, int, int],
    device: torch.device,
) -> torch.Tensor:
    resized = torch.nn.functional.interpolate(
        image.unsqueeze(0),
        size=coarse_shape,
        mode="trilinear",
        align_corners=False,
    ).to(device)
    prediction = nested_prediction(torch.sigmoid(model(resized))).float()
    return (
        torch.nn.functional.interpolate(
            prediction,
            size=tuple(image.shape[1:]),
            mode="nearest",
        )
        .squeeze(0)
        .bool()
        .cpu()
    )


def extract_training_features(
    rows: pd.DataFrame,
    refiner: ETRefinerUNet3D,
    patch_shape: tuple[int, int, int],
    device: torch.device,
    status_path: Path,
    started_at: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    case_ids: list[str] = []
    total = len(rows)
    refiner.to(device).eval()
    for position, (_, row) in enumerate(rows.iterrows(), 1):
        native = load_native_case(row)
        vector, _ = refiner_features_and_prediction(
            refiner,
            native["image"],
            native["target"][2],
            patch_shape,
            device,
        )
        features.append(vector)
        labels.append(int(native["target"][0].any()))
        case_ids.append(native["case_id"])
        if position % 25 == 0 or position == total:
            detail = f"training gate features: {position}/{total}"
            print(detail, flush=True)
            update_status(
                status_path,
                "feature_extraction",
                detail,
                started_at,
                completed=position,
                total=total,
            )
    return (
        np.stack(features).astype(np.float32),
        np.asarray(labels, dtype=np.float32),
        case_ids,
    )


def stratified_folds(
    labels: np.ndarray, folds: int, seed: int
) -> list[np.ndarray]:
    generator = np.random.default_rng(seed)
    assignments: list[list[int]] = [[] for _ in range(folds)]
    for label in (0, 1):
        indices = np.flatnonzero(labels == label)
        generator.shuffle(indices)
        for position, index in enumerate(indices):
            assignments[position % folds].append(int(index))
    return [np.asarray(sorted(fold), dtype=np.int64) for fold in assignments]


def fit_gate(
    features: np.ndarray,
    labels: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> tuple[PresenceGate, np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    mean = features.mean(axis=0)
    std = np.maximum(features.std(axis=0), 1e-6)
    standardized = torch.from_numpy((features - mean) / std).float()
    target = torch.from_numpy(labels).float()
    model = PresenceGate(features.shape[1])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.classifier_learning_rate,
        weight_decay=args.classifier_weight_decay,
    )
    positives = max(float(target.sum()), 1.0)
    negatives = max(float(len(target) - target.sum()), 1.0)
    positive_weight = torch.tensor(negatives / positives)
    for _ in range(args.classifier_steps):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            model(standardized),
            target,
            pos_weight=positive_weight,
        )
        loss.backward()
        optimizer.step()
    return model.eval(), mean.astype(np.float32), std.astype(np.float32)


@torch.inference_mode()
def predict_gate(
    model: PresenceGate,
    features: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    standardized = torch.from_numpy((features - mean) / std).float()
    return torch.sigmoid(model(standardized)).numpy()


def choose_threshold(
    probabilities: np.ndarray, labels: np.ndarray
) -> tuple[float, dict[str, float]]:
    best: tuple[float, float, float, float] | None = None
    best_metrics: dict[str, float] = {}
    for threshold in np.linspace(0.05, 0.95, 91):
        prediction = probabilities >= threshold
        sensitivity = float(prediction[labels == 1].mean())
        specificity = float((~prediction[labels == 0]).mean())
        balanced_accuracy = (sensitivity + specificity) / 2.0
        ranking = (
            balanced_accuracy,
            specificity,
            -abs(float(threshold) - 0.5),
            float(threshold),
        )
        if best is None or ranking > best:
            best = ranking
            best_metrics = {
                "threshold": float(threshold),
                "sensitivity": sensitivity,
                "specificity": specificity,
                "balanced_accuracy": balanced_accuracy,
            }
    return best_metrics["threshold"], best_metrics


def out_of_fold_gate(
    features: np.ndarray,
    labels: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, float, dict[str, float]]:
    probabilities = np.zeros(len(labels), dtype=np.float32)
    all_indices = np.arange(len(labels))
    for fold_index, validation_indices in enumerate(
        stratified_folds(labels, args.folds, args.seed)
    ):
        training_indices = np.setdiff1d(
            all_indices, validation_indices, assume_unique=True
        )
        model, mean, std = fit_gate(
            features[training_indices],
            labels[training_indices],
            args,
            args.seed + fold_index,
        )
        probabilities[validation_indices] = predict_gate(
            model, features[validation_indices], mean, std
        )
    threshold, metrics = choose_threshold(probabilities, labels)
    return probabilities, threshold, metrics


def aggregate_presence(
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    prediction = probabilities >= threshold
    sensitivity = float(prediction[labels == 1].mean())
    specificity = float((~prediction[labels == 0]).mean())
    return {
        "positive_cases": int(labels.sum()),
        "negative_cases": int(len(labels) - labels.sum()),
        "threshold": threshold,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": (sensitivity + specificity) / 2.0,
    }


@torch.inference_mode()
def evaluate(
    rows: pd.DataFrame,
    refiner: ETRefinerUNet3D,
    coarse_model: TinyUNet3D,
    coarse_shape: tuple[int, int, int],
    patch_shape: tuple[int, int, int],
    gate: PresenceGate,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    threshold: float,
    device: torch.device,
    args: argparse.Namespace,
    tracker: MemoryTracker,
    started_at: float,
) -> dict[str, Any]:
    refiner.to(device).eval()
    coarse_model.to(device).eval()
    per_case: list[dict[str, Any]] = []
    total = len(rows)
    for position, (_, row) in enumerate(rows.iterrows(), 1):
        native = load_native_case(row)
        image = native["image"]
        target = native["target"][0].bool()
        coarse = coarse_prediction(
            coarse_model, image, coarse_shape, device
        )
        center_mask = coarse[2] if coarse[2].any() else coarse[1]
        features, ungated_prediction = refiner_features_and_prediction(
            refiner,
            image,
            center_mask,
            patch_shape,
            device,
        )
        probability = float(
            predict_gate(
                gate,
                features[None, :],
                feature_mean,
                feature_std,
            )[0]
        )
        gated_prediction = (
            ungated_prediction
            if probability >= threshold
            else torch.zeros_like(ungated_prediction)
        )
        per_case.append(
            {
                "case_id": native["case_id"],
                "target_et_present": bool(target.any()),
                "gate_probability": probability,
                "gate_positive": bool(probability >= threshold),
                "coarse_baseline": case_metrics(coarse[0], target),
                "ungated_refiner": case_metrics(
                    ungated_prediction, target
                ),
                "presence_gated_refiner": case_metrics(
                    gated_prediction, target
                ),
            }
        )
        if position % 10 == 0 or position == total:
            detail = f"gated ET evaluation: {position}/{total}"
            print(detail, flush=True)
            tracker.sample()
            update_status(
                args.status_output,
                "evaluation",
                detail,
                started_at,
                completed=position,
                total=total,
                resource_usage=tracker.as_dict(),
            )

    methods = (
        "coarse_baseline",
        "ungated_refiner",
        "presence_gated_refiner",
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
    labels = np.asarray(
        [case["target_et_present"] for case in per_case],
        dtype=np.float32,
    )
    probabilities = np.asarray(
        [case["gate_probability"] for case in per_case],
        dtype=np.float32,
    )
    aggregate["presence_gate"] = aggregate_presence(
        probabilities, labels, threshold
    )
    aggregate["paired_differences"] = {
        "gated_refiner_vs_coarse": paired_difference(
            per_case,
            "presence_gated_refiner",
            "coarse_baseline",
            args.bootstrap_samples,
            args.seed + 200,
        ),
        "gated_vs_ungated_refiner": paired_difference(
            per_case,
            "presence_gated_refiner",
            "ungated_refiner",
            args.bootstrap_samples,
            args.seed + 201,
        ),
    }
    return {"aggregate": aggregate, "per_case": per_case}


def main() -> None:
    args = parse_args()
    started_at = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = select_device(args.device, args.mps_memory_fraction)
    tracker = MemoryTracker(device)

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

    refiner_bundle = torch.load(
        args.refiner_model, map_location="cpu", weights_only=False
    )
    refiner_metadata = refiner_bundle["metadata"]
    patch_shape = tuple(refiner_metadata["patch_shape_dhw"])
    refiner = ETRefinerUNet3D(refiner_metadata["base_channels"])
    refiner.load_state_dict(refiner_bundle["model_state"])

    coarse_bundle = torch.load(
        args.coarse_model, map_location="cpu", weights_only=False
    )
    coarse_metadata = coarse_bundle["metadata"]
    coarse_shape = tuple(coarse_metadata["shape_dhw"])
    coarse_model = TinyUNet3D(coarse_metadata["base_channels"])
    coarse_model.load_state_dict(coarse_bundle["model_state"])

    config = {
        "device": str(device),
        "training_cases": len(train_rows),
        "evaluation_cases": len(eval_rows),
        "training_sources": args.peds_train_sources,
        "evaluation_sources": args.peds_eval_sources,
        "patch_shape_dhw": patch_shape,
        "coarse_shape_dhw": coarse_shape,
        "feature_policy": (
            "16 pooled frozen-refiner bottleneck features plus 7 "
            "segmentation-probability summary features"
        ),
        "threshold_policy": (
            f"{args.folds}-fold stratified out-of-fold balanced accuracy "
            "on training sources only"
        ),
    }
    print(json.dumps(config, indent=2, ensure_ascii=False), flush=True)
    update_status(
        args.status_output,
        "initializing",
        "Loaded frozen ET refiner and coarse localizer",
        started_at,
        config=config,
    )

    extraction_started = time.time()
    features, labels, training_case_ids = extract_training_features(
        train_rows,
        refiner,
        patch_shape,
        device,
        args.status_output,
        started_at,
    )
    feature_seconds = time.time() - extraction_started

    oof_probabilities, threshold, oof_metrics = out_of_fold_gate(
        features, labels, args
    )
    final_gate, feature_mean, feature_std = fit_gate(
        features, labels, args, args.seed + 100
    )
    args.gate_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": final_gate.state_dict(),
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "threshold": threshold,
            "metadata": config,
        },
        args.gate_model,
    )

    evaluation_started = time.time()
    evaluation = evaluate(
        eval_rows,
        refiner,
        coarse_model,
        coarse_shape,
        patch_shape,
        final_gate,
        feature_mean,
        feature_std,
        threshold,
        device,
        args,
        tracker,
        started_at,
    )
    evaluation_seconds = time.time() - evaluation_started
    result = {
        "status": "completed_training_only_et_presence_gate",
        "config": config,
        "training_gate": {
            "case_ids": training_case_ids,
            "labels": labels.astype(int).tolist(),
            "out_of_fold_probabilities": oof_probabilities.tolist(),
            "selected_threshold": threshold,
            "out_of_fold_metrics": oof_metrics,
        },
        "evaluation": evaluation,
        "timing_seconds": {
            "feature_extraction": feature_seconds,
            "evaluation": evaluation_seconds,
            "total": time.time() - started_at,
        },
        "resource_usage": {
            "device": str(device),
            "torch_threads": args.threads,
            **tracker.as_dict(),
        },
        "artifacts": {
            "gate_model": str(args.gate_model),
            "refiner_model": str(args.refiner_model),
        },
        "limitations": [
            (
                "The frozen refiner features were generated from cases used "
                "to fit that refiner; only the gate head is cross-fitted."
            ),
            (
                "Training ROIs use reference WT while DIPGr ROIs use coarse "
                "predicted WT."
            ),
            "The gate threshold was not tuned on DIPGr.",
        ],
    }
    write_json(args.output, result)
    write_json(
        args.status_output,
        {
            "state": "completed",
            "stage": "done",
            "detail": f"Results written to {args.output}",
            "elapsed_seconds": time.time() - started_at,
            "resource_usage": result["resource_usage"],
            "output": str(args.output),
        },
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "config": config,
                "training_gate": {
                    "selected_threshold": threshold,
                    "out_of_fold_metrics": oof_metrics,
                },
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
