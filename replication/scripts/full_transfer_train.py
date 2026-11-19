#!/usr/bin/env python3
"""Resource-bounded full-cohort adult-to-pediatric transfer baseline.

The labeled pediatric cohort is divided by source:
* fitting: CBTN + DFCI-BCH-BWH-PEDs-HGG
* held-out evaluation: DIPGr

All Adult BraTS 2021 labeled cases are used for pretraining.  An optional
final model is then fitted on all labeled pediatric cases after held-out
evaluation.  That final all-pediatric fit has no unbiased performance estimate.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from pilot_transfer import (
    REGIONS,
    TinyUNet3D,
    complete_rows,
    dice_loss,
    load_case,
    select_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/processed/manifest.csv")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/full_transfer_crosssource.json"),
    )
    parser.add_argument(
        "--status-output",
        type=Path,
        default=Path("results/full_transfer_status.json"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("results/checkpoints/full_transfer"),
    )
    parser.add_argument("--shape", type=int, nargs=3, default=(64, 80, 80))
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--adult-epochs", type=int, default=5)
    parser.add_argument("--pediatric-epochs", type=int, default=30)
    parser.add_argument("--final-pediatric-epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--peds-train-sources",
        nargs="+",
        default=["CBTN", "DFCI-BCH-BWH-PEDs-HGG"],
    )
    parser.add_argument(
        "--peds-eval-sources",
        nargs="+",
        default=["DIPGr"],
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    parser.add_argument(
        "--mps-memory-fraction",
        type=float,
        default=0.30,
        help="Fraction of the recommended MPS working set available to this run.",
    )
    parser.add_argument(
        "--inter-step-sleep-ms",
        type=float,
        default=10.0,
        help="Small cooperative pause after each update to reduce system contention.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume each stage from its latest epoch checkpoint.",
    )
    parser.add_argument(
        "--fit-final-all-peds",
        action="store_true",
        help="After evaluation, fit an adult-initialized model on all labeled pediatric cases.",
    )
    parser.add_argument(
        "--adult-limit",
        type=int,
        default=0,
        help="Development-only limit; zero uses every adult case.",
    )
    parser.add_argument(
        "--peds-train-limit",
        type=int,
        default=0,
        help="Development-only limit; zero uses every pediatric fitting case.",
    )
    parser.add_argument(
        "--peds-eval-limit",
        type=int,
        default=0,
        help="Development-only limit; zero uses every held-out pediatric case.",
    )
    parser.add_argument(
        "--final-peds-limit",
        type=int,
        default=0,
        help="Development-only limit; zero uses every labeled pediatric case.",
    )
    return parser.parse_args()


def limited(rows: pd.DataFrame, limit: int) -> pd.DataFrame:
    return rows.iloc[:limit] if limit > 0 else rows


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_ready(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


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


class MemoryTracker:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.peak_current_mb = 0.0
        self.peak_driver_mb = 0.0

    def sample(self) -> None:
        if self.device.type != "mps":
            return
        current = torch.mps.current_allocated_memory() / (1024**2)
        driver = torch.mps.driver_allocated_memory() / (1024**2)
        self.peak_current_mb = max(self.peak_current_mb, current)
        self.peak_driver_mb = max(self.peak_driver_mb, driver)

    def as_dict(self) -> dict[str, float]:
        self.sample()
        return {
            "observed_peak_mps_current_allocated_mb": round(
                self.peak_current_mb, 2
            ),
            "observed_peak_mps_driver_allocated_mb": round(
                self.peak_driver_mb, 2
            ),
        }


def prepare_cases(
    rows: pd.DataFrame,
    shape: tuple[int, int, int],
    label: str,
    status_path: Path,
    started_at: float,
) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    total = len(rows)
    stage_started = time.time()
    for position, (_, row) in enumerate(rows.iterrows(), 1):
        case = load_case(row, shape)
        # Half precision images and byte targets keep the full CPU cache bounded.
        case["image"] = case["image"].to(dtype=torch.float16)  # type: ignore[union-attr]
        case["target"] = case["target"].to(dtype=torch.uint8)  # type: ignore[union-attr]
        cases.append(case)
        if position == 1 or position % 25 == 0 or position == total:
            message = f"{label}: {position}/{total} cases cached"
            print(message, flush=True)
            update_status(
                status_path,
                "preprocessing",
                message,
                started_at,
                cached_cases=position,
                total_cases=total,
            )
    print(
        f"{label}: preprocessing completed in {time.time() - stage_started:.1f}s",
        flush=True,
    )
    return cases


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def move_optimizer_state(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: list[dict[str, float]],
    stage: str,
) -> dict[str, Any]:
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    for state in optimizer_state["state"].values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.detach().cpu()
    return {
        "stage": stage,
        "completed_epochs": epoch,
        "model_state": cpu_state_dict(model),
        "optimizer_state": optimizer_state,
        "history": history,
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: list[dict[str, float]],
    stage: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        checkpoint_payload(model, optimizer, epoch, history, stage),
        temporary,
    )
    temporary.replace(path)


def train_stage(
    stage: str,
    model: nn.Module,
    cases: list[dict[str, object]],
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    seed: int,
    sleep_seconds: float,
    checkpoint_path: Path,
    resume: bool,
    tracker: MemoryTracker,
    status_path: Path,
    run_started_at: float,
) -> tuple[list[dict[str, float]], float]:
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    history: list[dict[str, float]] = []
    start_epoch = 0
    if resume and checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        move_optimizer_state(optimizer, device)
        history = list(saved.get("history", []))
        start_epoch = int(saved["completed_epochs"])
        print(
            f"{stage}: resumed after epoch {start_epoch}/{epochs}",
            flush=True,
        )

    stage_started = time.time()
    for epoch_index in range(start_epoch, epochs):
        epoch_started = time.time()
        model.train()
        order = list(range(len(cases)))
        generator = random.Random(seed + epoch_index)
        generator.shuffle(order)
        total_loss = 0.0

        for step_index, case_index in enumerate(order, 1):
            case = cases[case_index]
            image = (
                case["image"]
                .unsqueeze(0)
                .to(device=device, dtype=torch.float32)  # type: ignore[union-attr]
            )
            target = (
                case["target"]
                .unsqueeze(0)
                .to(device=device, dtype=torch.float32)  # type: ignore[union-attr]
            )
            for dimension in (2, 3, 4):
                if generator.random() < 0.5:
                    image = torch.flip(image, dims=(dimension,))
                    target = torch.flip(target, dims=(dimension,))
            if generator.random() < 0.3:
                scale = 0.9 + 0.2 * generator.random()
                shift = 0.1 * (generator.random() - 0.5)
                image = image * scale + shift

            optimizer.zero_grad(set_to_none=True)
            logits = model(image)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            loss = loss + dice_loss(logits, target)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())

            if step_index % 100 == 0:
                tracker.sample()
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

        if device.type == "mps":
            torch.mps.synchronize()
        epoch_seconds = time.time() - epoch_started
        mean_loss = total_loss / max(len(cases), 1)
        history.append(
            {
                "epoch": float(epoch_index + 1),
                "mean_loss": mean_loss,
                "seconds": epoch_seconds,
            }
        )
        save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            epoch_index + 1,
            history,
            stage,
        )
        detail = (
            f"{stage}: epoch {epoch_index + 1}/{epochs}, "
            f"loss={mean_loss:.4f}, {epoch_seconds:.1f}s"
        )
        print(detail, flush=True)
        update_status(
            status_path,
            stage,
            detail,
            run_started_at,
            completed_epochs=epoch_index + 1,
            total_epochs=epochs,
            mean_loss=mean_loss,
            resource_usage=tracker.as_dict(),
        )

    return history, time.time() - stage_started


def nested_prediction(probability: torch.Tensor) -> torch.Tensor:
    raw = probability >= 0.5
    et = raw[:, 0]
    tc = raw[:, 1] | et
    wt = raw[:, 2] | tc
    return torch.stack((et, tc, wt), dim=1)


def bootstrap_mean_ci(
    values: list[float], samples: int, seed: int
) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "ci95_low": None, "ci95_high": None}
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled_indices = rng.integers(
        0, len(array), size=(samples, len(array)), endpoint=False
    )
    means = array[sampled_indices].mean(axis=1)
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    cases: list[dict[str, object]],
    device: torch.device,
    bootstrap_samples: int,
    seed: int,
    tracker: MemoryTracker,
) -> dict[str, Any]:
    model.to(device)
    model.eval()
    per_case: list[dict[str, Any]] = []
    for position, case in enumerate(cases, 1):
        image = (
            case["image"]
            .unsqueeze(0)
            .to(device=device, dtype=torch.float32)  # type: ignore[union-attr]
        )
        target = (
            case["target"]
            .unsqueeze(0)
            .to(device=device, dtype=torch.float32)  # type: ignore[union-attr]
        )
        probability = torch.sigmoid(model(image))
        prediction = nested_prediction(probability)
        target_binary = target >= 0.5
        axes = (0, 2, 3, 4)
        intersection = (prediction & target_binary).sum(dim=axes).float()
        denominator = prediction.sum(dim=axes) + target_binary.sum(dim=axes)
        hard_dice = torch.where(
            denominator > 0,
            2.0 * intersection / denominator.clamp_min(1),
            torch.ones_like(intersection),
        )
        target_voxels = target_binary.sum(dim=axes)
        prediction_voxels = prediction.sum(dim=axes)
        per_case.append(
            {
                "case_id": str(case["case_id"]),
                "hard_dice": {
                    region: float(value)
                    for region, value in zip(REGIONS, hard_dice.cpu())
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
        if position % 25 == 0 or position == len(cases):
            tracker.sample()
            print(f"evaluation: {position}/{len(cases)} cases", flush=True)

    all_case = {}
    reference_positive = {}
    for region_index, region in enumerate(REGIONS):
        values = [result["hard_dice"][region] for result in per_case]
        positive_values = [
            result["hard_dice"][region]
            for result in per_case
            if result["target_voxels"][region] > 0
        ]
        all_case[region] = bootstrap_mean_ci(
            values, bootstrap_samples, seed + region_index
        )
        reference_positive[region] = bootstrap_mean_ci(
            positive_values,
            bootstrap_samples,
            seed + 100 + region_index,
        )
    return {
        "aggregate": {
            "hard_dice_all_cases": all_case,
            "hard_dice_reference_positive": reference_positive,
        },
        "per_case": per_case,
    }


def release_model(model: nn.Module, device: torch.device) -> None:
    model.to("cpu")
    del model
    if device.type == "mps":
        torch.mps.empty_cache()


def save_inference_model(
    path: Path,
    state: dict[str, torch.Tensor],
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": state, "metadata": metadata}, path)


def main() -> None:
    args = parse_args()
    run_started_at = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = select_device(args.device, args.mps_memory_fraction)
    tracker = MemoryTracker(device)
    shape = tuple(args.shape)

    overlap = set(args.peds_train_sources) & set(args.peds_eval_sources)
    if overlap:
        raise SystemExit(f"Pediatric fitting/evaluation sources overlap: {overlap}")

    manifest = pd.read_csv(args.manifest)
    adult_rows = complete_rows(manifest, "brats_adult_2021")
    adult_rows = adult_rows.loc[adult_rows["seg"].fillna("").ne("")]
    adult_rows = adult_rows.sample(frac=1.0, random_state=args.seed)

    peds_rows = complete_rows(manifest, "brats_peds")
    peds_rows = peds_rows.loc[
        peds_rows["split"].astype(str).str.lower().eq("training")
        & peds_rows["seg"].fillna("").ne("")
    ]
    peds_train_rows = peds_rows.loc[
        peds_rows["source"].isin(args.peds_train_sources)
    ].sample(frac=1.0, random_state=args.seed + 1)
    peds_eval_rows = peds_rows.loc[
        peds_rows["source"].isin(args.peds_eval_sources)
    ].sample(frac=1.0, random_state=args.seed + 2)
    all_peds_rows = peds_rows.sample(frac=1.0, random_state=args.seed + 3)

    adult_rows = limited(adult_rows, args.adult_limit)
    peds_train_rows = limited(peds_train_rows, args.peds_train_limit)
    peds_eval_rows = limited(peds_eval_rows, args.peds_eval_limit)
    all_peds_rows = limited(all_peds_rows, args.final_peds_limit)
    if min(len(adult_rows), len(peds_train_rows), len(peds_eval_rows)) == 0:
        raise SystemExit("Adult, pediatric fitting, and evaluation sets must be non-empty.")

    config = {
        "device": str(device),
        "mps_memory_fraction": args.mps_memory_fraction,
        "batch_size": 1,
        "shape_dhw": shape,
        "base_channels": args.base_channels,
        "adult_epochs": args.adult_epochs,
        "pediatric_epochs": args.pediatric_epochs,
        "final_pediatric_epochs": args.final_pediatric_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "inter_step_sleep_ms": args.inter_step_sleep_ms,
        "seed": args.seed,
        "threads": args.threads,
        "pediatric_train_sources": args.peds_train_sources,
        "pediatric_eval_sources": args.peds_eval_sources,
        "fit_final_all_peds": args.fit_final_all_peds,
        "adult_cases": len(adult_rows),
        "pediatric_fitting_cases": len(peds_train_rows),
        "pediatric_evaluation_cases": len(peds_eval_rows),
        "all_labeled_pediatric_cases": len(all_peds_rows),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
    }
    update_status(
        args.status_output,
        "initializing",
        "Validated leakage-safe source split",
        run_started_at,
        config=config,
    )
    print(json.dumps(config, indent=2, ensure_ascii=False), flush=True)

    preprocessing_started = time.time()
    adult_cases = prepare_cases(
        adult_rows,
        shape,
        "adult",
        args.status_output,
        run_started_at,
    )
    peds_train_cases = prepare_cases(
        peds_train_rows,
        shape,
        "pediatric fitting",
        args.status_output,
        run_started_at,
    )
    peds_eval_cases = prepare_cases(
        peds_eval_rows,
        shape,
        "pediatric held-out",
        args.status_output,
        run_started_at,
    )
    all_peds_cases: list[dict[str, object]] = []
    if args.fit_final_all_peds:
        pediatric_cache = {
            str(case["case_id"]): case
            for case in peds_train_cases + peds_eval_cases
        }
        uncached_rows = all_peds_rows.loc[
            ~all_peds_rows["case_id"].astype(str).isin(pediatric_cache)
        ]
        if len(uncached_rows):
            uncached_cases = prepare_cases(
                uncached_rows,
                shape,
                "remaining pediatric",
                args.status_output,
                run_started_at,
            )
            pediatric_cache.update(
                {str(case["case_id"]): case for case in uncached_cases}
            )
        all_peds_cases = [
            pediatric_cache[str(case_id)]
            for case_id in all_peds_rows["case_id"].astype(str)
        ]
        print(
            f"all pediatric: reused {len(all_peds_cases)} cached cases",
            flush=True,
        )
    preprocessing_seconds = time.time() - preprocessing_started

    checkpoint_dir = args.checkpoint_dir
    sleep_seconds = max(args.inter_step_sleep_ms, 0.0) / 1000.0
    stage_times: dict[str, float] = {}

    adult_model = TinyUNet3D(args.base_channels)
    adult_history, stage_times["adult_pretraining"] = train_stage(
        "adult_pretraining",
        adult_model,
        adult_cases,
        args.adult_epochs,
        args.learning_rate,
        args.weight_decay,
        device,
        args.seed + 10,
        sleep_seconds,
        checkpoint_dir / "adult_pretraining_latest.pt",
        args.resume,
        tracker,
        args.status_output,
        run_started_at,
    )
    adult_state = cpu_state_dict(adult_model)
    zero_shot = evaluate(
        adult_model,
        peds_eval_cases,
        device,
        args.bootstrap_samples,
        args.seed + 20,
        tracker,
    )
    release_model(adult_model, device)

    fine_tuned_model = TinyUNet3D(args.base_channels)
    fine_tuned_model.load_state_dict(adult_state)
    fine_tune_history, stage_times["pediatric_finetuning"] = train_stage(
        "pediatric_finetuning",
        fine_tuned_model,
        peds_train_cases,
        args.pediatric_epochs,
        args.learning_rate,
        args.weight_decay,
        device,
        args.seed + 30,
        sleep_seconds,
        checkpoint_dir / "pediatric_finetuning_latest.pt",
        args.resume,
        tracker,
        args.status_output,
        run_started_at,
    )
    fine_tuned = evaluate(
        fine_tuned_model,
        peds_eval_cases,
        device,
        args.bootstrap_samples,
        args.seed + 40,
        tracker,
    )
    fine_tuned_state = cpu_state_dict(fine_tuned_model)
    save_inference_model(
        checkpoint_dir / "pediatric_finetuned_model.pt",
        fine_tuned_state,
        config,
    )
    release_model(fine_tuned_model, device)

    pediatric_model = TinyUNet3D(args.base_channels)
    pediatric_history, stage_times["pediatric_only"] = train_stage(
        "pediatric_only",
        pediatric_model,
        peds_train_cases,
        args.pediatric_epochs,
        args.learning_rate,
        args.weight_decay,
        device,
        args.seed + 50,
        sleep_seconds,
        checkpoint_dir / "pediatric_only_latest.pt",
        args.resume,
        tracker,
        args.status_output,
        run_started_at,
    )
    pediatric_only = evaluate(
        pediatric_model,
        peds_eval_cases,
        device,
        args.bootstrap_samples,
        args.seed + 60,
        tracker,
    )
    save_inference_model(
        checkpoint_dir / "pediatric_only_model.pt",
        cpu_state_dict(pediatric_model),
        config,
    )
    release_model(pediatric_model, device)

    final_history: list[dict[str, float]] | None = None
    if args.fit_final_all_peds:
        final_model = TinyUNet3D(args.base_channels)
        final_model.load_state_dict(adult_state)
        final_history, stage_times["final_all_pediatric_fit"] = train_stage(
            "final_all_pediatric_fit",
            final_model,
            all_peds_cases,
            args.final_pediatric_epochs,
            args.learning_rate,
            args.weight_decay,
            device,
            args.seed + 70,
            sleep_seconds,
            checkpoint_dir / "final_all_pediatric_fit_latest.pt",
            args.resume,
            tracker,
            args.status_output,
            run_started_at,
        )
        save_inference_model(
            checkpoint_dir / "final_all_pediatric_model.pt",
            cpu_state_dict(final_model),
            {
                **config,
                "warning": (
                    "Fitted on all labeled pediatric cases; no held-out estimate "
                    "is attached to this final model."
                ),
            },
        )
        release_model(final_model, device)

    elapsed_seconds = time.time() - run_started_at
    histories = {
        "adult_pretraining": adult_history,
        "pediatric_finetuning": fine_tune_history,
        "pediatric_only": pediatric_history,
        "final_all_pediatric_fit": final_history,
    }
    cumulative_stage_seconds = {
        stage: float(sum(epoch["seconds"] for epoch in (history or [])))
        for stage, history in histories.items()
    }
    result = {
        "status": "full_cohort_resource_bounded_baseline",
        "inference_scope": (
            "Held-out DIPGr metrics are valid for this fixed configuration; "
            "the optional all-pediatric final model has no unbiased test metric."
        ),
        "metric_space": (
            "Dice was measured on the configured reduced-resolution DHW grid, "
            "not in native-resolution physical space."
        ),
        "config": config,
        "case_ids": {
            "adult_pretraining": [case["case_id"] for case in adult_cases],
            "pediatric_fitting": [
                case["case_id"] for case in peds_train_cases
            ],
            "pediatric_held_out": [
                case["case_id"] for case in peds_eval_cases
            ],
            "final_all_pediatric": [
                case["case_id"] for case in all_peds_cases
            ],
        },
        "training": histories,
        "experiments": {
            "adult_zero_shot": zero_shot,
            "adult_initialized_pediatric_finetune": fine_tuned,
            "pediatric_only": pediatric_only,
        },
        "timing_seconds": {
            "current_invocation_preprocessing": preprocessing_seconds,
            "current_invocation_stage_work": stage_times,
            "current_invocation_total": elapsed_seconds,
            "cumulative_epoch_training_by_stage": cumulative_stage_seconds,
            "cumulative_epoch_training_total": float(
                sum(cumulative_stage_seconds.values())
            ),
        },
        "resource_usage": {
            "torch_threads": args.threads,
            "device": str(device),
            **tracker.as_dict(),
        },
        "artifacts": {
            "checkpoint_dir": str(checkpoint_dir),
            "held_out_finetuned_model": str(
                checkpoint_dir / "pediatric_finetuned_model.pt"
            ),
            "pediatric_only_model": str(
                checkpoint_dir / "pediatric_only_model.pt"
            ),
            "final_all_pediatric_model": (
                str(checkpoint_dir / "final_all_pediatric_model.pt")
                if args.fit_final_all_peds
                else None
            ),
        },
    }
    write_json(args.output, result)
    write_json(
        args.status_output,
        {
            "state": "completed",
            "stage": "done",
            "detail": f"Results written to {args.output}",
            "current_invocation_elapsed_seconds": elapsed_seconds,
            "cumulative_epoch_training_seconds": float(
                sum(cumulative_stage_seconds.values())
            ),
            "resource_usage": result["resource_usage"],
            "output": str(args.output),
        },
    )
    compact = {
        "status": result["status"],
        "config": config,
        "aggregate": {
            name: experiment["aggregate"]
            for name, experiment in result["experiments"].items()
        },
        "timing_seconds": result["timing_seconds"],
        "resource_usage": result["resource_usage"],
        "artifacts": result["artifacts"],
    }
    print(json.dumps(compact, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
