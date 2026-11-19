#!/usr/bin/env python3
"""Run one reproducible nnU-Net fold/seed with resume support."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-folder", default="Dataset501_BraTSPEDs")
    parser.add_argument("--plans", default="nnUNetResEncUNetMPlans")
    parser.add_argument("--configuration", default="3d_fullres")
    parser.add_argument(
        "--fold", required=True, choices=("0", "1", "2", "all")
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--device", choices=("cuda", "mps", "cpu"), default="cuda"
    )
    parser.add_argument("--data-processes", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-probabilities", action="store_true")
    parser.add_argument(
        "--trainer-suffix",
        default="",
        help=(
            "Appended to the trainer class name so variant runs (e.g. adult-"
            "initialised fine-tuning) get their own output folders."
        ),
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Override nnU-Net's default 1000 epochs (used for pretraining).",
    )
    parser.add_argument(
        "--pretrained-weights",
        type=Path,
        default=None,
        help="Checkpoint to initialise from (ignored when resuming).",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip perform_actual_validation (pretraining has no held-out set).",
    )
    parser.add_argument(
        "--status-dir",
        type=Path,
        default=Path("results/nnunet_runs"),
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    started_at = time.time()
    os.environ["nnUNet_n_proc_DA"] = str(args.data_processes)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    from batchgenerators.utilities.file_and_folder_operations import load_json
    from nnunetv2.paths import nnUNet_preprocessed
    from nnunetv2.run.run_training import maybe_load_checkpoint
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS was requested but is unavailable.")

    fold: int | str = args.fold if args.fold == "all" else int(args.fold)
    trainer_name = (
        f"nnUNetTrainerPedGlioSeed{args.seed}{args.trainer_suffix}"
    )
    trainer_class = type(trainer_name, (nnUNetTrainer,), {})
    dataset_dir = (
        Path(str(nnUNet_preprocessed)) / args.dataset_folder
    )
    plans = load_json(str(dataset_dir / f"{args.plans}.json"))
    dataset_json = load_json(str(dataset_dir / "dataset.json"))
    if fold == "all":
        train_cases = int(dataset_json["numTraining"])
        validation_cases = 0
    else:
        splits = load_json(str(dataset_dir / "splits_final.json"))
        train_cases = len(splits[fold]["train"])
        validation_cases = len(splits[fold]["val"])
    plans["continue_training"] = args.resume
    config = {
        "dataset_folder": args.dataset_folder,
        "plans": args.plans,
        "configuration": args.configuration,
        "fold": args.fold,
        "seed": args.seed,
        "trainer": trainer_name,
        "device": args.device,
        "data_processes": args.data_processes,
        "resume": args.resume,
        "save_probabilities": args.save_probabilities,
        "num_epochs": args.num_epochs,
        "pretrained_weights": (
            str(args.pretrained_weights) if args.pretrained_weights else None
        ),
        "skip_validation": args.skip_validation,
        "train_cases": train_cases,
        "validation_cases": validation_cases,
    }
    status_path = (
        args.status_dir
        / f"fold{args.fold}_seed{args.seed}{args.trainer_suffix}.json"
    )
    write_json(status_path, {"status": "running", "config": config})
    trainer = trainer_class(
        plans=plans,
        configuration=args.configuration,
        fold=fold,
        dataset_json=dataset_json,
        device=device,
    )
    if args.num_epochs is not None:
        trainer.num_epochs = args.num_epochs
    checkpoint_exists = any(
        (Path(trainer.output_folder) / name).is_file()
        for name in (
            "checkpoint_final.pth",
            "checkpoint_latest.pth",
            "checkpoint_best.pth",
        )
    )
    continue_training = args.resume and checkpoint_exists
    pretrained_weights = (
        None if continue_training else args.pretrained_weights
    )
    if args.pretrained_weights and not continue_training:
        if not args.pretrained_weights.is_file():
            raise SystemExit(
                f"Pretrained weights not found: {args.pretrained_weights}"
            )
    try:
        maybe_load_checkpoint(
            trainer,
            continue_training=continue_training,
            validation_only=False,
            pretrained_weights_file=(
                str(pretrained_weights) if pretrained_weights else None
            ),
        )
        trainer.run_training()
        if not args.skip_validation:
            trainer.perform_actual_validation(args.save_probabilities)
    except Exception as error:
        write_json(
            status_path,
            {
                "status": "failed",
                "config": config,
                "error_type": type(error).__name__,
                "error": str(error),
                "elapsed_seconds": time.time() - started_at,
                "trainer_output_folder": trainer.output_folder,
            },
        )
        raise
    result = {
        "status": "completed",
        "config": config,
        "elapsed_seconds": time.time() - started_at,
        "trainer_output_folder": trainer.output_folder,
    }
    write_json(status_path, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
