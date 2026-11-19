#!/usr/bin/env python3
"""Create a space-efficient nnU-Net v2 BraTS-PEDs dataset using symlinks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


MODALITIES = {
    "0000": ("t1n", "T1"),
    "0001": ("t1c", "T1ce"),
    "0002": ("t2w", "T2"),
    "0003": ("t2f", "FLAIR"),
}


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
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/nnunet"),
    )
    parser.add_argument("--dataset-id", type=int, default=501)
    parser.add_argument("--dataset-name", default="BraTSPEDs")
    parser.add_argument(
        "--link-mode",
        choices=("symlink", "hardlink"),
        default="symlink",
    )
    parser.add_argument(
        "--skip-blind",
        action="store_true",
        help=(
            "Skip imagesTs links for the blind validation cases (used while "
            "their transfer is still in flight; rerun without this flag "
            "later, the script is idempotent)."
        ),
    )
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def safe_link(source: Path, destination: Path, mode: str) -> str:
    source = source.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if destination.is_symlink():
        if destination.resolve() == source:
            return "existing"
        raise FileExistsError(
            f"Symlink target mismatch: {destination} -> "
            f"{destination.resolve()}, expected {source}"
        )
    if destination.exists():
        try:
            if os.path.samefile(source, destination):
                return "existing"
        except OSError:
            pass
        raise FileExistsError(
            f"Refusing to replace existing file: {destination}"
        )
    if mode == "symlink":
        destination.symlink_to(source)
    else:
        os.link(source, destination)
    return "created"


def pediatric_rows(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = manifest.loc[manifest["dataset"].eq("brats_peds")].copy()
    rows = rows.loc[rows["missing_files"].fillna("").eq("")]
    rows = rows.loc[
        rows["geometry_ok"].astype(str).str.lower().eq("true")
    ]
    return rows.set_index("case_id", drop=False)


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    split_definition = json.loads(args.splits.read_text(encoding="utf-8"))
    rows = pediatric_rows(manifest)
    dataset_folder = f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    raw_root = args.root / "raw"
    preprocessed_root = args.root / "preprocessed"
    results_root = args.root / "results"
    raw_dataset = raw_root / dataset_folder
    images_tr = raw_dataset / "imagesTr"
    labels_tr = raw_dataset / "labelsTr"
    images_ts = raw_dataset / "imagesTs"
    preprocessed_dataset = preprocessed_root / dataset_folder
    for path in (
        images_tr,
        labels_tr,
        images_ts,
        preprocessed_dataset,
        results_root,
    ):
        path.mkdir(parents=True, exist_ok=True)

    labeled_ids = split_definition["pediatric_labeled_case_ids"]
    blind_ids = split_definition["pediatric_blind_validation_case_ids"]
    created = 0
    existing = 0
    for case_id in labeled_ids:
        row = rows.loc[case_id]
        for suffix, (column, _) in MODALITIES.items():
            state = safe_link(
                Path(str(row[column])),
                images_tr / f"{case_id}_{suffix}.nii.gz",
                args.link_mode,
            )
            created += int(state == "created")
            existing += int(state == "existing")
        state = safe_link(
            Path(str(row["seg"])),
            labels_tr / f"{case_id}.nii.gz",
            args.link_mode,
        )
        created += int(state == "created")
        existing += int(state == "existing")

    linked_blind_ids = [] if args.skip_blind else blind_ids
    for case_id in linked_blind_ids:
        row = rows.loc[case_id]
        for suffix, (column, _) in MODALITIES.items():
            state = safe_link(
                Path(str(row[column])),
                images_ts / f"{case_id}_{suffix}.nii.gz",
                args.link_mode,
            )
            created += int(state == "created")
            existing += int(state == "existing")

    dataset_json = {
        "channel_names": {
            suffix[-1]: modality_name
            for suffix, (_, modality_name) in MODALITIES.items()
        },
        "labels": {
            "background": 0,
            "ET": 1,
            "NET": 2,
            "CC": 3,
            "ED": 4,
        },
        "numTraining": len(labeled_ids),
        "file_ending": ".nii.gz",
    }
    write_json(raw_dataset / "dataset.json", dataset_json)

    nnunet_splits = [
        {
            "train": fold["train"],
            "val": fold["val"],
        }
        for fold in sorted(
            split_definition["outer_folds"], key=lambda item: item["fold"]
        )
    ]
    split_path = preprocessed_dataset / "splits_final.json"
    write_json(split_path, nnunet_splits)
    write_json(args.root / f"splits_final_{args.dataset_id:03d}.json", nnunet_splits)

    expected_training_images = len(labeled_ids) * len(MODALITIES)
    expected_test_images = len(linked_blind_ids) * len(MODALITIES)
    summary = {
        "status": "ready",
        "link_mode": args.link_mode,
        "dataset_folder": dataset_folder,
        "paths": {
            "nnUNet_raw": str(raw_root.resolve()),
            "nnUNet_preprocessed": str(preprocessed_root.resolve()),
            "nnUNet_results": str(results_root.resolve()),
            "raw_dataset": str(raw_dataset.resolve()),
            "splits_final": str(split_path.resolve()),
        },
        "counts": {
            "labeled_cases": len(labeled_ids),
            "blind_validation_cases": len(blind_ids),
            "training_images": len(list(images_tr.glob("*.nii.gz"))),
            "training_labels": len(list(labels_tr.glob("*.nii.gz"))),
            "blind_images": len(list(images_ts.glob("*.nii.gz"))),
            "expected_training_images": expected_training_images,
            "expected_training_labels": len(labeled_ids),
            "expected_blind_images": expected_test_images,
            "created_links": created,
            "existing_links": existing,
        },
        "folds": [
            {
                "fold": index,
                "train_cases": len(fold["train"]),
                "validation_cases": len(fold["val"]),
            }
            for index, fold in enumerate(nnunet_splits)
        ],
    }
    count_keys = (
        ("training_images", "expected_training_images"),
        ("training_labels", "expected_training_labels"),
        ("blind_images", "expected_blind_images"),
    )
    mismatches = [
        f"{actual}={summary['counts'][actual]}, "
        f"{expected}={summary['counts'][expected]}"
        for actual, expected in count_keys
        if summary["counts"][actual] != summary["counts"][expected]
    ]
    if mismatches:
        summary["status"] = "failed"
        summary["count_mismatches"] = mismatches
    write_json(args.root / "prepare_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if mismatches:
        raise SystemExit("nnU-Net dataset count audit failed.")


if __name__ == "__main__":
    main()
