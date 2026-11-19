#!/usr/bin/env python3
"""Create the nnU-Net v2 adult BraTS 2021 pretraining dataset (Dataset502).

Images are symlinked. Segmentation labels are rewritten because nnU-Net v2
requires consecutive integer labels: adult native {0,1,2,4} -> {0,1,2,3}
(ET moves from 4 to 3). Channel order matches Dataset501 exactly so that
pretrained encoder weights transfer cleanly.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd


MODALITIES = {
    "0000": ("t1n", "T1"),
    "0001": ("t1c", "T1ce"),
    "0002": ("t2w", "T2"),
    "0003": ("t2f", "FLAIR"),
}
ALLOWED_NATIVE_LABELS = {0, 1, 2, 4}


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
    parser.add_argument("--dataset-id", type=int, default=502)
    parser.add_argument("--dataset-name", default="BraTSAdult")
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def safe_link(source: Path, destination: Path) -> str:
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
        raise FileExistsError(f"Refusing to replace existing file: {destination}")
    destination.symlink_to(source)
    return "created"


def remap_label(task: tuple[str, str]) -> str:
    source, destination = (Path(task[0]), Path(task[1]))
    if destination.exists():
        return "existing"
    image = nib.load(str(source))
    data = np.asanyarray(image.dataobj).astype(np.uint8)
    values = {int(value) for value in np.unique(data)}
    if not values.issubset(ALLOWED_NATIVE_LABELS):
        raise ValueError(f"unexpected native labels {sorted(values)} in {source}")
    data[data == 4] = 3
    output = nib.Nifti1Image(data, image.affine, image.header)
    output.header.set_data_dtype(np.uint8)
    temporary = destination.with_name(destination.name + ".tmp.nii.gz")
    nib.save(output, str(temporary))
    temporary.rename(destination)
    return "created"


def adult_rows(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = manifest.loc[manifest["dataset"].eq("brats_adult_2021")].copy()
    rows = rows.loc[rows["missing_files"].fillna("").eq("")]
    rows = rows.loc[rows["geometry_ok"].astype(str).str.lower().eq("true")]
    rows = rows.loc[rows["labels_ok"].astype(str).str.lower().eq("true")]
    return rows.set_index("case_id", drop=False)


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    split_definition = json.loads(args.splits.read_text(encoding="utf-8"))
    rows = adult_rows(manifest)
    case_ids = split_definition["adult_pretraining_case_ids"]
    missing_cases = sorted(set(case_ids) - set(rows.index))
    if missing_cases:
        raise SystemExit(
            f"{len(missing_cases)} adult cases missing or failing QA, e.g. "
            f"{missing_cases[:5]}"
        )

    dataset_folder = f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    raw_dataset = args.root / "raw" / dataset_folder
    images_tr = raw_dataset / "imagesTr"
    labels_tr = raw_dataset / "labelsTr"
    for path in (images_tr, labels_tr):
        path.mkdir(parents=True, exist_ok=True)

    created_links = 0
    existing_links = 0
    label_tasks: list[tuple[str, str]] = []
    for case_id in case_ids:
        row = rows.loc[case_id]
        for suffix, (column, _) in MODALITIES.items():
            state = safe_link(
                Path(str(row[column])),
                images_tr / f"{case_id}_{suffix}.nii.gz",
            )
            created_links += int(state == "created")
            existing_links += int(state == "existing")
        label_tasks.append(
            (str(row["seg"]), str(labels_tr / f"{case_id}.nii.gz"))
        )

    with mp.Pool(processes=args.workers) as pool:
        states = pool.map(remap_label, label_tasks, chunksize=8)
    created_labels = sum(state == "created" for state in states)
    existing_labels = sum(state == "existing" for state in states)

    rng = np.random.default_rng(2026)
    sample_ids = rng.choice(case_ids, size=min(5, len(case_ids)), replace=False)
    for case_id in sample_ids:
        data = np.asanyarray(
            nib.load(str(labels_tr / f"{case_id}.nii.gz")).dataobj
        )
        values = {int(value) for value in np.unique(data)}
        if not values.issubset({0, 1, 2, 3}):
            raise SystemExit(
                f"remapped labels wrong for {case_id}: {sorted(values)}"
            )

    dataset_json = {
        "channel_names": {
            suffix[-1]: modality_name
            for suffix, (_, modality_name) in MODALITIES.items()
        },
        "labels": {
            "background": 0,
            "NCR_NET": 1,
            "ED": 2,
            "ET": 3,
        },
        "numTraining": len(case_ids),
        "file_ending": ".nii.gz",
    }
    write_json(raw_dataset / "dataset.json", dataset_json)

    summary = {
        "status": "ready",
        "dataset_folder": dataset_folder,
        "counts": {
            "cases": len(case_ids),
            "training_images": len(list(images_tr.glob("*.nii.gz"))),
            "training_labels": len(
                [
                    path
                    for path in labels_tr.glob("*.nii.gz")
                    if not path.name.endswith(".tmp.nii.gz")
                ]
            ),
            "expected_training_images": len(case_ids) * len(MODALITIES),
            "expected_training_labels": len(case_ids),
            "created_links": created_links,
            "existing_links": existing_links,
            "created_labels": created_labels,
            "existing_labels": existing_labels,
        },
        "label_remap": {"0": 0, "1": 1, "2": 2, "4": 3},
        "label_sample_checked": [str(case_id) for case_id in sample_ids],
    }
    counts = summary["counts"]
    mismatches = [
        key
        for key, expected in (
            ("training_images", "expected_training_images"),
            ("training_labels", "expected_training_labels"),
        )
        if counts[key] != counts[expected]
    ]
    if mismatches:
        summary["status"] = "failed"
        summary["count_mismatches"] = mismatches
    write_json(args.root / "prepare_adult_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if mismatches:
        raise SystemExit("Adult nnU-Net dataset count audit failed.")


if __name__ == "__main__":
    main()
