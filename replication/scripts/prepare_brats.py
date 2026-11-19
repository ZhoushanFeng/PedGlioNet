#!/usr/bin/env python3
"""Build one leakage-aware manifest for pediatric and adult BraTS NIfTI data.

The script does not duplicate the large image volumes. It records absolute paths,
checks modality geometry, and defines the common ET/TC/WT targets used by the
adult-to-pediatric transfer pilot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
import pandas as pd


PED_SUFFIXES = {
    "t1n": "-t1n.nii.gz",
    "t1c": "-t1c.nii.gz",
    "t2w": "-t2w.nii.gz",
    "t2f": "-t2f.nii.gz",
    "seg": "-seg.nii.gz",
}

ADULT_SUFFIXES = {
    "t1n": "_t1.nii.gz",
    "t1c": "_t1ce.nii.gz",
    "t2w": "_t2.nii.gz",
    "t2f": "_flair.nii.gz",
    "seg": "_seg.nii.gz",
}

LABEL_SCHEMAS = {
    "brats_peds": {
        "native": {"0": "background", "1": "ET", "2": "NET", "3": "CC", "4": "ED"},
        "common_regions": {
            "ET": [1],
            "TC": [1, 2, 3],
            "WT": [1, 2, 3, 4],
        },
    },
    "brats_adult_2021": {
        "native": {"0": "background", "1": "NCR/NET", "2": "ED", "4": "ET"},
        "common_regions": {"ET": [4], "TC": [1, 4], "WT": [1, 2, 4]},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--peds-root", type=Path, default=Path("data/raw/brats_peds")
    )
    parser.add_argument(
        "--adult-root", type=Path, default=Path("data/raw/brats_adult_2021")
    )
    parser.add_argument(
        "--peds-metadata",
        type=Path,
        default=Path("data/raw/brats_peds/BraTS-PEDs_metadata.tsv"),
    )
    parser.add_argument(
        "--adult-mapping",
        type=Path,
        default=Path("data/raw/brats_adult_2021/BraTS2021_MappingToTCIA.xlsx"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/manifest.csv")
    )
    parser.add_argument(
        "--inspect-labels",
        action="store_true",
        help="Read every segmentation volume and verify native label values.",
    )
    parser.add_argument(
        "--datasets",
        choices=("both", "peds", "adult"),
        default="both",
        help=(
            "Which cohorts to scan. peds-only lets pediatric training start "
            "before the adult data sync has finished."
        ),
    )
    return parser.parse_args()


def strip_suffix(name: str, suffix: str) -> str:
    if not name.endswith(suffix):
        raise ValueError(f"{name!r} does not end with {suffix!r}")
    return name[: -len(suffix)]


def paths_for_case(
    anchor_path: Path, suffixes: dict[str, str], anchor_key: str
) -> dict[str, Path]:
    case_id = strip_suffix(anchor_path.name, suffixes[anchor_key])
    paths = {
        key: anchor_path.parent / f"{case_id}{suffix}"
        for key, suffix in suffixes.items()
    }
    return paths


def geometry_status(paths: Iterable[Path]) -> tuple[bool, str, str]:
    images = [nib.load(str(path)) for path in paths]
    reference = images[0]
    shapes = [tuple(int(value) for value in image.shape) for image in images]
    affine_ok = all(
        np.allclose(image.affine, reference.affine, rtol=0.0, atol=1e-4)
        for image in images[1:]
    )
    shape_ok = all(shape == shapes[0] for shape in shapes[1:])
    return shape_ok and affine_ok, json.dumps(shapes), str(reference.header.get_zooms()[:3])


def discover(
    root: Path,
    dataset: str,
    suffixes: dict[str, str],
    metadata: pd.DataFrame | None,
    metadata_id_col: str | None,
    inspect_labels: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    metadata_by_id = (
        metadata.set_index(metadata_id_col).to_dict(orient="index")
        if metadata is not None and metadata_id_col is not None
        else {}
    )

    anchor_key = "t1n"
    for anchor_path in sorted(root.rglob(f"*{suffixes[anchor_key]}")):
        case_paths = paths_for_case(anchor_path.resolve(), suffixes, anchor_key)
        case_id = strip_suffix(anchor_path.name, suffixes[anchor_key])
        meta = metadata_by_id.get(case_id, {})
        if dataset == "brats_peds":
            split = str(meta.get("BraTS2025_cohort", "training"))
            source = meta.get("Source", "")
            organization = meta.get("Organization", "")
            site_id = ""
            age_days = meta.get("Age at imaging (days)", "")
            sex = meta.get("Sex_at_birth", "")
            os_days = meta.get("Overall survival (days)", "")
        else:
            split = str(meta.get("Segmentation (Task 1) Cohort", "Training"))
            source = meta.get("Data Collection (as on TCIA+additional)", "")
            organization = ""
            site_id = meta.get("Site ID", "")
            age_days = ""
            sex = ""
            os_days = ""
        required_keys = ["t1n", "t1c", "t2w", "t2f"]
        if dataset == "brats_adult_2021" or split.lower() == "training":
            required_keys.append("seg")
        missing = [key for key in required_keys if not case_paths[key].exists()]
        geometry_ok = False
        shapes = ""
        spacing = ""
        if not missing:
            geometry_paths = [case_paths[key] for key in required_keys]
            geometry_ok, shapes, spacing = geometry_status(geometry_paths)

        native_labels = ""
        labels_ok: bool | str = ""
        if inspect_labels and not missing and case_paths["seg"].exists():
            values = np.unique(np.asanyarray(nib.load(str(case_paths["seg"])).dataobj))
            native_labels = ",".join(str(int(value)) for value in values)
            expected = {int(value) for value in LABEL_SCHEMAS[dataset]["native"]}
            labels_ok = set(int(value) for value in values).issubset(expected)

        row: dict[str, object] = {
            "dataset": dataset,
            "case_id": case_id,
            "split": split,
            "source": source,
            "organization": organization,
            "site_id": site_id,
            "age_days": age_days,
            "sex": sex,
            "os_days": os_days,
            "event_status": "",
            "t1n": str(case_paths["t1n"]),
            "t1c": str(case_paths["t1c"]),
            "t2w": str(case_paths["t2w"]),
            "t2f": str(case_paths["t2f"]),
            "seg": str(case_paths["seg"]) if case_paths["seg"].exists() else "",
            "missing_files": ",".join(missing),
            "geometry_ok": geometry_ok,
            "shapes": shapes,
            "spacing": spacing,
            "native_labels": native_labels,
            "labels_ok": labels_ok,
        }
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    metadata = pd.read_csv(args.peds_metadata, sep="\t")
    adult_mapping = (
        pd.read_excel(args.adult_mapping)
        if args.adult_mapping.exists()
        else None
    )
    rows: list[dict[str, object]] = []
    if args.datasets in ("both", "peds"):
        rows.extend(
            discover(
                args.peds_root,
                "brats_peds",
                PED_SUFFIXES,
                metadata,
                "BraTS-SubjectID",
                args.inspect_labels,
            )
        )
    if args.datasets in ("both", "adult"):
        rows.extend(
            discover(
                args.adult_root,
                "brats_adult_2021",
                ADULT_SUFFIXES,
                adult_mapping,
                "BraTS2021 ID",
                args.inspect_labels,
            )
        )

    manifest = pd.DataFrame(rows)
    if not manifest.empty:
        manifest["_is_incomplete"] = manifest["missing_files"].ne("")
        manifest = (
            manifest.sort_values(
                ["dataset", "case_id", "_is_incomplete"],
                kind="stable",
            )
            .drop_duplicates(["dataset", "case_id"], keep="first")
            .drop(columns="_is_incomplete")
            .reset_index(drop=True)
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False)
    schema_path = args.output.with_name("label_schema.json")
    schema_path.write_text(
        json.dumps(LABEL_SCHEMAS, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    summary = {
        "manifest": str(args.output),
        "cases": int(len(manifest)),
        "by_dataset": (
            manifest["dataset"].value_counts().to_dict() if len(manifest) else {}
        ),
        "complete_cases": (
            int((manifest["missing_files"] == "").sum()) if len(manifest) else 0
        ),
        "geometry_valid_cases": (
            int(manifest["geometry_ok"].fillna(False).sum()) if len(manifest) else 0
        ),
        "labeled_cases": (
            int(manifest["seg"].astype(str).ne("").sum()) if len(manifest) else 0
        ),
        "labels_checked_ok": (
            int(
                manifest["labels_ok"]
                .astype(str)
                .str.lower()
                .eq("true")
                .sum()
            )
            if len(manifest)
            else 0
        ),
        "by_dataset_split": (
            {
                f"{dataset}/{split}": int(count)
                for (dataset, split), count in manifest.groupby(
                    ["dataset", "split"], dropna=False
                ).size().items()
            }
            if len(manifest)
            else {}
        ),
    }
    qa_path = args.output.with_name("qa_summary.json")
    qa_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if manifest.empty:
        raise SystemExit(
            "No segmentation files were found. Extract/download the NIfTI packages first."
        )


if __name__ == "__main__":
    main()
