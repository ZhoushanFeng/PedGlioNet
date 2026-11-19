#!/usr/bin/env python3
"""Build and audit frozen source-held-out BraTS-PEDs splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/manifest.csv"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/formal_experiment_v1.json"),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("FORMAL_EXPERIMENT_PROTOCOL_v1.md"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/formal_splits_v1.json"),
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path("results/formal_split_audit.json"),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ids(rows: pd.DataFrame) -> list[str]:
    return sorted(rows["case_id"].astype(str).tolist())


def list_hash(values: list[str]) -> str:
    payload = "\n".join(values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def complete_pediatric_rows(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = manifest.loc[manifest["dataset"].eq("brats_peds")].copy()
    rows = rows.loc[rows["missing_files"].fillna("").eq("")]
    rows = rows.loc[
        rows["geometry_ok"].astype(str).str.lower().eq("true")
    ]
    return rows


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = pd.read_csv(args.manifest)
    pediatric = complete_pediatric_rows(manifest)
    labeled = pediatric.loc[
        pediatric["split"].astype(str).str.lower().eq("training")
        & pediatric["seg"].fillna("").ne("")
    ].copy()
    blind_validation = pediatric.loc[
        pediatric["split"].astype(str).str.lower().eq("validation")
        & pediatric["seg"].fillna("").eq("")
    ].copy()
    adult = manifest.loc[
        manifest["dataset"].eq("brats_adult_2021")
        & manifest["missing_files"].fillna("").eq("")
        & manifest["geometry_ok"].astype(str).str.lower().eq("true")
        & manifest["seg"].fillna("").ne("")
    ].copy()

    expected = config["pediatric_dataset"]
    errors: list[str] = []
    if labeled["case_id"].duplicated().any():
        errors.append("Duplicate labeled pediatric case IDs.")
    if len(labeled) != expected["expected_labeled_cases"]:
        errors.append(
            "Expected "
            f"{expected['expected_labeled_cases']} labeled pediatric cases; "
            f"found {len(labeled)}."
        )
    if len(blind_validation) != expected["expected_unlabeled_validation_cases"]:
        errors.append(
            "Expected "
            f"{expected['expected_unlabeled_validation_cases']} blind cases; "
            f"found {len(blind_validation)}."
        )

    actual_source_counts = (
        labeled["source"].value_counts(dropna=False).to_dict()
    )
    for source, expected_count in expected["sources"].items():
        actual_count = int(actual_source_counts.get(source, 0))
        if actual_count != expected_count:
            errors.append(
                f"Source {source}: expected {expected_count}, "
                f"found {actual_count}."
            )

    all_labeled_ids = ids(labeled)
    folds: list[dict[str, Any]] = []
    validation_occurrences: dict[str, int] = {
        case_id: 0 for case_id in all_labeled_ids
    }
    for fold_definition in config["outer_folds"]:
        fold_number = int(fold_definition["fold"])
        held_out_source = str(fold_definition["held_out_source"])
        validation_rows = labeled.loc[
            labeled["source"].eq(held_out_source)
        ]
        training_rows = labeled.loc[
            ~labeled["source"].eq(held_out_source)
        ]
        training_ids = ids(training_rows)
        validation_ids = ids(validation_rows)
        overlap = sorted(set(training_ids) & set(validation_ids))
        if overlap:
            errors.append(
                f"Fold {fold_number} has {len(overlap)} train/validation overlaps."
            )
        if set(training_ids) | set(validation_ids) != set(all_labeled_ids):
            errors.append(f"Fold {fold_number} does not cover all labeled cases.")
        for case_id in validation_ids:
            validation_occurrences[case_id] += 1

        development_sources = sorted(
            training_rows["source"].dropna().astype(str).unique().tolist()
        )
        inner_folds: list[dict[str, Any]] = []
        for inner_number, inner_validation_source in enumerate(
            development_sources
        ):
            inner_validation_rows = training_rows.loc[
                training_rows["source"].eq(inner_validation_source)
            ]
            inner_training_rows = training_rows.loc[
                ~training_rows["source"].eq(inner_validation_source)
            ]
            inner_folds.append(
                {
                    "inner_fold": inner_number,
                    "validation_source": inner_validation_source,
                    "train": ids(inner_training_rows),
                    "val": ids(inner_validation_rows),
                }
            )

        folds.append(
            {
                "fold": fold_number,
                "held_out_source": held_out_source,
                "development_sources": development_sources,
                "train": training_ids,
                "val": validation_ids,
                "train_sha256": list_hash(training_ids),
                "val_sha256": list_hash(validation_ids),
                "inner_source_folds": inner_folds,
            }
        )

    invalid_occurrences = {
        case_id: count
        for case_id, count in validation_occurrences.items()
        if count != 1
    }
    if invalid_occurrences:
        errors.append(
            f"{len(invalid_occurrences)} cases do not appear exactly once "
            "as an outer validation case."
        )

    payload = {
        "schema_version": "1.0.0",
        "protocol_version": config["protocol_version"],
        "frozen_at": config["frozen_at"],
        "hashes": {
            "manifest_sha256": sha256(args.manifest),
            "config_sha256": sha256(args.config),
            "protocol_sha256": sha256(args.protocol),
            "all_labeled_case_ids_sha256": list_hash(all_labeled_ids),
        },
        "pediatric_labeled_case_ids": all_labeled_ids,
        "pediatric_blind_validation_case_ids": ids(blind_validation),
        "adult_pretraining_case_ids": ids(adult),
        "outer_folds": folds,
        "final_fit": {
            "train": all_labeled_ids,
            "blind_validation": ids(blind_validation),
        },
    }
    audit = {
        "status": "failed" if errors else "passed",
        "errors": errors,
        "labeled_pediatric_cases": len(labeled),
        "blind_pediatric_validation_cases": len(blind_validation),
        "adult_pretraining_cases": len(adult),
        "source_counts": {
            str(source): int(count)
            for source, count in actual_source_counts.items()
        },
        "outer_folds": [
            {
                "fold": fold["fold"],
                "held_out_source": fold["held_out_source"],
                "train_cases": len(fold["train"]),
                "validation_cases": len(fold["val"]),
                "train_validation_overlap": len(
                    set(fold["train"]) & set(fold["val"])
                ),
                "inner_folds": [
                    {
                        "inner_fold": inner["inner_fold"],
                        "validation_source": inner["validation_source"],
                        "train_cases": len(inner["train"]),
                        "validation_cases": len(inner["val"]),
                    }
                    for inner in fold["inner_source_folds"]
                ],
            }
            for fold in folds
        ],
        "outer_validation_occurrence_errors": invalid_occurrences,
        "hashes": payload["hashes"],
    }
    write_json(args.output, payload)
    write_json(args.audit_output, audit)
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit("Formal split audit failed.")


if __name__ == "__main__":
    main()
