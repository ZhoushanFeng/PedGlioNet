#!/usr/bin/env python3
"""Run the six fixed adult-initialization replications with per-job locks.

Start one worker per available GPU. Workers claim fold/seed jobs dynamically,
resume interrupted training, and retain native-space per-case evaluations.
The seed-2026 adult pretraining checkpoint and historical pediatric plans are
fixed: this replicates pediatric fine-tuning, not adult pretraining.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DATASET = "Dataset501_BraTSPEDs"
PLANS = "nnUNetResEncUNetMPlans"
CONFIGURATION = "3d_fullres"
REGIONS = {"ET", "NET", "CC", "ED", "TC", "WT"}
PRETRAIN = Path(
    "data/nnunet/results/Dataset502_BraTSAdult/"
    "nnUNetTrainerPedGlioSeed2026AdultPre__nnUNetResEncUNetMPlansFrom501__"
    "3d_fullres/fold_all/checkpoint_final.pth"
)
QUEUE = Path("results/adult_transfer_replication_20260914")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def paths(root: Path, fold: int, seed: int) -> dict[str, Path]:
    trainer = f"nnUNetTrainerPedGlioSeed{seed}AdultInit"
    model = root / "data/nnunet/results" / DATASET / f"{trainer}__{PLANS}__{CONFIGURATION}" / f"fold_{fold}"
    evaluation = root / f"results/eval/a1ft_fold{fold}_seed{seed}.json"
    return {
        "model": model,
        "prediction": model / "validation",
        "training_status": root / f"results/nnunet_runs/fold{fold}_seed{seed}AdultInit.json",
        "evaluation": evaluation,
        "per_case": evaluation.with_name(evaluation.stem + "_per_case.csv"),
        "job_status": root / QUEUE / f"a1ft_fold{fold}_seed{seed}.json",
        "job_lock": root / QUEUE / f"a1ft_fold{fold}_seed{seed}.lock",
        "train_log": root / f"logs/a1ft_fold{fold}_seed{seed}.log",
        "eval_log": root / f"logs/a1ft_eval_fold{fold}_seed{seed}.log",
    }


def commands(args: argparse.Namespace, fold: int, seed: int) -> tuple[list[str], list[str]]:
    p = paths(args.root, fold, seed)
    train = [
        str(args.python), "-u", "scripts/run_nnunet_seeded.py",
        "--dataset-folder", DATASET, "--plans", PLANS,
        "--configuration", CONFIGURATION, "--fold", str(fold), "--seed", str(seed),
        "--trainer-suffix", "AdultInit", "--num-epochs", "1000",
        "--pretrained-weights", str(PRETRAIN), "--save-probabilities", "--resume",
        "--device", "cuda", "--data-processes", str(args.data_processes),
        "--status-dir", "results/nnunet_runs",
    ]
    evaluate = [
        str(args.python), "-u", "scripts/evaluate_formal_segmentation.py",
        "--prediction-dir", str(p["prediction"]), "--fold", str(fold),
        "--output", str(p["evaluation"]), "--per-case-output", str(p["per_case"]),
        "--bootstrap-samples", "2000", "--seed", "2026", "--surface-tolerance-mm", "1.0",
    ]
    return train, evaluate


def validate_config(root: Path, fold: int, seed: int, data: dict[str, Any], *, historical: bool = False) -> None:
    config = data.get("config", {})
    expected = {
        "dataset_folder": DATASET, "plans": PLANS, "configuration": CONFIGURATION,
        "fold": str(fold), "seed": seed, "trainer": f"nnUNetTrainerPedGlioSeed{seed}AdultInit",
        "save_probabilities": True, "skip_validation": False,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"fold {fold} seed {seed}: incompatible {key}: {config.get(key)!r}")
    if config.get("num_epochs") not in ((None, 1000) if historical else (1000,)):
        raise ValueError(f"fold {fold} seed {seed}: training duration is not the fixed 1000 epochs")
    checkpoint = Path(config.get("pretrained_weights") or "")
    if not checkpoint.is_absolute():
        checkpoint = root / checkpoint
    if checkpoint.resolve() != (root / PRETRAIN).resolve():
        raise ValueError(f"fold {fold} seed {seed}: different adult checkpoint")


def validate_predictions(prediction: Path, case_ids: set[str]) -> bool:
    for extension in (".nii.gz", ".npz"):
        files = list(prediction.glob(f"*{extension}"))
        if {p.name[:-len(extension)] for p in files} != case_ids:
            return False
        if any(p.stat().st_size == 0 for p in files):
            return False
    return True


def evaluation_valid(p: dict[str, Path], fold: int, definition: dict[str, Any]) -> bool:
    """Require all case/region rows, metadata and matching aggregate Dice."""
    if not p["evaluation"].is_file() or not p["per_case"].is_file():
        return False
    try:
        result = read_json(p["evaluation"])
        case_ids = set(definition["val"])
        if (result.get("status") != "completed" or result.get("fold") != fold
                or result.get("evaluated_cases") != len(case_ids)
                or result.get("held_out_source") != definition["held_out_source"]
                or Path(result.get("prediction_dir", "")).resolve() != p["prediction"].resolve()
                or Path(result.get("per_case_output", "")).resolve() != p["per_case"].resolve()
                or result.get("surface_tolerance_mm") != 1.0):
            return False
        with p["per_case"].open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        expected = {(case, region) for case in case_ids for region in REGIONS}
        if len(rows) != len(expected) or {(r["case_id"], r["region"]) for r in rows} != expected:
            return False
        for row in rows:
            dice = float(row["dice"])
            if not math.isfinite(dice) or not 0 <= dice <= 1:
                return False
            if int(row["fold"]) != fold or row["source"] != definition["held_out_source"]:
                return False
        for region in REGIONS:
            values = [float(r["dice"]) for r in rows if r["region"] == region]
            metric = result["metrics"][region]["all_cases"]["dice"]
            if metric["n"] != len(case_ids) or not math.isclose(metric["mean"], sum(values) / len(values), abs_tol=1e-10):
                return False
        return True
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError):
        return False


def preflight(args: argparse.Namespace) -> dict[int, dict[str, Any]]:
    root = args.root
    formal = read_json(root / "data/processed/formal_splits_v1.json")
    definitions = {int(row["fold"]): row for row in formal["outer_folds"]}
    preprocessed = root / "data/nnunet/preprocessed" / DATASET
    splits = json.loads((preprocessed / "splits_final.json").read_text())
    current_plan = read_json(preprocessed / f"{PLANS}.json")
    current_plan.pop("continue_training", None)
    for fold in range(3):
        if set(splits[fold]["val"]) != set(definitions[fold]["val"]):
            raise ValueError(f"Fold {fold}: native evaluation and training validation sets differ")
        if set(splits[fold]["train"]) != set(definitions[fold]["train"]):
            raise ValueError(f"Fold {fold}: native evaluation and training development sets differ")
        historical = paths(root, fold, 2026)
        status = read_json(historical["training_status"])
        validate_config(root, fold, 2026, status, historical=True)
        if status.get("status") != "completed":
            raise ValueError(f"Historical seed 2026 fold {fold} is not complete")
        historical_plan = read_json(historical["model"].parent / "plans.json")
        historical_plan.pop("continue_training", None)
        if historical_plan != current_plan:
            raise ValueError(f"Fold {fold}: current plan differs from the seed-2026 adult model")
    pinned_paths = [
        PRETRAIN, Path("data/processed/formal_splits_v1.json"),
        Path("data/processed/manifest.csv"), preprocessed.relative_to(root) / "splits_final.json",
        preprocessed.relative_to(root) / f"{PLANS}.json", preprocessed.relative_to(root) / "dataset.json",
        Path("scripts/run_nnunet_seeded.py"), Path("scripts/evaluate_formal_segmentation.py"),
    ]
    if not args.python.is_file():
        raise FileNotFoundError(args.python)
    pinned = {str(path): {"sha256": sha256(root / path), "bytes": (root / path).stat().st_size} for path in pinned_paths}
    contract = {"dataset": DATASET, "plans": PLANS, "configuration": CONFIGURATION,
                "epochs": 1000, "seeds": [2027, 2028], "folds": [0, 1, 2],
                "pretrained_checkpoint": str(PRETRAIN), "pinned_inputs": pinned}
    queue = root / QUEUE
    with (queue / "manifest.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest_path = queue / "manifest.json"
        if manifest_path.exists():
            previous = read_json(manifest_path)
            if previous["contract"] != contract:
                raise ValueError("Replication inputs changed since the manifest was pinned; review before resuming")
        else:
            save(manifest_path, {"created_unix": time.time(), "contract": contract,
                                 "interpretation": "Replicate pediatric fine-tuning seeds conditional on one fixed adult-pretraining checkpoint."})
    return definitions


def execute(command: list[str], log_path: Path, args: argparse.Namespace, env: dict[str, str],
            locks: tuple[int, int], state: dict[str, Any], status_path: Path, worker_path: Path) -> None:
    state.update(command=command, child_pid=None)
    save(status_path, state)
    save(worker_path, state)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] {state['stage']} GPU {args.gpu}\n")
        log.flush()
        child = subprocess.Popen(command, cwd=args.root, env=env, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True, pass_fds=locks)
        state["child_pid"] = child.pid
        save(status_path, state)
        save(worker_path, state)
        try:
            code = child.wait()
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            raise
    if code:
        raise RuntimeError(f"{state['stage']} exited {code}; see {log_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    folds = parser.add_mutually_exclusive_group()
    folds.add_argument("--fold", type=int, choices=(0, 1, 2))
    folds.add_argument("--folds", type=int, choices=(0, 1, 2), nargs="+", default=None)
    parser.add_argument("--seeds", type=int, choices=(2027, 2028), nargs="+", default=[2027, 2028])
    parser.add_argument("--data-processes", type=int, default=12)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--python", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print commands and prerequisite presence without writing or launching")
    parser.add_argument("--preflight-only", action="store_true", help="Validate/pin inputs without launching training")
    args = parser.parse_args()
    if args.gpu < 0 or args.data_processes < 1:
        parser.error("GPU must be nonnegative and data-processes positive")
    args.root = args.root.resolve()
    args.python = (args.python or args.root / ".venv/bin/python").absolute()
    args.folds = [args.fold] if args.fold is not None else args.folds or [1, 0, 2]
    args.folds = list(dict.fromkeys(args.folds))
    args.seeds = list(dict.fromkeys(args.seeds))
    return args


def main() -> int:
    args = parse_args()
    # The historical fold-1 runs are longest: dispatch both before shorter folds.
    jobs = [(fold, seed) for fold in args.folds for seed in args.seeds]
    if args.dry_run:
        print(json.dumps({"dry_run": True, "gpu": args.gpu, "root": str(args.root),
                          "checkpoint_exists": (args.root / PRETRAIN).is_file(),
                          "python_exists": args.python.is_file(),
                          "jobs": [{"fold": fold, "seed": seed, "commands": commands(args, fold, seed)} for fold, seed in jobs]}, indent=2))
        return 0
    queue = args.root / QUEUE
    queue.mkdir(parents=True, exist_ok=True)
    (args.root / "logs").mkdir(exist_ok=True)
    worker_path = queue / f"gpu{args.gpu}_worker_status.json"
    gpu_lock = (queue / f"gpu{args.gpu}.lock").open("a+")
    try:
        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"Another adult-replication worker holds GPU {args.gpu}", file=sys.stderr)
        return 3
    definitions = preflight(args)
    if args.preflight_only:
        print(json.dumps({"status": "preflight_passed", "manifest": str(queue / "manifest.json")}))
        return 0
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), nnUNet_n_proc_DA=str(args.data_processes),
               OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
               nnUNet_raw=str(args.root / "data/nnunet/raw"),
               nnUNet_preprocessed=str(args.root / "data/nnunet/preprocessed"),
               nnUNet_results=str(args.root / "data/nnunet/results"))

    def interrupted(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    completed: list[str] = []
    busy: list[str] = []
    for fold, seed in jobs:
        p = paths(args.root, fold, seed)
        job = f"a1ft_fold{fold}_seed{seed}"
        with p["job_lock"].open("a+") as job_lock:
            try:
                fcntl.flock(job_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                busy.append(job)
                continue
            state = {"status": "running", "job": job, "fold": fold, "seed": seed,
                     "gpu": args.gpu, "pid": os.getpid(), "data_processes": args.data_processes,
                     "started_unix": time.time(), "stage": "checking_existing_artifacts"}
            try:
                training_complete = False
                if p["training_status"].exists():
                    status = read_json(p["training_status"])
                    validate_config(args.root, fold, seed, status)
                    training_complete = status.get("status") == "completed"
                elif any(p["model"].glob("checkpoint_*.pth")):
                    raise RuntimeError("Checkpoints exist without a training config; refusing an unverified resume")
                training_complete = (training_complete and (p["model"] / "checkpoint_final.pth").is_file()
                                     and validate_predictions(p["prediction"], set(definitions[fold]["val"])))
                train, evaluate = commands(args, fold, seed)
                if not training_complete:
                    state["stage"] = "training_and_probability_validation"
                    execute(train, p["train_log"], args, env, (job_lock.fileno(), gpu_lock.fileno()), state, p["job_status"], worker_path)
                    status = read_json(p["training_status"])
                    validate_config(args.root, fold, seed, status)
                    if status.get("status") != "completed" or not (p["model"] / "checkpoint_final.pth").is_file():
                        raise RuntimeError("Training returned without a completed final checkpoint")
                if not validate_predictions(p["prediction"], set(definitions[fold]["val"])):
                    raise RuntimeError("Native predictions/probabilities do not cover exactly the held-out cases")
                if not training_complete or not evaluation_valid(p, fold, definitions[fold]):
                    state["stage"] = "native_space_evaluation"
                    execute(evaluate, p["eval_log"], args, env, (job_lock.fileno(), gpu_lock.fileno()), state, p["job_status"], worker_path)
                if not evaluation_valid(p, fold, definitions[fold]):
                    raise RuntimeError("Evaluation completeness or case/region checks failed")
                state.update(status="completed", stage="completed", child_pid=None,
                             finished_unix=time.time(), evaluated_cases=len(definitions[fold]["val"]),
                             evaluation=str(p["evaluation"]), per_case=str(p["per_case"]),
                             per_case_sha256=sha256(p["per_case"]))
                save(p["job_status"], state)
                save(worker_path, state)
                completed.append(job)
                print(f"GPU {args.gpu}: {job} training_and_evaluation_completed", flush=True)
            except BaseException as error:
                state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                             error=str(error), finished_unix=time.time(), child_pid=None)
                save(p["job_status"], state)
                save(worker_path, state)
                raise
    state = {"status": "worker_finished", "gpu": args.gpu, "pid": os.getpid(),
             "finished_unix": time.time(), "completed_or_verified_jobs": completed,
             "jobs_claimed_by_other_workers": busy}
    save(worker_path, state)
    print(json.dumps(state, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
