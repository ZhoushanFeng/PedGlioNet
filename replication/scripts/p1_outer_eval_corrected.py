#!/usr/bin/env python3
"""Auditable frozen-P1 evaluation correction; original outputs are untouched.

Corrects header spacing and scoring domain only. Inference is imported from
the historical evaluator and checked against its stored cropped-space scores.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from et_small_target_refinement import mask_center, nonzero_bbox, patch_spec
from evaluate_formal_segmentation import dice_score, lesion_counts
from p1_outer_eval import PATCH_SHAPE, SEEDS, dev_presence_thresholds, load_models, predict_case


METHODS = ('b0_ensemble', 'p1_ungated', 'p1_gated')


def write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def image_tensor(row):
    """Exactly the historical image-only preprocessing, plus crop geometry."""
    volumes = [nib.load(str(row[c])) for c in ('t1n', 't1c', 't2w', 't2f')]
    first = volumes[0]
    for image in volumes[1:]:
        assert image.shape == first.shape
        assert np.allclose(image.affine, first.affine, atol=1e-4)
    arrays = [np.asarray(v.dataobj, dtype=np.float32) for v in volumes]
    image = np.stack(arrays, axis=0).transpose(0, 3, 2, 1)
    full_shape = image.shape[1:]
    crop = nonzero_bbox(image)
    image = image[(slice(None),) + crop]
    foreground = np.any(np.abs(image) > 1e-6, axis=0)
    for channel in range(image.shape[0]):
        values = image[channel][foreground]
        if values.size:
            mean = float(values.mean())
            std = max(float(values.std()), 1e-6)
            image[channel] = np.clip((image[channel] - mean) / std, -5., 5.)
            image[channel][~foreground] = 0.
    return torch.from_numpy(image).contiguous(), crop, full_shape, first


def restore(mask: np.ndarray, crop, full_shape):
    full = np.zeros(full_shape, dtype=mask.dtype)
    assert full[crop].shape == mask.shape
    full[crop] = mask
    return full


def read_labels(path: Path, image_geometry):
    labels = nib.load(path)
    if labels.shape != image_geometry.shape or not np.allclose(labels.affine, image_geometry.affine, atol=1e-4):
        raise ValueError(f'Geometry mismatch: {path}')
    return np.asarray(labels.dataobj, dtype=np.uint8).transpose(2, 1, 0), labels


def voxel_volume(image):
    spacing = np.asarray(image.header.get_zooms()[:3], dtype=float)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError('NIfTI header has invalid spatial spacing')
    return float(np.prod(spacing))


def metrics(reference, prediction, volume):
    return {'dice': dice_score(reference, prediction),
            'reference_nonempty': bool(reference.any()),
            'prediction_nonempty': bool(prediction.any()),
            'reference_volume_mm3': float(reference.sum()) * volume,
            'prediction_volume_mm3': float(prediction.sum()) * volume,
            **lesion_counts(reference, prediction)}


def coverage(reference, roi):
    count = int(reference.sum())
    return float((reference & roi).sum() / count) if count else None


def save_mask(path: Path, mask, reference_image):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = reference_image.header.copy()
    header.set_data_dtype(np.uint8)
    image = nib.Nifti1Image(mask.transpose(2, 1, 0).astype(np.uint8), reference_image.affine, header)
    nib.save(image, path)


def summarize(frame):
    result = {}
    positive = frame.loc[frame.b0_ensemble_reference_nonempty]
    negative = frame.loc[~frame.b0_ensemble_reference_nonempty]
    for m in METHODS:
        result[m] = {
            'all_case_dice': float(frame[f'{m}_dice'].mean()),
            'positive_case_dice': float(positive[f'{m}_dice'].mean()) if len(positive) else None,
            'negative_specificity': float((~negative[f'{m}_prediction_nonempty']).mean()) if len(negative) else None,
            'negative_fp_volume_mm3': float(negative[f'{m}_prediction_volume_mm3'].mean()) if len(negative) else None,
            'reference_lesions': int(frame[f'{m}_reference_lesions'].sum()),
            'predicted_lesions': int(frame[f'{m}_predicted_lesions'].sum()),
            'lesion_sensitivity': float(frame[f'{m}_detected_reference_lesions'].sum() / frame[f'{m}_reference_lesions'].sum()) if frame[f'{m}_reference_lesions'].sum() else None,
            'lesion_ppv': float(frame[f'{m}_correct_predicted_lesions'].sum() / frame[f'{m}_predicted_lesions'].sum()) if frame[f'{m}_predicted_lesions'].sum() else None,
        }
    result['roi_et_coverage_positive_mean'] = float(positive.roi_et_coverage.mean()) if len(positive) else None
    result['roi_et_fully_covered_positive_count'] = int((positive.roi_et_coverage == 1).sum())
    result['roi_et_positive_count'] = len(positive)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, default=Path('data/processed/manifest.csv'))
    parser.add_argument('--splits', type=Path, default=Path('data/processed/formal_splits_v1.json'))
    parser.add_argument('--output-dir', type=Path, default=Path('results/p1_outer_corrected_20260913'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--smoke', type=int, default=0, help='N fit cases/fold; mechanical check, not manuscript data')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / 'status.json'
    definitions = json.loads(args.splits.read_text())
    manifest = pd.read_csv(args.manifest).set_index('case_id', drop=False)
    checkpoints = Path('results/checkpoints')
    run_dir = Path('results/p1')
    historical_path = Path('results/p1_outer/p1_outer_per_case.csv')
    historical = pd.read_csv(historical_path).set_index('case_id')
    artifacts = [args.manifest, args.splits, historical_path, Path(__file__),
                 Path('scripts/p1_outer_eval.py'), Path('scripts/et_small_target_refinement.py'),
                 Path('scripts/joint_et_refiner.py'), Path('scripts/evaluate_formal_segmentation.py')]
    for fold in range(3):
        artifacts.append(Path(f'results/eval/ensemble_b0_fold{fold}_per_case.csv'))
        for seed in SEEDS:
            artifacts.append(checkpoints / f'joint_et_v2_fold{fold}_seed{seed}/stageB/joint_et_best.pt')
            artifacts.append(run_dir / f'joint_et_v2_fold{fold}_seed{seed}.json')
    provenance = {'revision': 'native-space-and-header-spacing-correction-20260913',
                  'original_model_protocol': 'FORMAL_P1_OUTER_EVAL_LOCK.md',
                  'smoke_cases_per_fold': args.smoke, 'device': str(device),
                  'patch_shape': PATCH_SHAPE, 'seeds': SEEDS,
                  'sha256': {str(p): sha256(p) for p in artifacts}}
    provenance_path = args.output_dir / 'provenance.json'
    if provenance_path.exists() and json.loads(provenance_path.read_text()) != json.loads(json.dumps(provenance)):
        raise ValueError('Refuse to mix outputs with different input/code/model provenance')
    write_json(provenance_path, provenance)
    records = []
    expected_total = sum(min(args.smoke, len(f['train'])) if args.smoke else len(f['val']) for f in definitions['outer_folds'])
    try:
        for fold_def in definitions['outer_folds']:
            fold = int(fold_def['fold'])
            models = load_models(checkpoints, fold, device)
            threshold = dev_presence_thresholds(run_dir, fold)
            if args.smoke:
                run = json.loads((run_dir / f'joint_et_v2_fold{fold}_seed{SEEDS[0]}.json').read_text())
                case_ids = run['split']['fit_case_ids'][:args.smoke]
            else:
                case_ids = fold_def['val']
            b0_expected = pd.read_csv(f'results/eval/ensemble_b0_fold{fold}_per_case.csv')
            b0_expected = b0_expected.loc[b0_expected.region.eq('ET')].set_index('case_id')
            for case_id in case_ids:
                case_path = args.output_dir / 'case_records' / f'{case_id}_fold{fold}.json'
                if case_path.exists():
                    entry = json.loads(case_path.read_text())
                    for m in METHODS[1:]:
                        assert (args.output_dir / 'predictions' / m / f'{case_id}_fold{fold}.nii.gz').exists()
                    records.append(entry)
                    continue
                row = manifest.loc[case_id]
                image, crop, shape, image_geometry = image_tensor(row)
                reference, ref_image = read_labels(Path(row['seg']), image_geometry)
                vv = voxel_volume(ref_image)
                if args.smoke:
                    # Mechanics check only: no formal performance claim for fit cases.
                    b0 = reference.copy()
                    from et_small_target_refinement import load_native_case
                    assert torch.equal(image, load_native_case(row)['image']), 'Preprocessing differs from historical implementation'
                else:
                    b0, _ = read_labels(Path(f'results/ensemble/b0_fold{fold}/{case_id}.nii.gz'), image_geometry)
                roi_source = torch.from_numpy(b0[crop] > 0)
                center = mask_center(roi_source)
                p1_crop, presence = predict_case(models, image, center, device)
                p1 = restore(p1_crop.numpy(), crop, shape)
                gated = p1 if presence >= threshold else np.zeros_like(p1)
                spec = patch_spec(center, tuple(image.shape[1:]), PATCH_SHAPE)
                roi_crop = np.zeros(tuple(image.shape[1:]), dtype=bool)
                roi_crop[spec.source] = True
                roi = restore(roi_crop, crop, shape)
                oracle_center = mask_center(torch.from_numpy(reference[crop] > 0))
                oracle_spec = patch_spec(oracle_center, tuple(image.shape[1:]), PATCH_SHAPE)
                oracle_crop = np.zeros_like(roi_crop)
                oracle_crop[oracle_spec.source] = True
                oracle_roi = restore(oracle_crop, crop, shape)
                entry = {'case_id': case_id, 'fold': fold, 'source': str(row['source']),
                         'smoke': bool(args.smoke), 'presence_probability': presence,
                         'gate_threshold': threshold, 'roi_nonempty': bool(roi_source.any()),
                         'voxel_volume_mm3': vv, 'roi_et_coverage': coverage(reference == 1, roi),
                         'roi_wt_coverage': coverage(reference > 0, roi),
                         'oracle_roi_et_coverage': coverage(reference == 1, oracle_roi),
                         'reference_et_voxels_outside_image_crop': int((reference == 1).sum() - (reference[crop] == 1).sum())}
                for m, mask in zip(METHODS, (b0 == 1, p1, gated)):
                    entry.update({f'{m}_{k}': v for k, v in metrics(reference == 1, mask, vv).items()})
                if not args.smoke:
                    check = b0_expected.loc[case_id]
                    for key in ('dice', 'reference_volume_mm3', 'prediction_volume_mm3', 'reference_lesions', 'detected_reference_lesions', 'predicted_lesions', 'correct_predicted_lesions'):
                        if not np.isclose(entry[f'b0_ensemble_{key}'], check[key], rtol=1e-8, atol=1e-8):
                            raise AssertionError(f'Full-space B0 disagreement: {case_id} {key}')
                    hist = historical.loc[case_id]
                    entry['historical_presence_abs_difference'] = abs(presence - float(hist.presence_probability))
                    for m, mask in [('p1_ungated', p1), ('p1_gated', gated)]:
                        cropped_dice = dice_score(reference[crop] == 1, mask[crop])
                        historical_voxels = float(hist[f'{m}_prediction_volume_mm3']) / 1024.
                        entry[f'{m}_historical_cropped_dice_abs_difference'] = abs(cropped_dice - float(hist[f'{m}_dice']))
                        entry[f'{m}_historical_predicted_voxels_difference'] = int(mask.sum()) - historical_voxels
                        if entry[f'{m}_historical_cropped_dice_abs_difference'] > 1e-6 or abs(entry[f'{m}_historical_predicted_voxels_difference']) > 0.5:
                            raise AssertionError(f'Frozen inference changed relative to historical run: {case_id} {m}')
                for m, mask in [('p1_ungated', p1), ('p1_gated', gated)]:
                    save_mask(args.output_dir / 'predictions' / m / f'{case_id}_fold{fold}.nii.gz', mask, ref_image)
                write_json(case_path, entry)
                records.append(entry)
                write_json(status_path, {'status': 'running', 'pid': os.getpid(), 'completed_cases': len(records), 'expected_cases': expected_total,
                                         'fold': fold, 'elapsed_seconds': round(time.time() - started, 1), 'smoke': bool(args.smoke)})
                if len(records) % 10 == 0 or len(records) == 1:
                    print(f'completed={len(records)}/{expected_total} fold={fold} seconds={time.time()-started:.1f}', flush=True)
            del models
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        frame = pd.DataFrame(records)
        assert len(frame) == expected_total
        if not args.smoke:
            assert frame.case_id.is_unique and set(frame.case_id) == set(definitions['pediatric_labeled_case_ids'])
        frame.to_csv(args.output_dir / 'p1_outer_per_case_corrected.csv', index=False)
        payload = {'status': 'completed', 'smoke': bool(args.smoke), 'n_cases': len(frame),
                   'elapsed_seconds': round(time.time() - started, 1), 'provenance': provenance,
                   'pooled': summarize(frame), 'folds': [{'fold': int(f), 'n_cases': len(g), **summarize(g)} for f, g in frame.groupby('fold')],
                   'inference_checks': {c: float(frame[c].abs().max()) for c in frame.columns if 'difference' in c}}
        write_json(args.output_dir / 'p1_outer_eval_corrected.json', payload)
        write_json(status_path, {'status': 'completed', 'pid': os.getpid(), 'completed_cases': len(frame), 'expected_cases': expected_total,
                                 'elapsed_seconds': round(time.time() - started, 1), 'smoke': bool(args.smoke)})
        print(json.dumps({'status': 'completed', 'n_cases': len(frame), 'smoke': bool(args.smoke), 'pooled': payload['pooled'], 'inference_checks': payload['inference_checks']}), flush=True)
    except Exception as error:
        write_json(status_path, {'status': 'failed', 'pid': os.getpid(), 'completed_cases': len(records), 'expected_cases': expected_total,
                                 'error_type': type(error).__name__, 'error': str(error), 'elapsed_seconds': round(time.time() - started, 1)})
        raise


if __name__ == '__main__':
    main()
