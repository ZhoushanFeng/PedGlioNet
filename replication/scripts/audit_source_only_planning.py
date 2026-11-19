#!/usr/bin/env python3
"""Compare development-source-only nnU-Net geometry planning to historical plans.

No training/preprocessing is launched. Existing per-case geometry properties
are restricted to each fold's training cases. MRI intensity summary entries are
neutral placeholders because ZScoreNormalization does not consume them.
"""
from pathlib import Path
import contextlib
import copy
import hashlib
import importlib
import io
import json
import os
import pickle
import sys

import numpy as np


def main():
    root = Path.cwd()
    out = root / 'results/planning_sensitivity_20260913'
    original = root / 'data/nnunet/preprocessed/Dataset501_BraTSPEDs'
    dataset = json.loads((original / 'dataset.json').read_text())
    plans = json.loads((original / 'nnUNetResEncUNetMPlans.json').read_text())
    splits = json.loads((root / 'data/processed/formal_splits_v1.json').read_text())
    original_hash = hashlib.sha256((original / 'nnUNetResEncUNetMPlans.json').read_bytes()).hexdigest()
    os.environ['nnUNet_raw'] = str(out / 'raw')
    os.environ['nnUNet_preprocessed'] = str(out / 'preprocessed')
    os.environ['nnUNet_results'] = str(out / 'unused_training_results')
    from nnunetv2.experiment_planning.experiment_planners.residual_unets.residual_encoder_unet_planners import nnUNetPlannerResEncM
    from nnunetv2.preprocessing.normalization.default_normalization_schemes import ZScoreNormalization
    import inspect
    assert 'intensityproperties' not in inspect.getsource(ZScoreNormalization.run)
    all_properties = {}
    property_paths = {}
    for case in splits['pediatric_labeled_case_ids']:
        paths = list(original.glob(f'*/{case}.pkl'))
        if not paths:
            raise FileNotFoundError(f'No preprocessed geometry for {case}')
        preferred = [p for p in paths if p.parent.name == plans['configurations']['3d_fullres']['data_identifier']]
        path = preferred[0] if preferred else sorted(paths)[0]
        with path.open('rb') as stream:
            props = pickle.load(stream)
        all_properties[case] = props
        property_paths[case] = path
    # Check that reconstructed aggregate geometry agrees with the fingerprint.
    fp = json.loads((original / 'dataset_fingerprint.json').read_text())
    shapes = [list(all_properties[c]['shape_after_cropping_and_before_resampling']) for c in splits['pediatric_labeled_case_ids']]
    shape_match = sorted(shapes) == sorted(fp['shapes_after_crop'])
    if not shape_match:
        raise AssertionError('Stored preprocessed crop geometries do not reconstruct original fingerprint')
    checks = []
    fields = ['spacing', 'patch_size', 'batch_size', 'normalization_schemes', 'use_mask_for_norm',
              'resampling_fn_data', 'resampling_fn_data_kwargs', 'resampling_fn_seg', 'resampling_fn_seg_kwargs',
              'resampling_fn_probabilities', 'resampling_fn_probabilities_kwargs', 'architecture', 'preprocessor_name', 'batch_dice']
    for f in splits['outer_folds']:
        fold = int(f['fold'])
        name = f'Dataset{611+fold}_PedGlioDevelopmentOnlyFold{fold}'
        raw, prep = out / 'raw' / name, out / 'preprocessed' / name
        raw.mkdir(parents=True, exist_ok=True); prep.mkdir(parents=True, exist_ok=True)
        d = copy.deepcopy(dataset)
        d['numTraining'] = len(f['train'])
        # File mapping is only used to infer the reader and number of cases.
        d['dataset'] = {case: {
            'images': [str(root / f'data/nnunet/raw/Dataset501_BraTSPEDs/imagesTr/{case}_{m:04d}.nii.gz') for m in range(4)],
            'label': str(root / f'data/nnunet/raw/Dataset501_BraTSPEDs/labelsTr/{case}.nii.gz')
        } for case in f['train']}
        (raw / 'dataset.json').write_text(json.dumps(d))
        development_props = [all_properties[c] for c in f['train']]
        dev_fp = {
            'spacings': [[float(x) for x in p['spacing']] for p in development_props],
            'shapes_after_crop': [[int(x) for x in p['shape_after_cropping_and_before_resampling']] for p in development_props],
            'median_relative_size_after_cropping': float(np.median([np.prod(p['shape_after_cropping_and_before_resampling']) / np.prod(p['shape_before_cropping']) for p in development_props])),
            'foreground_intensity_properties_per_channel': {str(i): {'mean': 0., 'median': 0., 'std': 1., 'min': 0., 'max': 1., 'percentile_00_5': 0., 'percentile_99_5': 1.} for i in range(4)}}
        (prep / 'dataset_fingerprint.json').write_text(json.dumps(dev_fp))
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            planner = nnUNetPlannerResEncM(name)
            new_plans = planner.plan_experiment()
        (out / f'fold{fold}_planner.log').write_text(stream.getvalue())
        if new_plans is None:
            new_plans = json.loads((prep / f'{planner.plans_identifier}.json').read_text())
        # Planner output can contain tuples while disk JSON contains lists.
        new_plans = json.loads(json.dumps(new_plans))
        old_c, new_c = plans['configurations']['3d_fullres'], new_plans['configurations']['3d_fullres']
        changed = {k: {'historical': old_c.get(k), 'development_only': new_c.get(k)} for k in fields if old_c.get(k) != new_c.get(k)}
        for k in ['transpose_forward', 'transpose_backward']:
            if plans.get(k) != new_plans.get(k):
                changed[k] = {'historical': plans.get(k), 'development_only': new_plans.get(k)}
        checks.append({'fold': fold, 'development_cases': len(f['train']), 'held_out_cases': len(f['val']),
                       'material_configuration_fields_compared': fields + ['transpose_forward', 'transpose_backward'],
                       'changed_fields': changed, 'all_material_fields_identical': not changed,
                       'historical_patch_size': old_c['patch_size'], 'development_patch_size': new_c['patch_size'],
                       'development_median_crop_fraction': dev_fp['median_relative_size_after_cropping']})
        # Save an explicitly named plan for a sensitivity run, retaining exactly
        # the verified-equivalent case-wise preprocessed representation.
        derived_name = f'nnUNetResEncMDevelopmentFold{fold}'
        derived = copy.deepcopy(new_plans)
        derived['dataset_name'] = plans['dataset_name']
        derived['plans_name'] = derived_name
        derived['configurations']['3d_fullres']['data_identifier'] = old_c['data_identifier']
        (out / f'{derived_name}.json').write_text(json.dumps(derived, indent=2) + '\n')
        print(json.dumps(checks[-1]), flush=True)
    assert hashlib.sha256((original / 'nnUNetResEncUNetMPlans.json').read_bytes()).hexdigest() == original_hash
    result = {'status': 'completed', 'post_hoc': True, 'all_case_geometry_reconstructs_fingerprint': shape_match,
              'historical_plans_sha256_unchanged': original_hash,
              'MRI_intensity_statistics_consumed': False,
              'method': 'Restrict saved preprocessed crop geometries to outer training case IDs, regenerate ResEnc M plans in an isolated directory, compare all network/normalization/resampling/batch/transpose settings.',
              'limit': 'Configuration equivalence is a sensitivity audit, not a retroactive claim that original planning excluded test-source images.',
              'folds': checks}
    (out / 'planning_comparison.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
