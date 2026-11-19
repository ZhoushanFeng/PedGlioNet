#!/usr/bin/env python3
"""Deterministic, audited MRI examples; no synthesized images or model ensembles."""
from pathlib import Path
import hashlib
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from matplotlib.lines import Line2D
import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
PREDICTIONS = Path(os.environ.get('PEDGLIO_PREDICTIONS', str(ROOT / 'authorized_predictions')))
RAW = Path(os.environ.get('PEDGLIO_BRATS_TRAINING', str(ROOT / 'authorized_training')))
OUT = ROOT / 'output/authorized_figure10'
SEEDS = (2026, 2027, 2028)
ET_COLOR = '#FFD24A'
WT_COLOR = '#54DDE3'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dice(a, b):
    n = int(a.sum()) + int(b.sum())
    return float(2 * np.count_nonzero(a & b) / n) if n else 1.0


def prediction_path(case, method, seed):
    return PREDICTIONS / method / f"fold{case['fold']}_seed{seed}" / f"{case['case_id']}.nii.gz"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    membership_path = DATA / 'omission_membership.json'
    manifest_path = DATA / 'figure10_prediction_checksums.json'
    cases = json.loads(membership_path.read_text())['cases']
    manifest = json.loads(manifest_path.read_text())
    tables = {}
    for method, prefix in [('b0', 'b0'), ('adult_init', 'a1ft')]:
        for seed in SEEDS:
            tables[method, seed] = pd.concat([
                pd.read_csv(DATA / f'eval/{prefix}_fold{fold}_seed{seed}_per_case.csv')
                for fold in range(3)
            ]).set_index(['case_id', 'region'])

    persistent = [c for c in cases if c['reference_et_positive']
                  and c['all_three_seed_et_empty']['b0']
                  and c['all_three_seed_et_empty']['adult_init']
                  and all(float(tables[m, 2026].loc[(c['case_id'], 'WT'), 'dice']) >= .9
                          for m in ['b0', 'adult_init'])]
    persistent_median = float(np.median([c['reference_et_volume_mm3'] for c in persistent]))
    row_a = min(persistent, key=lambda c: (abs(c['reference_et_volume_mm3'] - persistent_median), c['case_id']))
    changed = [c for c in cases if c['reference_et_positive']
               and c['all_three_seed_et_empty']['b0'] and c['seed_et_presence']['adult_init']['2026']]
    changed_median = float(np.median([c['seed_et_dice']['adult_init']['2026'] for c in changed]))
    row_b = min(changed, key=lambda c: (abs(c['seed_et_dice']['adult_init']['2026'] - changed_median), c['case_id']))
    both = [c for c in cases if c['reference_et_positive']
            and c['seed_et_presence']['b0']['2026'] and c['seed_et_presence']['adult_init']['2026']]
    both_medians = np.median([[c['seed_et_dice'][m]['2026'] for m in ['b0', 'adult_init']]
                             for c in both], axis=0)
    row_c = min(both, key=lambda c: (sum((c['seed_et_dice'][m]['2026'] - med)**2
                                       for m, med in zip(['b0', 'adult_init'], both_medians)), c['case_id']))

    selection = {
        'a': {'rule': 'Reference-positive ET; all three seeds predict empty ET under both methods; '
                      'both seed-2026 WT Dice values >=0.90. Select nearest median reference ET volume; '
                      'ties resolved lexicographically by case ID.',
              'eligible_count': len(persistent), 'eligible_case_ids': [c['case_id'] for c in persistent],
              'median_reference_et_mm3': persistent_median},
        'b': {'rule': 'Reference-positive ET; all B0 seeds predict empty ET; AdultInit seed 2026 predicts '
                      'nonempty ET. Select nearest median AdultInit seed-2026 ET Dice; '
                      'ties resolved lexicographically by case ID.',
              'eligible_count': len(changed), 'eligible_case_ids': [c['case_id'] for c in changed],
              'median_adult_init_et_dice': changed_median},
        'c': {'rule': 'Reference-positive ET; both methods predict nonempty ET at seed 2026. Select '
                      'minimum squared Euclidean distance from the pair of method-specific median '
                      'seed-2026 ET Dice values; ties resolved lexicographically by case ID.',
              'eligible_count': len(both), 'eligible_case_ids': [c['case_id'] for c in both],
              'median_et_dice': dict(zip(['b0', 'adult_init'], both_medians.tolist()))}}
    titles = ['Persistent ET omission', 'Nonempty ET with AdultInit', 'Both methods predict ET']
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 9,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})
    fig = plt.figure(figsize=(7.2, 7.6), facecolor='white')
    left, right, gap = .035, .985, .012
    width = (right - left - 3 * gap) / 4
    height = width * 7.2 / 7.6
    row_tops = [.87, .58, .29]
    for col, heading in enumerate(['Postcontrast T1', 'Reference', 'B0', 'AdultInit']):
        fig.text(left + col * (width + gap) + width/2, .965, heading,
                 ha='center', va='center', fontsize=10, fontweight='bold')
    fig.legend(handles=[Line2D([], [], color=ET_COLOR, lw=1.8, label='ET'),
                        Line2D([], [], color=WT_COLOR, lw=1.5, ls='--', label='WT')],
               loc='upper center', bbox_to_anchor=(.5, .945), ncol=2,
               frameon=False, fontsize=9, handlelength=2.4, columnspacing=2)
    records = []
    for row, case in enumerate([row_a, row_b, row_c]):
        case_id = case['case_id']
        files = {'t1c': RAW/case_id/f'{case_id}-t1c.nii.gz',
                 'reference': RAW/case_id/f'{case_id}-seg.nii.gz'}
        for method in ['b0', 'adult_init']:
            for seed in SEEDS:
                files[f'{method}_{seed}'] = prediction_path(case, method, seed)
        hashes = {key: sha(path) for key, path in files.items()}
        for key, path in files.items():
            if key not in ['t1c', 'reference']:
                assert hashes[key] == manifest[path.relative_to(PREDICTIONS).as_posix()]
        images = {k: nib.load(p) for k, p in files.items()}
        reference = images['reference']
        for image in images.values():
            assert image.shape == reference.shape
            np.testing.assert_allclose(image.affine, reference.affine, rtol=0, atol=1e-5)
        canonical = {k: nib.as_closest_canonical(im) for k, im in images.items()}
        assert nib.aff2axcodes(canonical['reference'].affine) == ('R', 'A', 'S')
        spacing = canonical['reference'].header.get_zooms()[:3]
        np.testing.assert_allclose(spacing, [1, 1, 1], rtol=0, atol=1e-6)
        arrays = {k: np.asarray(im.dataobj) for k, im in canonical.items()}
        gt_et = arrays['reference'] == 1
        gt_wt = np.isin(arrays['reference'], [1, 2, 3, 4])
        assert int(gt_et.sum()) == int(case['reference_et_volume_mm3'])
        metrics = {}
        for method in ['b0', 'adult_init']:
            metrics[method] = {}
            for seed in SEEDS:
                key = f'{method}_{seed}'
                pred_et = arrays[key] == 1
                pred_wt = np.isin(arrays[key], [1, 2, 3, 4])
                values = {'ET_dice': dice(gt_et, pred_et), 'WT_dice': dice(gt_wt, pred_wt),
                          'ET_prediction_mm3': int(pred_et.sum()), 'WT_prediction_mm3': int(pred_wt.sum()),
                          'ET_nonempty': bool(pred_et.any())}
                assert values['ET_nonempty'] == case['seed_et_presence'][method][str(seed)]
                np.testing.assert_allclose(values['ET_dice'], case['seed_et_dice'][method][str(seed)], atol=1e-12)
                for region in ['ET', 'WT']:
                    np.testing.assert_allclose(values[f'{region}_dice'],
                                               tables[method, seed].loc[(case_id, region), 'dice'], atol=1e-12)
                metrics[method][str(seed)] = values
        areas = gt_et.sum(axis=(0, 1))
        maxima = np.flatnonzero(areas == areas.max())
        z = int(maxima[len(maxima)//2])
        # Show all reference/prediction WT contours in the selected slice with equal field of view.
        union = gt_wt[:, :, z].copy()
        for method in ['b0', 'adult_init']:
            union |= np.isin(arrays[f'{method}_2026'][:, :, z], [1, 2, 3, 4])
        points = np.argwhere(union)
        lo, hi = points.min(0), points.max(0)
        side = int(min(min(union.shape), max(96, max(hi - lo + 1) + 24)))
        center = (lo + hi)/2
        x0, y0 = np.clip(np.round(center-side/2).astype(int), 0, np.array(union.shape)-side)
        x0, y0 = int(x0), int(y0)
        positive_values = arrays['t1c'][arrays['t1c'] > 0]
        window = [float(v) for v in np.percentile(positive_values, [1, 99.5])]
        image_crop = arrays['t1c'][x0:x0+side, y0:y0+side, z].T
        top = row_tops[row]
        fig.text(left, top+.010, f"{chr(97+row)}  PED-{case_id.split('-')[2]}  |  {titles[row]}",
                 fontsize=10, fontweight='bold', va='bottom')
        for col, key in enumerate(['t1c', 'reference', 'b0_2026', 'adult_init_2026']):
            ax = fig.add_axes([left + col*(width+gap), top-height, width, height])
            ax.imshow(image_crop, origin='lower', cmap='gray', vmin=window[0], vmax=window[1], interpolation='nearest')
            if key != 't1c':
                crop = arrays[key][x0:x0+side, y0:y0+side, z].T
                wt_crop, et_crop = np.isin(crop, [1, 2, 3, 4]), crop == 1
                if wt_crop.any():
                    contour = ax.contour(wt_crop, levels=[.5], colors=[WT_COLOR], linewidths=.85,
                                         linestyles='--', antialiased=True)
                    contour.set_path_effects([patheffects.withStroke(linewidth=1.3, foreground='#222222')])
                if et_crop.any():
                    contour = ax.contour(et_crop, levels=[.5], colors=[ET_COLOR], linewidths=1.05,
                                         linestyles='-', antialiased=True)
                    contour.set_path_effects([patheffects.withStroke(linewidth=1.5, foreground='#222222')])
            if col == 0:
                ax.plot([7, 27], [7, 7], color='white', lw=1.6)
                ax.text(17, 10, '20 mm', ha='center', va='bottom', color='white', fontsize=8)
                ax.text(3, side-4, 'L', color='white', fontsize=8, va='top')
                ax.text(side-4, side-4, 'R', color='white', fontsize=8, ha='right', va='top')
            ax.set_xlim(-.5, side-.5)
            ax.set_ylim(-.5, side-.5)
            ax.set_axis_off()
            if col == 0:
                note = f"{case['source']} | slice {z}"
            elif col == 1:
                note = f"ET volume {int(gt_et.sum()):,} mm³"
            else:
                values = metrics['b0' if col == 2 else 'adult_init']['2026']
                note = f"ET {values['ET_dice']:.3f} | WT {values['WT_dice']:.3f}"
            fig.text(left+col*(width+gap)+width/2, top-height-.011, note,
                     ha='center', va='top', fontsize=8.5)
        records.append({'panel': chr(97+row), 'case_id': case_id, 'source': case['source'],
                        'fold': case['fold'], 'reference_ET_mm3': int(gt_et.sum()),
                        'reference_WT_mm3': int(gt_wt.sum()), 'selection': selection[chr(97+row)],
                        'per_seed_metrics_recomputed_from_masks': metrics,
                        'canonical_axial_slice_index_zero_based': z,
                        'slice_reference_ET_area_mm2': int(areas[z]),
                        'slice_reference_ET_max_area_tied_indices': maxima.tolist(),
                        'canonical_crop_xy': [x0, y0], 'crop_side_mm': side,
                        'canonical_affine': canonical['reference'].affine.tolist(),
                        'slice_world_z_mm': float((canonical['reference'].affine @ [0, 0, z, 1])[2]),
                        'intensity_window': window, 'source_sha256': hashes})
    fig.text(.5, .026, 'Seed 2026 predictions • ET and WT values denote whole-volume Dice',
             ha='center', va='center', fontsize=8.5)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [(text, text.get_window_extent(renderer)) for text in fig.texts]
    for i, (text, box) in enumerate(boxes):
        assert fig.bbox.contains(box.x0, box.y0) and fig.bbox.contains(box.x1, box.y1), text.get_text()
        for other, other_box in boxes[i+1:]:
            assert not box.overlaps(other_box), (text.get_text(), other.get_text())
    for suffix in ['png', 'pdf', 'svg']:
        fig.savefig(OUT/f'Figure10_mri.{suffix}', dpi=600, facecolor='white')
    plt.close(fig)
    caption = (
        'Figure 10. Illustrative held-out MRI examinations showing distinct ET prediction states. '
        'Each row shows the same postcontrast T1 slice without overlays, with the dataset reference, '
        'with B0, and with AdultInit. Yellow solid contours denote enhancing tumor (ET); cyan dashed '
        'contours denote whole tumor (WT). Both predictions use seed 2026; no probability ensemble '
        'is shown. Numbers below the model panels are whole-volume ET and WT Dice, not slice Dice. '
        '(a) PED-00075 has empty ET predictions for all three seeds under both methods despite '
        'seed-2026 WT Dice of 0.909 for both methods. Among eight such persistent-omission cases '
        'with seed-2026 WT Dice at least 0.90 for both methods, the examination nearest the median '
        'reference ET volume was selected, with ties resolved by case ID. '
        '(b) PED-00001 has empty ET predictions for all B0 seeds and a nonempty AdultInit prediction '
        'at seed 2026 (ET Dice 0.174). This examination is the median seed-2026 AdultInit ET Dice '
        'among the three eligible examinations; nonempty prediction does not establish complete '
        'or accurate recovery. AdultInit seeds 2027 and 2028 also omit ET in this examination. '
        '(c) PED-00055 is nearest the pair of method-specific median ET Dice values among the '
        '141 reference-positive examinations with nonempty predictions from both methods at '
        'seed 2026, using squared Euclidean distance. For every row, the axial slice maximizes '
        'reference ET area, using the central index for tied maxima. A common square crop includes '
        'all reference and seed-2026 predicted WT on that slice plus a 12-mm margin, with a minimum '
        'side of 96 mm. Intensity windows use the 1st and 99.5th percentiles of positive intensities '
        'in the whole T1 image and are identical across each row. Display is canonical RAS, with '
        'patient right on image right. Selection used numeric rules before visual inspection; '
        'all displayed and companion-seed masks were checked against the original metric tables. '
        'These examples illustrate model outputs and dataset annotations, without new clinical '
        'adjudication or population-level inference.\n')
    (OUT/'Figure10_mri_caption.txt').write_text(caption)
    provenance = {
        'schema_version': 1, 'unit': 'MRI examination', 'displayed_seed': 2026,
        'displayed_prediction_type': 'single-seed hard segmentation; not an ensemble',
        'labels': {'ET': [1], 'WT': [1, 2, 3, 4]},
        'font': 'DejaVu Serif', 'figure_inches': [7.2, 7.6], 'png_dpi': 600,
        'minimum_font_pt': 8, 'orientation': 'Canonical RAS; neurological display, patient right on image right',
        'slice_rule': 'Maximum reference ET area, central index of tied maxima',
        'crop_rule': 'Union bounding box of reference, B0 seed2026 and AdultInit seed2026 WT in slice; '
                     'square side maximum of 96 mm and bounding-box long axis plus 24 mm, centered and clipped to image bounds',
        'window_rule': 'Whole-image positive T1 voxel 1st and 99.5th percentiles, shared within row',
        'selection_table_sha256': sha(membership_path), 'prediction_manifest_sha256': sha(manifest_path),
        'script_sha256': sha(Path(__file__)), 'source_images_are_real': True, 'clinical_adjudication': False,
        'selection_before_visual_inspection': True,
        'programmatic_QA': {'source_hashes_match_prediction_manifest': True,
                           'geometry_and_metrics_checked_prediction_count': 18,
                           'figure_text_overlap_count': 0, 'figure_text_clipping_count': 0},
        'cases': records}
    (OUT/'Figure10_mri_data.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps({'cases': [r['case_id'] for r in records],
                      'crop_sides_mm': [r['crop_side_mm'] for r in records],
                      'verified_prediction_masks': 18, 'outputs': str(OUT)}, indent=2))


if __name__ == '__main__':
    main()
