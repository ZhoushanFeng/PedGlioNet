#!/usr/bin/env python3
"""Aggregate corrected P1 results on the server; no case identifiers in output."""
from pathlib import Path
import json
import numpy as np
import pandas as pd


def indices(frame):
    rng = np.random.default_rng(2026)
    groups = [np.flatnonzero(frame.fold.to_numpy() == f) for f in sorted(frame.fold.unique())]
    return np.concatenate([rng.choice(g, (2000, len(g)), replace=True) for g in groups], axis=1)


def ci(point, values):
    finite = np.asarray(values)[np.isfinite(values)]
    return {'value': float(point) if np.isfinite(point) else None,
            'ci_low': float(np.quantile(finite, .025)) if len(finite) else None,
            'ci_high': float(np.quantile(finite, .975)) if len(finite) else None}


def mean_ci(frame, column):
    g = frame.loc[frame[column].notna()]
    if g.empty:
        return {'n': 0, 'value': None, 'ci_low': None, 'ci_high': None}
    x = g[column].to_numpy(dtype=float)
    return {'n': len(g), **ci(x.mean(), x[indices(g)].mean(axis=1))}



def main():
    import argparse
    parser = argparse.ArgumentParser(description="Recalculate the corrected P1 records from native-space, case-level results.")
    parser.add_argument("--cases", required=True, type=Path, help="p1_outer_per_case_corrected.csv")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    data = pd.read_csv(args.cases)
    assert len(data) == 257 and data.case_id.is_unique and not data.smoke.any()
    records = []
    for scope, g in [('pooled', data), *[(f'fold{f}', b) for f, b in data.groupby('fold')]]:
        pos = g.loc[g.b0_ensemble_reference_nonempty].copy()
        neg = g.loc[~g.b0_ensemble_reference_nonempty].copy()
        idx = indices(g)
        roi = {name: mean_ci(pos, name) for name in ['roi_et_coverage', 'oracle_roi_et_coverage']}
        for method in ['b0_ensemble', 'p1_ungated', 'p1_gated']:
            pos['delta_dice'] = pos[f'{method}_dice'] - pos.b0_ensemble_dice
            neg['specificity'] = 1 - neg[f'{method}_prediction_nonempty'].astype(int)
            neg['delta_specificity'] = neg.b0_ensemble_prediction_nonempty.astype(int) - neg[f'{method}_prediction_nonempty'].astype(int)
            ratios = {}
            for metric, numerator, denominator in [('lesion_sensitivity', 'detected_reference_lesions', 'reference_lesions'), ('lesion_ppv', 'correct_predicted_lesions', 'predicted_lesions')]:
                a, b = g[f'{method}_{numerator}'].to_numpy(), g[f'{method}_{denominator}'].to_numpy()
                base_a, base_b = g[f'b0_ensemble_{numerator}'].to_numpy(), g[f'b0_ensemble_{denominator}'].to_numpy()
                with np.errstate(divide='ignore', invalid='ignore'):
                    boot = a[idx].sum(axis=1) / b[idx].sum(axis=1)
                    base_boot = base_a[idx].sum(axis=1) / base_b[idx].sum(axis=1)
                ratios[metric] = ci(a.sum()/b.sum(), boot)
                ratios[metric+'_difference_vs_b0'] = ci(a.sum()/b.sum()-base_a.sum()/base_b.sum(), boot-base_boot)
            records.append({'scope': scope, 'method': method, 'n': len(g),
                            'positive_dice': mean_ci(pos, f'{method}_dice'), 'all_dice': mean_ci(g, f'{method}_dice'),
                            'positive_dice_difference_vs_b0': mean_ci(pos, 'delta_dice'),
                            'negative_specificity': mean_ci(neg, 'specificity'),
                            'specificity_difference_vs_b0': mean_ci(neg, 'delta_specificity'),
                            'negative_fp_mm3': mean_ci(neg, f'{method}_prediction_volume_mm3'),
                            'roi_coverage': roi, **ratios})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"records":records}, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
