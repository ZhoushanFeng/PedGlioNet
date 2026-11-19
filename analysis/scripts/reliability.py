#!/usr/bin/env python3
"""Post-hoc reliability analysis specified in ANALYSIS_ADDENDUM_2026-09-13.md.

Only previously generated held-out predictions/metrics are read. No model or
decision threshold is fitted. Source-stratified bootstrap keeps paired scores.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]

BOOT = 2000
COVERAGES = [.5, .8, .9, 1.]


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def interval(point, samples):
    samples = np.asarray(samples, dtype=float)
    finite = samples[np.isfinite(samples)]
    return {'value': point, 'ci_low': np.quantile(finite, .025) if len(finite) else None,
            'ci_high': np.quantile(finite, .975) if len(finite) else None,
            'valid_bootstrap': len(finite)}


def bootstrap_indices(frame):
    rng = np.random.default_rng(2026)
    fold = frame.fold.to_numpy()
    groups = [np.flatnonzero(fold == f) for f in sorted(np.unique(fold))]
    return np.concatenate([rng.choice(g, size=(BOOT, len(g)), replace=True) for g in groups], axis=1)


def auc(y, score):
    n1 = int(y.sum())
    n0 = len(y) - n1
    if not n1 or not n0:
        return np.nan
    return float((rankdata(score)[y].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def average_precision(y, score):
    if not y.any():
        return np.nan
    order = np.argsort(-score, kind='stable')
    s, y = score[order], y[order]
    ends = np.r_[np.flatnonzero(np.diff(s) != 0), len(y) - 1]
    tp = np.cumsum(y)[ends]
    return float(np.sum(np.diff(np.r_[0, tp]) / y.sum() * tp / (ends + 1)))


def ordered_expected_losses(score, *loss_arrays):
    """Expected retained loss under random ordering within tied score blocks."""
    order = np.argsort(score, kind='stable')
    ordered_score = score[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ordered_score) != 0) + 1]
    lengths = np.diff(np.r_[starts, len(order)])
    arrays = []
    for loss in loss_arrays:
        sorted_loss = np.asarray(loss, dtype=float)[order]
        block_means = np.add.reduceat(sorted_loss, starts) / lengths
        arrays.append(np.repeat(block_means, lengths))
    return arrays


def risk_summary(score, dice, failures):
    ordered_loss, ordered_failures = ordered_expected_losses(score, 1. - dice, failures)
    risk = np.cumsum(ordered_loss) / np.arange(1, len(dice) + 1)
    fail_risk = np.cumsum(ordered_failures) / np.arange(1, len(dice) + 1)
    values = [float(risk.mean())]
    for coverage in COVERAGES:
        k = max(1, round(coverage * len(dice)))
        values.extend([float(risk[k - 1]), float(fail_risk[k - 1])])
    return np.array(values)


def mean_interval(frame, col):
    d = frame.loc[np.isfinite(frame[col])]
    if d.empty:
        return {'n': 0, 'mean': None, 'ci_low': None, 'ci_high': None}
    x = d[col].to_numpy()
    boot = x[bootstrap_indices(d)].mean(axis=1)
    return {'n': len(x), 'mean': float(x.mean()), 'ci_low': float(np.quantile(boot, .025)), 'ci_high': float(np.quantile(boot, .975))}


def main(development_planning=False):
    folder = ROOT / 'data'
    OUT = ROOT / 'output' / ('planning_reliability' if development_planning else 'reliability')
    OUT.mkdir(parents=True, exist_ok=True)
    files, unc_files, seeds = [], [], []
    for fold in range(3):
        use_development = development_planning and fold != 1
        metric_dir = folder / ('eval_source_planned' if use_development else 'eval')
        uncertainty_dir = folder / ('uncertainty_source_planned' if use_development else 'uncertainty')
        files.append(metric_dir / f'ensemble_b0_fold{fold}_per_case.csv')
        unc_files.append(uncertainty_dir / f'b0_fold{fold}_uncertainty_per_case.csv')
        prefix = 'b0source' if use_development else 'b0'
        seeds.extend(metric_dir / f'{prefix}_fold{fold}_seed{seed}_per_case.csv' for seed in (2026, 2027, 2028))
    data = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    unc = pd.concat([pd.read_csv(p) for p in unc_files], ignore_index=True)
    assert len(data) == 257 * 6 and len(unc) == 257 and unc.case_id.is_unique
    splits = json.loads((ROOT / 'data/folds.json').read_text())
    for fold in splits['outer_folds']:
        assert set(data.loc[data.fold.eq(fold['fold']), 'case_id']) == set(fold['val'])

    region_summary = []
    sizes = []
    for region, region_df in data.groupby('region'):
        for scope, g in [('pooled', region_df), *[(f'fold{f}', b) for f, b in region_df.groupby('fold')]]:
            positive = g.loc[g.reference_nonempty]
            negative = g.loc[~g.reference_nonempty]
            for subset, block in [('all', g), ('positive', positive)]:
                for metric in ['dice', 'hd95_mm', 'nsd_1mm']:
                    region_summary.append({'region': region, 'scope': scope, 'subset': subset, 'metric': metric,
                                           'total_cases': len(block), **mean_interval(block, metric)})
            region_summary.append({'region': region, 'scope': scope, 'subset': 'empty_audit', 'metric': 'counts',
                                   'cases': len(g), 'reference_positive': len(positive), 'reference_negative': len(negative),
                                   'both_empty': int((~negative.prediction_nonempty).sum()),
                                   'false_negative_empty': int((~positive.prediction_nonempty).sum()),
                                   'false_positive_nonempty': int(negative.prediction_nonempty.sum()),
                                   'negative_specificity': float((~negative.prediction_nonempty).mean()) if len(negative) else None,
                                   'reference_lesions': int(g.reference_lesions.sum()),
                                   'lesion_sensitivity': g.detected_reference_lesions.sum() / g.reference_lesions.sum() if g.reference_lesions.sum() else None,
                                   'lesion_ppv': g.correct_predicted_lesions.sum() / g.predicted_lesions.sum() if g.predicted_lesions.sum() else None})
            if len(positive) >= 3:
                correlation = spearmanr(np.log1p(positive.reference_volume_mm3), positive.dice)
                sizes.append({'region': region, 'scope': scope, 'n': len(positive),
                              'spearman_volume_vs_dice': correlation.statistic, 'exploratory_p': correlation.pvalue})
    pd.DataFrame(region_summary).to_csv(OUT / 'regional_metrics_and_empty_audit.csv', index=False)
    pd.DataFrame(sizes).to_csv(OUT / 'exploratory_region_size_associations.csv', index=False)

    wt_vol = data.loc[data.region.eq('WT'), ['case_id', 'prediction_volume_mm3']].rename(columns={'prediction_volume_mm3': 'predicted_wt_volume'})
    targets = {'WT': data.loc[data.region.eq('WT')],
               'ET_positive': data.loc[data.region.eq('ET') & data.reference_nonempty],
               'ET_all': data.loc[data.region.eq('ET')]}
    results, curves, comparisons = [], [], []
    for target, target_df in targets.items():
        target_df = target_df.merge(unc, on=['case_id', 'fold'], validate='one_to_one').merge(wt_vol, on='case_id', validate='one_to_one')
        target_df['wt_disagreement'] = 1. - target_df.wt_pairwise_dice
        target_df['et_disagreement'] = 1. - target_df.et_pairwise_dice
        target_df['low_predicted_wt_volume'] = -np.log1p(target_df.predicted_wt_volume)
        target_df['low_predicted_et_volume'] = -np.log1p(target_df.prediction_volume_mm3)
        signals = ['mean_entropy', 'seed_disagreement', 'wt_disagreement', 'wt_volume_cv', 'low_predicted_wt_volume']
        if target.startswith('ET'):
            signals += ['et_disagreement', 'low_predicted_et_volume']
        for scope, g in [('pooled', target_df), *[(f'fold{f}', b) for f, b in target_df.groupby('fold')]]:
            g = g.sort_values('case_id').reset_index(drop=True)
            dice = g.dice.to_numpy()
            indices = bootstrap_indices(g)
            primary_failures = dice < .5
            oracle_score = 1. - dice
            random_risk = float((1. - dice).mean())
            oracle = risk_summary(oracle_score, dice, primary_failures)
            boot_aurc = {}
            point_aurc = {}
            for signal in signals:
                scores = g[signal].to_numpy()
                assert np.isfinite(scores).all()
                risk_point = risk_summary(scores, dice, primary_failures)
                boot_risk = np.array([risk_summary(scores[i], dice[i], primary_failures[i]) for i in indices])
                boot_aurc[signal] = boot_risk[:, 0]
                point_aurc[signal] = risk_point[0]
                for threshold in [.5, .8, .9]:
                    y = dice < threshold
                    boot_auc = [auc(y[i], scores[i]) for i in indices]
                    boot_ap = [average_precision(y[i], scores[i]) for i in indices]
                    results.append({'target': target, 'scope': scope, 'signal': signal, 'failure_threshold': threshold,
                                    'n': len(g), 'failures': int(y.sum()), 'prevalence': float(y.mean()),
                                    'sparse_class_warning': min(int(y.sum()), len(g) - int(y.sum())) < 5,
                                    'auroc': interval(auc(y, scores), boot_auc),
                                    'average_precision': interval(average_precision(y, scores), boot_ap),
                                    'aurc': interval(risk_point[0], boot_risk[:, 0]),
                                    'random_aurc': random_risk, 'oracle_aurc': oracle[0]})
                for j, coverage in enumerate(COVERAGES):
                    k = max(1, round(coverage * len(g)))
                    curves.append({'target': target, 'scope': scope, 'ranking': 'global_within_scope', 'signal': signal,
                                   'nominal_coverage': coverage, 'retained_n': k, 'actual_coverage': k / len(g),
                                   'risk': interval(risk_point[1 + 2*j], boot_risk[:, 1 + 2*j]),
                                   'failure_rate_dice_lt_0p5': interval(risk_point[2 + 2*j], boot_risk[:, 2 + 2*j]),
                                   'random_risk': random_risk, 'oracle_risk': float(oracle[1 + 2*j])})
                if scope == 'pooled':
                    # Equal coverage within every source; no across-source thresholds are learned.
                    for coverage in COVERAGES:
                        loss_sum, fail_sum, keep_total = 0., 0., 0
                        for _, source_rows in g.groupby('fold'):
                            loss, fail = ordered_expected_losses(source_rows[signal].to_numpy(), 1-source_rows.dice.to_numpy(), source_rows.dice.to_numpy() < .5)
                            k = max(1, round(coverage * len(source_rows)))
                            loss_sum += loss[:k].sum(); fail_sum += fail[:k].sum(); keep_total += k
                        curves.append({'target': target, 'scope': scope, 'ranking': 'matched_coverage_within_sources', 'signal': signal,
                                       'nominal_coverage': coverage, 'retained_n': keep_total, 'actual_coverage': keep_total / len(g),
                                       'risk': {'value': loss_sum / keep_total}, 'failure_rate_dice_lt_0p5': {'value': fail_sum / keep_total}})
                print(f'completed target={target} scope={scope} signal={signal}', flush=True)
            reference_signal = 'wt_disagreement' if target == 'WT' else 'et_disagreement'
            for signal in signals:
                comparisons.append({'target': target, 'scope': scope, 'signal': signal, 'reference_signal': reference_signal,
                                    'contrast': 'AURC(signal) minus AURC(reference); lower is better',
                                    **interval(point_aurc[signal] - point_aurc[reference_signal], boot_aurc[signal] - boot_aurc[reference_signal])})

    source_pairs = []
    for target in targets:
        selected = [r for r in results if r['target'] == target and r['scope'] != 'pooled']
        for signal in sorted({r['signal'] for r in selected}):
            for threshold in [.5, .8, .9]:
                rows = [r for r in selected if r['signal'] == signal and r['failure_threshold'] == threshold]
                weighted, total = 0., 0
                for r in rows:
                    weight = r['failures'] * (r['n'] - r['failures'])
                    if weight:
                        weighted += weight * r['auroc']['value']; total += weight
                source_pairs.append({'target': target, 'signal': signal, 'failure_threshold': threshold,
                                     'within_source_pair_weighted_auroc': weighted / total if total else None,
                                     'within_source_positive_negative_pairs': total})

    payload = {'analysis_type': 'post_hoc_addendum_20260913', 'bootstrap': BOOT, 'seed': 2026,
               'failure_definitions': [.5, .8, .9], 'risk': '1-Dice; expected random ordering within exact score ties',
               'caveats': ['No threshold fitted; coverage points are retrospective rankings.',
                           'Intervals conditional on the existing models and sources.',
                           'Very small failure classes can produce degenerate bootstrap intervals; flag them explicitly.'],
               'discrimination': results, 'risk_coverage': curves, 'paired_aurc_comparisons': comparisons,
               'source_conditioned_auroc': source_pairs,
               'input_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files + unc_files + seeds + [ROOT / "data/folds.json", Path(__file__)]}}
    (OUT / 'reliability_analysis.json').write_text(json.dumps(clean(payload), indent=2, allow_nan=False) + '\n')
    flat = []
    for row in results:
        out = {k:v for k,v in row.items() if not isinstance(v, dict)}
        for key in ['auroc', 'average_precision', 'aurc']:
            out.update({f'{key}_{k}':v for k,v in row[key].items()})
        flat.append(out)
    pd.DataFrame(flat).to_csv(OUT / 'reliability_discrimination.csv', index=False)
    pd.DataFrame(comparisons).to_csv(OUT / 'paired_aurc_comparisons.csv', index=False)
    pd.DataFrame(source_pairs).to_csv(OUT / 'source_conditioned_auroc.csv', index=False)
    seed_frames = []
    for p in seeds:
        d = pd.read_csv(p)
        d = d.loc[d.region.eq('ET')].copy()
        d['seed'] = int(p.name.split('seed')[1].split('_')[0])
        seed_frames.append(d)
    seed_data = pd.concat(seed_frames, ignore_index=True)
    consensus = seed_data.groupby(['case_id', 'fold', 'source'], as_index=False).agg(
        reference_nonempty=('reference_nonempty', 'first'), seed_count=('seed', 'nunique'),
        any_seed_positive=('prediction_nonempty', 'any'), all_seed_positive=('prediction_nonempty', 'all'))
    assert (consensus.seed_count == 3).all()
    consensus = consensus.merge(data.loc[data.region.eq('WT'), ['case_id', 'dice']].rename(columns={'dice': 'wt_dice'}), on='case_id', validate='one_to_one')
    consensus = consensus.merge(data.loc[data.region.eq('ET'), ['case_id', 'dice', 'reference_volume_mm3', 'prediction_nonempty']], on='case_id', validate='one_to_one')
    absence = []
    for scope, g in [('pooled', consensus), *[(f'fold{f}', b) for f, b in consensus.groupby('fold')]]:
        pos, neg = g.loc[g.reference_nonempty], g.loc[~g.reference_nonempty]
        missed = pos.loc[~pos.any_seed_positive]
        absence.append({'scope': scope, 'n': len(g), 'reference_positive': len(pos), 'reference_negative': len(neg),
                        'all_seeds_empty_positive': len(missed), 'all_seeds_empty_negative': int((~neg.any_seed_positive).sum()),
                        'positive_failure_dice_lt_0p5': int((pos.dice < .5).sum()),
                        'ensemble_empty_positive': int((~pos.prediction_nonempty).sum()),
                        'consensus_missed_positive_with_wt_dice_ge_0p9': int((missed.wt_dice >= .9).sum()),
                        'consensus_missed_positive_reference_et_volume_median_mm3': float(missed.reference_volume_mm3.median()),
                        'consensus_missed_positive_wt_dice_median': float(missed.wt_dice.median())})
    (OUT / 'consensus_absence_diagnostic.json').write_text(json.dumps(clean({'analysis_type': 'further_exploratory_after_ET_failure_result', 'summary': absence}), indent=2, allow_nan=False) + '\n')
    consensus.to_csv(OUT / 'consensus_absence_case_audit.csv', index=False)
    print('RELIABILITY_ANALYSIS_COMPLETE', flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--development-planning', action='store_true')
    main(parser.parse_args().development_planning)
