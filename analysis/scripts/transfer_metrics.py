"""Frozen three-seed paired analysis and stratified case bootstrap."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

SEEDS = (2026, 2027, 2028)
REGIONS = ('ET', 'NET', 'CC', 'ED', 'TC', 'WT')
BOOTSTRAPS = 2000
BOOTSTRAP_SEED = 2026
KEYS = ['case_id', 'fold', 'region', 'seed']
REFERENCE = ['source', 'reference_nonempty', 'reference_volume_mm3', 'reference_lesions']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_evaluation(path, fold, seed, split):
    data = pd.read_csv(path)
    required = set(KEYS) - {'seed'} | set(REFERENCE) | {
        'dice', 'nsd_1mm', 'prediction_nonempty', 'prediction_volume_mm3'}
    require(required <= set(data.columns), 'missing required metric columns')
    expected = set(split['val'])
    require(set(data.case_id) == expected and len(data) == len(expected) * len(REGIONS),
            'case coverage or row count differs from frozen outer split')
    require(data.fold.eq(fold).all() and data.source.eq(split['held_out_source']).all(),
            'fold or source mismatch')
    require(not data.duplicated(['case_id', 'region']).any(), 'duplicate case-region rows')
    require(all(set(group.region) == set(REGIONS) for _, group in data.groupby('case_id')),
            'each case must have exactly six regions')
    for column in ['dice', 'nsd_1mm']:
        require(data[column].between(0, 1).all(), f'invalid or missing {column}')
    for prefix in ['reference', 'prediction']:
        volume = data[f'{prefix}_volume_mm3']
        presence = data[f'{prefix}_nonempty']
        require(pd.api.types.is_bool_dtype(presence), f'{prefix} presence is not boolean')
        require(np.isfinite(volume).all() and volume.ge(0).all(), f'invalid {prefix} volume')
        require(volume.gt(0).eq(presence).all(), f'{prefix} volume/presence mismatch')
    require(data.reference_lesions.ge(0).all(), 'invalid reference lesion counts')
    both_empty = ~data.reference_nonempty & ~data.prediction_nonempty
    one_empty = data.reference_nonempty != data.prediction_nonempty
    require(data.loc[both_empty, 'dice'].eq(1).all(), 'both-empty Dice convention differs')
    require(data.loc[one_empty, 'dice'].eq(0).all(), 'one-empty Dice convention differs')
    data['seed'] = seed
    return data


def mean_interval(frame, column):
    if frame.empty:
        return {'mean': None, 'ci95': [None, None]}
    frame = frame.sort_values(['fold', 'case_id'])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sums = np.zeros(BOOTSTRAPS)
    for _, group in frame.groupby('fold', sort=True):
        values = group[column].to_numpy(dtype=float)
        draws = rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))
        sums += values[draws].sum(axis=1)
    return {'mean': float(frame[column].mean()),
            'ci95': np.quantile(sums / len(frame), [.025, .975]).tolist()}


def endpoint_role(region, subset):
    if region == 'ET' and subset == 'positive':
        return 'primary_for_this_replication'
    if subset == 'all' and region in ['WT', 'TC']:
        return 'secondary'
    return 'descriptive'


def summarize(frame, aggregation, seed=None):
    rows = []
    for region in REGIONS:
        regional = frame.loc[frame.region.eq(region)]
        for scope, group in [('pooled', regional), *[(f'fold{f}', g) for f, g in regional.groupby('fold')]]:
            for subset, block in [('all', group), ('positive', group.loc[group.reference_nonempty])]:
                rows.append({'aggregation': aggregation, 'seed': seed, 'region': region,
                             'scope': scope, 'source': 'all three sources' if scope == 'pooled' else str(group.source.iloc[0]),
                             'subset': subset, 'endpoint_role': endpoint_role(region, subset),
                             'n_cases': len(block), 'b0_dice': mean_interval(block, 'dice_b0'),
                             'adult_init_dice': mean_interval(block, 'dice_a1ft'),
                             'paired_difference_adult_minus_b0': mean_interval(block, 'difference')})
    return rows


def presence_audit(paired):
    rows = []
    for seed in SEEDS:
        for region in REGIONS:
            frame = paired.loc[paired.seed.eq(seed) & paired.region.eq(region)]
            for scope, group in [('pooled', frame), *[(f'fold{f}', g) for f, g in frame.groupby('fold')]]:
                positive, negative = group.loc[group.reference_nonempty], group.loc[~group.reference_nonempty]
                row = {'seed': seed, 'region': region, 'scope': scope,
                       'positive_cases': len(positive), 'negative_cases': len(negative)}
                for model in ['b0', 'a1ft']:
                    row[model] = {'positive_empty_predictions': int((~positive[f'prediction_nonempty_{model}']).sum()),
                                  'negative_nonempty_predictions': int(negative[f'prediction_nonempty_{model}'].sum()),
                                  'negative_specificity': float((~negative[f'prediction_nonempty_{model}']).mean()) if len(negative) else None}
                rows.append(row)
    return rows


def consensus_audit(paired):
    et = paired.loc[paired.region.eq('ET')]
    consensus = et.groupby(['case_id', 'fold', 'source', 'reference_nonempty'], as_index=False).agg(
        b0_any_present=('prediction_nonempty_b0', 'any'),
        a1ft_any_present=('prediction_nonempty_a1ft', 'any'))
    rows = []
    for scope, group in [('pooled', consensus), *[(f'fold{f}', g) for f, g in consensus.groupby('fold')]]:
        positive, negative = group.loc[group.reference_nonempty], group.loc[~group.reference_nonempty]
        rows.append({'scope': scope, 'positive_cases': len(positive), 'negative_cases': len(negative),
                     'b0_all_three_empty_positive': int((~positive.b0_any_present).sum()),
                     'a1ft_all_three_empty_positive': int((~positive.a1ft_any_present).sum()),
                     'both_methods_all_three_empty_positive': int((~positive.b0_any_present & ~positive.a1ft_any_present).sum()),
                     'b0_all_three_empty_negative': int((~negative.b0_any_present).sum()),
                     'a1ft_all_three_empty_negative': int((~negative.a1ft_any_present).sum())})
    return rows


def analyze(paired):
    per_seed = [row for seed in SEEDS for row in summarize(paired.loc[paired.seed.eq(seed)], 'single_seed', seed)]
    averaged = paired.groupby(['case_id', 'fold', 'source', 'region', 'reference_nonempty'], as_index=False).agg(
        dice_b0=('dice_b0', 'mean'), dice_a1ft=('dice_a1ft', 'mean'), difference=('difference', 'mean'))
    three_seed = summarize(averaged, 'mean_of_three_seed_case_metrics')
    replication = paired.loc[paired.seed.isin(SEEDS[1:])].groupby(
        ['case_id', 'fold', 'source', 'region', 'reference_nonempty'], as_index=False).agg(
            dice_b0=('dice_b0', 'mean'), dice_a1ft=('dice_a1ft', 'mean'), difference=('difference', 'mean'))
    new_seeds = summarize(replication, 'mean_of_new_seed_2027_2028_case_metrics')
    variation = []
    for aggregate in three_seed:
        entries = [r for r in per_seed if all(r[k] == aggregate[k] for k in ['region', 'scope', 'subset'])]
        values = [r['paired_difference_adult_minus_b0']['mean'] for r in entries]
        values = [v for v in values if v is not None]
        variation.append({k: aggregate[k] for k in ['region', 'scope', 'subset', 'n_cases']} | {
            'seed_differences': [{'seed': r['seed'], 'difference': r['paired_difference_adult_minus_b0']['mean']} for r in entries],
            'minimum': min(values) if values else None, 'maximum': max(values) if values else None,
            'sample_standard_deviation': float(np.std(values, ddof=1)) if len(values) > 1 else None,
            'positive_seeds': sum(v > 0 for v in values), 'negative_seeds': sum(v < 0 for v in values)})
    return {'per_seed_dice': per_seed, 'three_seed_mean_case_dice': three_seed,
            'new_seeds_mean_case_dice': new_seeds,
            'descriptive_seed_variation': variation, 'presence_audit': presence_audit(paired),
            'descriptive_et_consensus_absence': consensus_audit(paired)}


