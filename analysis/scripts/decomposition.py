import numpy as np
from transfer_metrics import require, REFERENCE, REGIONS

BOOTSTRAPS = 2000
RNG_SEED = 2026
MODEL_METRICS = ('dice', 'prediction_volume_mm3', 'predicted_lesions',
                 'correct_predicted_lesions', 'detected_reference_lesions',
                 'negative_false_positive', 'positive_empty')


def collapse_seeds(frame, seeds):
    require(set(frame.seed) == set(seeds), 'wrong seeds supplied for case averaging')
    keys = ['case_id', 'fold', 'region']
    require(frame.groupby(keys).size().eq(len(seeds)).all(), 'unequal seed coverage per case')
    aggregation = {x: 'first' for x in REFERENCE}
    aggregation.update({f'{m}_{model}': 'mean'
                        for m in MODEL_METRICS for model in ['b0', 'adult']})
    return frame.groupby(keys, as_index=False).agg(aggregation)


def finite_number(value):
    return float(value) if np.isfinite(value) else None


def safe_divide(numerator, denominator):
    numerator, denominator = np.broadcast_arrays(numerator, denominator)
    return np.divide(numerator, denominator, out=np.full(numerator.shape, np.nan),
                     where=denominator != 0)


def paired_metric(b0, adult, boot_b0, boot_adult, unit='proportion'):
    delta = np.asarray(boot_adult - boot_b0)
    valid = delta[np.isfinite(delta)]
    return {'b0': finite_number(b0), 'adult_init': finite_number(adult),
            'difference_adult_minus_b0': finite_number(adult - b0),
            'difference_ci95': np.quantile(valid, [.025, .975]).tolist() if len(valid) else [None, None],
            'bootstrap_valid_draws': int(len(valid)), 'unit': unit}


def summarize(frame, scope, seeds, averaging):
    rows = []
    for region in REGIONS:
        data = frame.loc[frame.region.eq(region)].sort_values(['fold', 'case_id']).reset_index(drop=True)
        require(not data.case_id.duplicated().any(), 'case must be bootstrap unit')
        positive = data.reference_nonempty.to_numpy(bool)
        negative = ~positive
        count = len(data)
        npos, nneg = int(positive.sum()), int(negative.sum())
        rng = np.random.default_rng(RNG_SEED)
        draw_blocks = []
        for _, group in data.groupby('fold', sort=True):
            indices = group.index.to_numpy()
            draw_blocks.append(indices[rng.integers(0, len(indices), size=(BOOTSTRAPS, len(indices)))])
        draws = np.concatenate(draw_blocks, axis=1)
        boot_npos, boot_nneg = positive[draws].sum(axis=1), negative[draws].sum(axis=1)

        def sums(values):
            values = np.asarray(values, dtype=float)
            return values.sum(), values[draws].sum(axis=1)

        def metric(column, mask=None, denominator=1, boot_denominator=1, unit='proportion'):
            values = [data[f'{column}_{model}'].to_numpy(float) for model in ['b0', 'adult']]
            if mask is not None:
                values = [v * mask for v in values]
            totals = [sums(v) for v in values]
            return paired_metric(*[safe_divide(t[0], denominator) for t in totals],
                                 *[safe_divide(t[1], boot_denominator) for t in totals], unit=unit)

        b0, adult = [data[f'dice_{model}'].to_numpy(float) for model in ['b0', 'adult']]
        delta = adult - b0
        dpositive, dnegative = delta * positive, delta * negative
        boot_positive = dpositive[draws].mean(axis=1)
        boot_negative = dnegative[draws].mean(axis=1)
        boot_all = delta[draws].mean(axis=1)
        point_residual = float(dpositive.mean() + dnegative.mean() - delta.mean())
        boot_residual = float(np.max(np.abs(boot_positive + boot_negative - boot_all)))
        require(abs(point_residual) < 1e-12 and boot_residual < 1e-12,
                'Dice decomposition identity failed')
        row = {'scope': scope, 'region': region, 'seeds': list(seeds),
               'seed_aggregation': averaging, 'n_unique_cases': count,
               'n_reference_positive': npos, 'n_reference_negative': nneg,
               'source_case_counts': {str(source): int(len(g)) for source, g in data.groupby('source')},
               'dice_all': metric('dice', denominator=count, boot_denominator=count),
               'dice_positive': metric('dice', positive, npos, boot_npos),
               'dice_delta_decomposition': {
                   'positive_contribution': {'estimate': float(dpositive.mean()),
                                            'ci95': np.quantile(boot_positive, [.025, .975]).tolist()},
                   'negative_contribution': {'estimate': float(dnegative.mean()),
                                            'ci95': np.quantile(boot_negative, [.025, .975]).tolist()},
                   'sum_contributions': float(dpositive.mean() + dnegative.mean()),
                   'identity_point_residual': point_residual,
                   'identity_max_abs_bootstrap_residual': boot_residual}}
        fp = metric('negative_false_positive', denominator=nneg, boot_denominator=boot_nneg)
        specificity = {'b0': None if fp['b0'] is None else 1 - fp['b0'],
                       'adult_init': None if fp['adult_init'] is None else 1 - fp['adult_init'],
                       'difference_adult_minus_b0': None if fp['difference_adult_minus_b0'] is None else -fp['difference_adult_minus_b0'],
                       'difference_ci95': [None, None] if fp['difference_ci95'][0] is None else [-fp['difference_ci95'][1], -fp['difference_ci95'][0]],
                       'bootstrap_valid_draws': fp['bootstrap_valid_draws'], 'unit': 'proportion'}
        row['reference_negative_audit'] = {
            'false_positive_cases': metric('negative_false_positive', unit='seed-averaged case count' if len(seeds) > 1 else 'case count'),
            'specificity': specificity,
            'false_positive_volume_total_mm3': metric('prediction_volume_mm3', negative, unit='mm3'),
            'false_positive_volume_mean_per_negative_case_mm3': metric('prediction_volume_mm3', negative, nneg, boot_nneg, 'mm3')}
        row['reference_positive_audit'] = {
            'complete_empty_predictions': metric('positive_empty', unit='seed-averaged case count' if len(seeds) > 1 else 'case count'),
            'complete_empty_prediction_rate': metric('positive_empty', denominator=npos, boot_denominator=boot_npos)}
        component_totals = {model: sums(data[f'predicted_lesions_{model}']) for model in ['b0', 'adult']}
        correct_totals = {model: sums(data[f'correct_predicted_lesions_{model}']) for model in ['b0', 'adult']}
        ppv = paired_metric(*[safe_divide(correct_totals[m][0], component_totals[m][0]) for m in ['b0', 'adult']],
                            *[safe_divide(correct_totals[m][1], component_totals[m][1]) for m in ['b0', 'adult']])
        false_totals = {m: (component_totals[m][0] - correct_totals[m][0],
                            component_totals[m][1] - correct_totals[m][1]) for m in ['b0', 'adult']}
        row['connected_component_audit'] = {
            'ppv': ppv,
            'predicted_components': metric('predicted_lesions', unit='seed-averaged component count' if len(seeds) > 1 else 'component count'),
            'correct_predicted_components': metric('correct_predicted_lesions', unit='seed-averaged component count' if len(seeds) > 1 else 'component count'),
            'false_positive_components': paired_metric(*[false_totals[m][0] for m in ['b0', 'adult']],
                                                       *[false_totals[m][1] for m in ['b0', 'adult']],
                                                       unit='seed-averaged component count' if len(seeds) > 1 else 'component count')}
        rows.append(row)
    return rows


def summarize_scopes(frame, seeds, averaging, prefix, finalizer_key, single_seed=None):
    rows = []
    for fold, block in [(None, frame), *list(frame.groupby('fold'))]:
        scope = 'all_sources' if fold is None else f'fold{fold}_{block.source.iloc[0]}'
        for row in summarize(block, f'{prefix}_{scope}', seeds, averaging):
            row['finalizer_key'] = finalizer_key
            row['finalizer_scope'] = 'pooled' if fold is None else f'fold{fold}'
            row['finalizer_seed'] = single_seed
            rows.append(row)
    return rows


