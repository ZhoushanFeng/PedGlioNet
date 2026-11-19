"""Verify numerical acceptance criteria against independently archived figure inputs."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from reproduce import DATA, ROOT, ensemble, read_pairs


def read_json(path):
    return json.loads(path.read_text())


def verify_omissions(paired):
    et = paired.loc[paired.region.eq('ET')].set_index(['case_id', 'seed'])
    members = read_json(DATA / 'omission_membership.json')['cases']
    assert len(members) == len({r['case_id'] for r in members}) == 257
    historical = ensemble().set_index(['case_id', 'region'])
    replanned = ensemble(development_only=True).set_index(['case_id', 'region'])
    new_seeds = {}
    for seed in (2026, 2027, 2028):
        frames = []
        for fold in range(3):
            folder = 'eval' if fold == 1 else 'eval_source_planned'
            stem = 'b0' if fold == 1 else 'b0source'
            frames.append(pd.read_csv(DATA / folder / f'{stem}_fold{fold}_seed{seed}_per_case.csv'))
        rows = pd.concat(frames, ignore_index=True)
        new_seeds[seed] = rows.loc[rows.region.eq('ET')].set_index('case_id')
    for case in members:
        identifier = case['case_id']
        for recipe, scores in [('historical', historical), ('development_plan', replanned)]:
            for region in ['ET', 'WT']:
                np.testing.assert_allclose(scores.loc[(identifier, region), 'dice'], case[f'ensemble_{region.lower()}_dice'][recipe])
        for seed in new_seeds:
            pair = et.loc[(identifier, seed)]
            assert int(pair.fold) == case['fold'] and pair.source == case['source']
            assert bool(pair.reference_nonempty) == case['reference_et_positive']
            assert float(pair.reference_volume_mm3) == case['reference_et_volume_mm3']
            planned = new_seeds[seed].loc[identifier]
            values = {'b0': (pair.prediction_nonempty_b0, pair.dice_b0),
                      'adult_init': (pair.prediction_nonempty_a1ft, pair.dice_a1ft),
                      'development_plan': (planned.prediction_nonempty, planned.dice)}
            for key, (present, dice) in values.items():
                assert bool(present) == case['seed_et_presence'][key][str(seed)]
                np.testing.assert_allclose(dice, case['seed_et_dice'][key][str(seed)])
        for key, presence in case['seed_et_presence'].items():
            assert (not any(presence.values())) == case['all_three_seed_et_empty'][key]


def main():
    paired = read_pairs()
    verify_omissions(paired)
    saved = read_json(ROOT / 'output/adult_transfer_replication.json')
    expected = {'three_seed_mean_case_dice': [.5897, .6086, .0189, .0054, .0322],
                'new_seeds_mean_case_dice': [.0172, .0007, .0327]}
    for label, values in expected.items():
        row = next(r for r in saved[label] if r['scope']=='pooled' and r['region']=='ET' and r['subset']=='positive')
        assert row['n_cases'] == 171
        effect = row['paired_difference_adult_minus_b0']
        actual = [effect['mean'], *effect['ci95']]
        if label == 'three_seed_mean_case_dice':
            actual = [row['b0_dice']['mean'], row['adult_init_dice']['mean']] + actual
        np.testing.assert_allclose(actual, values, atol=5e-5, rtol=0)
    computed = read_json(ROOT / 'output/planning_dice.json')
    frozen = read_json(ROOT / 'figure_data/planning_reference.json')
    for row in frozen['paired_dice']:
        matches = [r for r in computed if (r['region'], r['scope'], r['subset']) == (row['region'],row['scope'],row['subset'])]
        assert len(matches) == 1
        comp = matches[0]
        assert comp['n'] == row['n']
        for key in ['historical_mean','development_only_mean']:
            np.testing.assert_allclose(comp[key],row[key],rtol=1e-12,atol=1e-12)
        for key in ['value','ci_low','ci_high']:
            np.testing.assert_allclose(comp['difference_new_minus_historical'][key],row['difference_new_minus_historical'][key],rtol=1e-12,atol=1e-12)
    print('Verified 257 omission records, both ET-positive transfer endpoints and four reported planning contrasts.')


if __name__ == '__main__':
    main()
