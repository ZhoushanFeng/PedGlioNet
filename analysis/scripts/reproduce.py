"""Recalculate paired transfer, planning sensitivity, decomposition and ensemble results."""
import json
from pathlib import Path

import pandas as pd

import decomposition
import planning_metrics
import transfer_metrics as transfer


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'


def read_pairs():
    splits = json.loads((DATA / 'folds.json').read_text())['outer_folds']
    models = []
    for model in ('b0', 'a1ft'):
        runs = []
        for seed in transfer.SEEDS:
            for split in splits:
                fold = split['fold']
                path = DATA / 'eval' / f'{model}_fold{fold}_seed{seed}_per_case.csv'
                runs.append(transfer.read_evaluation(path, fold, seed, split))
        data = pd.concat(runs, ignore_index=True)
        data['negative_false_positive'] = (~data.reference_nonempty & data.prediction_nonempty).astype(float)
        data['positive_empty'] = (data.reference_nonempty & ~data.prediction_nonempty).astype(float)
        models.append(data)
    paired = models[0].merge(models[1], on=transfer.KEYS, suffixes=('_b0', '_a1ft'), validate='one_to_one')
    transfer.require(len(paired) == 257 * 6 * 3, 'Incomplete model-by-case coverage')
    for col in transfer.REFERENCE:
        transfer.require(paired[f'{col}_b0'].eq(paired[f'{col}_a1ft']).all(), f'Inconsistent {col}')
        transfer.require(paired.groupby(transfer.KEYS[:-1])[f'{col}_b0'].nunique().eq(1).all(), f'Seed mismatch in {col}')
        paired[col] = paired[f'{col}_b0']
    paired['difference'] = paired.dice_a1ft - paired.dice_b0
    return paired


def ensemble(development_only=False):
    frames = []
    for fold in range(3):
        folder = 'eval_source_planned' if development_only and fold != 1 else 'eval'
        frames.append(pd.read_csv(DATA / folder / f'ensemble_b0_fold{fold}_per_case.csv'))
    return pd.concat(frames, ignore_index=True)


def run():
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    paired = read_pairs()
    summary = transfer.analyze(paired)
    (out / 'adult_transfer_replication.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    for key in ('three_seed_mean_case_dice','new_seeds_mean_case_dice'):
        row = next(r for r in summary[key] if r['scope'] == 'pooled' and r['region'] == 'ET' and r['subset'] == 'positive')
        print(key, 'n=', row['n_cases'], 'ET+ delta=', row['paired_difference_adult_minus_b0'])
    pair = paired.rename(columns={c:c.replace('_a1ft','_adult') for c in paired if c.endswith('_a1ft')})
    rows = []
    for seed in transfer.SEEDS:
        rows.extend(decomposition.summarize_scopes(pair.loc[pair.seed.eq(seed)], [seed], 'single_seed', f'seed{seed}', 'per_seed_dice', seed))
    for seeds, prefix, key in [(transfer.SEEDS[1:], 'new_seeds_mean', 'new_seeds_mean_case_dice'), (transfer.SEEDS, 'three_seeds_mean', 'three_seed_mean_case_dice')]:
        means = decomposition.collapse_seeds(pair.loc[pair.seed.isin(seeds)], seeds)
        rows.extend(decomposition.summarize_scopes(means, seeds, f'per_case_mean_of_{len(seeds)}_seed_metrics', prefix, key))
    (out / 'decomposition.json').write_text(json.dumps({'results': rows}, indent=2, allow_nan=False) + '\n')
    historical = ensemble()
    planning = planning_metrics.paired_dice(historical, ensemble(development_only=True))
    (out / 'planning_dice.json').write_text(json.dumps(planning, indent=2, allow_nan=False) + '\n')
    print('Historical ensemble mean Dice:', historical.groupby('region').dice.mean().round(6).to_dict())


if __name__ == '__main__':
    run()
