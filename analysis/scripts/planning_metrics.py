import numpy as np
import reliability as statistics

def paired_dice(old, new):
    keys = ['case_id', 'fold', 'region']
    assert not old.duplicated(keys).any() and not new.duplicated(keys).any()
    joined = old.merge(new, on=keys, suffixes=('_old', '_new'), validate='one_to_one')
    assert len(joined) == len(old) == len(new) == 257 * 6
    for column in ['source', 'reference_nonempty', 'reference_volume_mm3']:
        assert np.array_equal(joined[column+'_old'], joined[column+'_new']), column
    result = []
    for region, frame in joined.groupby('region'):
        scopes = [('pooled', frame), *[(f'fold{f}', g) for f, g in frame.groupby('fold')]]
        for scope, group in scopes:
            for subset, block in [('all', group), ('positive', group.loc[group.reference_nonempty_old])]:
                if block.empty:
                    continue
                block = block.sort_values('case_id').reset_index(drop=True)
                delta = (block.dice_new - block.dice_old).to_numpy()
                samples = delta[statistics.bootstrap_indices(block)].mean(axis=1)
                result.append({'region': region, 'scope': scope, 'subset': subset, 'n': len(block),
                               'historical_mean': block.dice_old.mean(), 'development_only_mean': block.dice_new.mean(),
                               'difference_new_minus_historical': statistics.interval(delta.mean(), samples)})
    return result


