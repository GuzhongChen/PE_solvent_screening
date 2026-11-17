"""Write manuscript tables and numerical data for the figures."""

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from analysis import GROUPS, report_conversion

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = {
    'Matern52_GP': 'Matérn GP',
    'Decision_tree': 'Decision tree',
    'Extra_trees': 'Extra trees',
    'Gradient_boosting': 'Gradient boosting',
    'kNN': 'k-nearest neighbors',
    'Kernel_ridge': 'Kernel ridge, linear',
    'Linear_SVR': 'Linear support vector',
    'Random_forest': 'Random forest',
    'Ridge': 'Ridge',
}



def read(path):
    return pd.read_csv(ROOT / path)


def interval(mean, sd):
    limits = np.asarray([mean - norm.ppf(.95) * sd, mean + norm.ppf(.95) * sd])
    return report_conversion(limits)


def save_table(directory, definitions, key, rows):
    with (directory / f'Table_{key}.csv').open('w', encoding='utf-8', newline='') as handle:
        csv.writer(handle).writerows([definitions[key]['headers']] + rows)


def main():
    table_dir = ROOT / 'tables'
    figure_dir = ROOT / 'figure_data'
    table_dir.mkdir(exist_ok=True)
    figure_dir.mkdir(exist_ok=True)
    definitions = json.loads((ROOT / 'data/table_definitions.json').read_text(encoding='utf-8'))
    names = definitions['solvent_names']
    definitions = definitions['tables']
    cohort = read('data/selected45.csv')
    models = read('results/model_summary.csv').set_index('model')
    selected = read('results/batches/selected15.csv')
    learning = read('results/learning_summary.csv').set_index('model')
    pairs = read('results/molecular_pair_predictions.csv')
    importance = read('results/group_importance.csv').set_index('group')

    save_table(table_dir, definitions, 'M1', [
        [str(row.ID), row.CAS, names[row.CAS], f'{row.Conversion_pct:g}',
         str(row.calculation_batch)] for row in cohort.itertuples()
    ])
    for key in ('M3', 'M4', 'M5'):
        save_table(table_dir, definitions, key, definitions[key]['rows'])
    save_table(table_dir, definitions, 'M6', [
        [label, f'{models.loc[key, "RMSE_mean"]:.2f} ± {models.loc[key, "RMSE_std"]:.2f}',
         f'{models.loc[key, "MAE_mean"]:.2f}', f'{models.loc[key, "R2_mean"]:.3f}']
        for key, label in MODEL_NAMES.items()
    ])
    save_table(table_dir, definitions, 'M7', [
        [str(row.batch), str(row.slot), names[row.CAS], f'{row.observed_pct:g}',
         f'{row.batch_start_prediction:.2f}', f'{row.KB_sd_latent:.2f}',
         f'{row.KB_EI_pp:.3g}'] for row in selected.itertuples()
    ])
    save_table(table_dir, definitions, 'M8', [
        [str(size), f'{learning.loc[f"n{size}", "R2_mean"]:.3f}',
         f'{learning.loc[f"n{size}", "RMSE_mean"]:.2f} ± {learning.loc[f"n{size}", "RMSE_std"]:.2f}',
         f'{learning.loc[f"n{size}", "MAE_mean"]:.2f} ± {learning.loc[f"n{size}", "MAE_std"]:.2f}']
        for size in (18, 27, 36)
    ])
    pair_rows = []
    for number, (_, group) in enumerate(pairs.groupby('pair', sort=False), 1):
        pair_rows.extend([
            [str(number), names[row.CAS], f'{row.observed:g}',
             f'{row.prediction:.2f}', f'{row.sd_observation:.2f}']
            for row in group.itertuples()
        ])
    save_table(table_dir, definitions, 'M9', pair_rows)
    save_table(table_dir, definitions, 'M10', [
        labels + [f'{importance.loc[group, "mean_absolute_contribution_pp"]:.2f}']
        for labels, group in zip(definitions['M10']['row_labels'], GROUPS)
    ])
    (table_dir / 'Captions.txt').write_text(
        '\n\n'.join(item['caption'] for item in definitions.values()) + '\n', encoding='utf-8')

    for result, figure in [
        ('validation_predictions', 'Figure1b_predictions'),
        ('identity_predictions', 'Figure1b_identity_means'),
        ('learning_repeat_metrics', 'Figure1e_repeat_metrics'),
        ('learning_predictions', 'Figure1e_all_predictions'),
        ('molecular_pair_predictions', 'Figure1f_pairs'),
        ('identity_grouped_shapley', 'Figure_M4_contributions'),
    ]:
        read(f'results/{result}.csv').to_csv(figure_dir / f'{figure}.csv', index=False)
    all_models = models.reset_index().sort_values('MAE_mean')
    all_models.to_csv(figure_dir / 'Figure1d_all_regressors.csv', index=False)
    all_models[all_models.model != 'Kernel_ridge'].to_csv(
        figure_dir / 'Figure1d_display.csv', index=False)

    initial = read('results/batches/round_1_full_library_scores.csv')
    initial = initial[~initial.in_training].sort_values('eligible_rank').copy()
    order = selected[selected.batch == 1].set_index('CAS').slot
    initial['first_batch_slot'] = initial.CAS.map(order)
    initial['selected_in_first_batch'] = initial.CAS.isin(order.index)
    limits = np.array([interval(row.raw_mean, row.sd_observation) for row in initial.itertuples()])
    initial['lower90_pct'] = limits[:, 0]
    initial['upper90_pct'] = limits[:, 1]
    initial.to_csv(figure_dir / 'Figure_M3_all229.csv', index=False)
    initial.head(10).to_csv(figure_dir / 'Figure1c_and_M3_top10.csv', index=False)
    initial[initial.CAS.isin(['67-66-3', '56-23-5'])][
        ['CAS', 'raw_mean', 'sd_latent', 'EI_pp']
    ].to_csv(figure_dir / 'Figure1a_distributions.csv', index=False)
    batch_scores = {
        batch: read(f'results/batches/round_{batch}_full_library_scores.csv').set_index('CAS')
        for batch in (1, 2, 3)
    }
    batch_rows = []
    for row in selected.itertuples():
        prediction = batch_scores[row.batch].loc[row.CAS]
        lower, upper = interval(prediction.raw_mean, prediction.sd_observation)
        batch_rows.append({
            'batch': row.batch, 'slot': row.slot, 'CAS': row.CAS,
            'prediction': prediction.prediction, 'lower90_pct': lower,
            'upper90_pct': upper, 'observed_pct': row.observed_pct,
        })
    pd.DataFrame(batch_rows).to_csv(figure_dir / 'Figure_M1.csv', index=False)



if __name__ == '__main__':
    main()
