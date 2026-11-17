"""Recompute validation, learning curves, structural attribution and solvent comparisons."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from descriptors import descriptor_frame
from gp import GaussianProcess

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {
    'Element counts': list(range(10)),
    'Carbon bonding and halogen environments': list(range(10, 21)),
    'Donor and functional motifs': list(range(21, 38)),
    'H/X pathway flags': [38, 39],
}
PAIRS = [
    ('67-66-3', '56-23-5', 'Chloromethane substitution'),
    ('79-34-5', '76-01-7', 'C2 chlorination'),
    ('79-34-5', '127-18-4', 'C2 saturation'),
    ('75-12-7', '123-39-7', 'Amide N substitution'),
    ('75-34-3', '107-06-2', 'Dichloroethane isomers'),
]


def metric_tables(predictions):
    rows = []
    for (model, repeat), group in predictions.groupby(['model', 'repeat'], sort=False):
        errors = group.prediction.to_numpy() - group.observed.to_numpy()
        denominator = np.sum((group.observed - group.observed.mean()) ** 2)
        rows.append(dict(model=model, repeat=repeat, R2=1 - np.sum(errors ** 2) / denominator,
                         RMSE=np.sqrt(np.mean(errors ** 2)), MAE=np.mean(abs(errors))))
    repeated = pd.DataFrame(rows)
    summary = repeated.groupby('model', sort=False)[['R2', 'RMSE', 'MAE']].agg(['mean', 'std'])
    summary.columns = ['_'.join(column) for column in summary.columns]
    return repeated, summary.reset_index()


def prediction_rows(solvents, responses, test, prediction, fold, model='Matern52_GP', **extra):
    mean, latent_sd, observation_sd = prediction
    return [dict(model=model, repeat=fold['repeat'], fold=fold['fold'], ID=int(solvents.ID.iloc[index]),
                 CAS=solvents.CAS.iloc[index], observed=float(responses[index]), raw_mean=float(mu),
                 prediction=float(np.clip(mu, 0, 100)), sd_latent=float(latent),
                 sd_observation=float(observation), **extra)
            for index, mu, latent, observation in zip(test, mean, latent_sd, observation_sd)]


def grouped_shapley(gp, inputs, test, solvents, responses, fold):
    masks = np.zeros((16, inputs.shape[1]), dtype=bool)
    for mask in range(16):
        for group_number, columns in enumerate(GROUPS.values()):
            if mask & (1 << group_number):
                masks[mask, columns] = True
    background = inputs[gp.train]
    rows = []
    for index in test:
        queries = np.where(masks[:, None, :], inputs[index][None, None, :], background[None, :, :])
        values = gp.predict_external(queries.reshape(-1, inputs.shape[1]), uncertainty=False)
        values = values.reshape(16, len(background)).mean(axis=1)
        for number, name in enumerate(GROUPS):
            contribution = 0.
            for mask in range(16):
                if mask & (1 << number):
                    continue
                size = mask.bit_count()
                weight = math.factorial(size) * math.factorial(3 - size) / math.factorial(4)
                contribution += weight * (values[mask | (1 << number)] - values[mask])
            rows.append(dict(repeat=fold['repeat'], fold=fold['fold'], CAS=solvents.CAS.iloc[index],
                             ID=int(solvents.ID.iloc[index]), group=name,
                             contribution_pp=float(contribution), baseline_pp=float(values[0]),
                             raw_prediction=float(values[-1]), observed=float(responses[index]),
                             background_n=len(background)))
    return rows


def report_conversion(values):
    values = np.asarray(values)
    return np.where(values < 5, 0., np.minimum(values, 100.))


def main(refit=False):
    data = ROOT / 'data'
    output = ROOT / 'results'
    output.mkdir(exist_ok=True)
    solvents = pd.read_csv(data / 'selected45.csv')
    descriptors = descriptor_frame(solvents)
    descriptors.to_csv(data / 'descriptors.csv', index=False)
    inputs = descriptors.iloc[:, 1:].to_numpy()
    responses = solvents.Conversion_pct.to_numpy()
    folds = json.loads((data / 'folds.json').read_text())
    states = json.loads((ROOT / 'parameters/gp_folds.json').read_text())
    parameters = {(state['kind'], state['repeat'], state['fold'], state.get('training_n')): state
                  for state in states}
    predictions, learning, shapley = [], [], []
    fitted_parameters = []
    for fold in folds:
        train, test = np.array(fold['train_indices']), np.array(fold['test_indices'])
        state = parameters['full', fold['repeat'], fold['fold'], None]
        gp = GaussianProcess(inputs, train, responses[train]).fit(None if refit else state)
        predictions.extend(prediction_rows(solvents, responses, test, gp.predict(test), fold))
        fitted_parameters.append(dict(kind='full', repeat=fold['repeat'], fold=fold['fold'], **gp.parameters()))
        shapley.extend(grouped_shapley(gp, inputs, test, solvents, responses, fold))
        seed = 2026100500 + 5 * (fold['repeat'] - 1) + fold['fold'] - 1
        shuffled = np.random.default_rng(seed).permutation(train)
        for size in (18, 27, 36):
            if size == 36:
                model = gp
            else:
                subset = shuffled[:size]
                state = parameters['learning', fold['repeat'], fold['fold'], size]
                model = GaussianProcess(inputs, subset, responses[subset]).fit(None if refit else state)
                fitted_parameters.append(dict(kind='learning', repeat=fold['repeat'], fold=fold['fold'],
                                              training_n=size, **model.parameters()))
            learning.extend(prediction_rows(solvents, responses, test, model.predict(test), fold,
                                            model=f'n{size}', training_n=size))
        print(f"Validation repeat {fold['repeat']}, fold {fold['fold']}", flush=True)

    predictions = pd.DataFrame(predictions)
    learning = pd.DataFrame(learning)
    shapley = pd.DataFrame(shapley)
    predictions.to_csv(output / 'validation_predictions.csv', index=False)
    learning.to_csv(output / 'learning_predictions.csv', index=False)
    shapley.to_csv(output / 'grouped_shapley.csv', index=False)
    repeated, summary = metric_tables(learning)
    repeated.to_csv(output / 'learning_repeat_metrics.csv', index=False)
    summary.to_csv(output / 'learning_summary.csv', index=False)
    identity = predictions.groupby('CAS').agg(
        observed=('observed', 'first'), prediction=('prediction', 'mean'),
        raw_prediction=('raw_mean', 'mean'), prediction_repeat_sd=('prediction', 'std'),
        mean_obs_sd=('sd_observation', 'mean')).reset_index()
    identity = solvents[['ID', 'CAS', 'Name', 'SMILES', 'Class']].merge(identity, on='CAS')
    identity.to_csv(output / 'identity_predictions.csv', index=False)
    contribution = shapley.groupby(['CAS', 'group'], sort=False).agg(
        contribution_pp=('contribution_pp', 'mean'), repeat_sd=('contribution_pp', 'std'),
        baseline_pp=('baseline_pp', 'mean'), raw_prediction=('raw_prediction', 'mean')).reset_index()
    contribution.to_csv(output / 'identity_grouped_shapley.csv', index=False)
    importance = contribution.groupby('group').agg(
        mean_absolute_contribution_pp=('contribution_pp', lambda values: np.mean(abs(values))),
        mean_contribution_pp=('contribution_pp', 'mean')).reset_index()
    importance.to_csv(output / 'group_importance.csv', index=False)
    pair_states = json.loads((ROOT / 'parameters/pairs.json').read_text())
    pair_rows, fitted_pairs = [], []
    for first, second, label in PAIRS:
        test = np.array([solvents.CAS.tolist().index(first), solvents.CAS.tolist().index(second)])
        train = np.setdiff1d(np.arange(len(solvents)), test)
        state = next(state for state in pair_states if state['pair'] == label)
        model = GaussianProcess(inputs, train, responses[train]).fit(None if refit else state)
        mean, latent, observation = model.predict(test)
        for index, mu, sd_latent, sd_observation in zip(test, mean, latent, observation):
            solvent = solvents.iloc[index]
            pair_rows.append(dict(pair=label, CAS=solvent.CAS, Name=solvent.Name, SMILES=solvent.SMILES,
                                  observed=float(responses[index]), raw_prediction=float(mu),
                                  prediction=float(np.clip(mu, 0, 100)), sd_latent=float(sd_latent),
                                  sd_observation=float(sd_observation), training_n=len(train), in_selected45=True))
        fitted_pairs.append(dict(pair=label, **model.parameters()))
    pd.DataFrame(pair_rows).to_csv(output / 'molecular_pair_predictions.csv', index=False)

    (output / 'gp_parameters.json').write_text(json.dumps(fitted_parameters, indent=2))
    (output / 'pair_parameters.json').write_text(json.dumps(fitted_pairs, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refit', action='store_true', help='Optimize GP parameters again instead of using the archived parameters.')
    main(parser.parse_args().refit)
