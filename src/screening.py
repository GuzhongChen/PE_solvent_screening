"""Recompute GP predictions and expected-improvement rankings for the three recorded rounds."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from descriptors import descriptor_frame
from gp import GaussianProcess, JITTER

ROOT = Path(__file__).resolve().parents[1]


def expected_improvement(mean, sd, threshold, response_scale):
    nodes, weights = np.polynomial.legendre.leggauss(64)
    values = threshold + (nodes + 1) * (100 - threshold) / 2
    probabilities = norm.sf((values[None, :] - mean[:, None]) / (sd[:, None] + 1e-6 * response_scale))
    improvement = (100 - threshold) / 2 * np.sum(probabilities * weights, axis=1)
    improvement[sd / response_scale < 0.01] = 0.
    return improvement


def main(refit=False):
    library = pd.read_csv(ROOT / 'data/library259.csv')
    experiments = pd.read_csv(ROOT / 'data/selected45.csv')
    unavailable = set(pd.read_csv(ROOT / 'data/unavailable_candidates.csv').CAS)
    descriptors = descriptor_frame(library)
    output = ROOT / 'results/batches'
    output.mkdir(parents=True, exist_ok=True)
    descriptors.to_csv(ROOT / 'results/library_descriptors.csv', index=False)
    inputs = descriptors.iloc[:, 1:].to_numpy()
    positions = {cas: index for index, cas in enumerate(library.CAS)}
    initial = experiments[experiments.calculation_batch == 0]
    train = np.array([positions[cas] for cas in initial.CAS])
    responses = initial.Conversion_pct.to_numpy()
    skipped, selected, events, summaries = set(), [], [], []
    fitted_parameters = []
    for batch in range(1, 4):
        state = json.loads((ROOT / f'parameters/batches/round_{batch}.json').read_text())
        state = state['parameters']
        gp = GaussianProcess(inputs, train, responses).fit(None if refit else state)
        mean, latent_sd, observation_sd = gp.predict(np.arange(len(library)))
        covariance = gp.latent_covariance(np.arange(len(library)))
        threshold = float(responses.max() + 0.01 * gp.scale)
        ei = expected_improvement(mean, latent_sd, threshold, gp.scale)
        eligible = set(range(len(library))) - set(train) - skipped
        ranks = {index: rank for rank, index in enumerate(sorted(eligible, key=lambda index: (-ei[index], index)), 1)}
        scores = library.copy()
        scores['batch'] = batch
        scores['training_n'] = len(train)
        scores['in_training'] = [index in train for index in range(len(library))]
        scores['skipped_no_record'] = [index in skipped for index in range(len(library))]
        scores['raw_mean'] = mean
        scores['prediction'] = np.clip(mean, 0, 100)
        scores['sd_latent'] = latent_sd
        scores['sd_observation'] = observation_sd
        scores['EI_pp'] = ei
        scores['threshold_pct'] = threshold
        scores['eligible_rank'] = [ranks.get(index, '') for index in range(len(library))]
        scores.to_csv(output / f'round_{batch}_full_library_scores.csv', index=False)
        fitted_parameters.append(dict(batch=batch, train_CAS=library.CAS.iloc[train].tolist(),
                                      **gp.parameters()))
        pending, batch_rows = [], []
        attempt = 0
        while len(pending) < 5:
            pool = set(range(len(library))) - set(train) - skipped - set(pending)
            conditioned_sd = np.sqrt(np.maximum(np.diag(covariance), 0))
            conditioned_ei = expected_improvement(mean, conditioned_sd, threshold, gp.scale)
            order = sorted(pool, key=lambda index: (-conditioned_ei[index], index))
            index = order[0]
            attempt += 1
            solvent = library.iloc[index]
            event = dict(batch=batch, attempt=attempt, slot=len(pending) + 1, library_index=index,
                         CAS=solvent.CAS, Name=solvent.Name, training_n=len(train), eligible_n=len(pool),
                         pending_n=len(pending), batch_start_rank=ranks[index],
                         batch_start_prediction=float(np.clip(mean[index], 0, 100)),
                         batch_start_sd_latent=float(latent_sd[index]),
                         batch_start_sd_observation=float(observation_sd[index]), batch_start_EI_pp=float(ei[index]),
                         KB_raw_mean=float(mean[index]), KB_sd_latent=float(conditioned_sd[index]),
                         KB_EI_pp=float(conditioned_ei[index]), threshold_pct=threshold,
                         status='skipped_no_record' if solvent.CAS in unavailable else 'accepted')
            events.append(event.copy())
            if solvent.CAS in unavailable:
                skipped.add(index)
                continue
            pending.append(index)
            batch_rows.append(event)
            column = covariance[:, index].copy()
            variance = covariance[index, index] + (gp.noise + JITTER) * gp.scale ** 2
            covariance -= np.outer(column, column) / variance
            covariance = (covariance + covariance.T) / 2
            np.fill_diagonal(covariance, np.maximum(np.diag(covariance), 0))
        measured = experiments.set_index('CAS').loc[library.CAS.iloc[pending], 'Conversion_pct'].to_numpy()
        for row, response in zip(batch_rows, measured):
            row['observed_pct'] = float(response)
            row['absolute_error_pp'] = abs(row['batch_start_prediction'] - response)
        selected.extend(batch_rows)
        summaries.append(dict(batch=batch, training_before=len(train), initial_eligible_n=len(eligible),
                              proposed_n=attempt, skipped_n=attempt - 5, accepted_n=5,
                              training_after=len(train) + 5, best_before_pct=float(responses.max()),
                              batch_best_pct=float(measured.max()), best_after_pct=float(max(responses.max(), measured.max())),
                              new_ge90=int(sum(measured >= 90)),
                              cumulative_new_ge90=sum(row['observed_pct'] >= 90 for row in selected),
                              batch_MAE_pp=float(np.mean([row['absolute_error_pp'] for row in batch_rows])),
                              maximum_EI_at_start_pp=float(ei[list(eligible)].max()), threshold_pct=threshold))
        train = np.r_[train, pending]
        responses = np.r_[responses, measured]
        print(f'Round {batch}: ' + ', '.join(row['Name'] for row in batch_rows), flush=True)
    pd.DataFrame(selected).to_csv(output / 'selected15.csv', index=False)
    pd.DataFrame(events).to_csv(output / 'proposals.csv', index=False)
    pd.DataFrame(summaries).to_csv(output / 'round_summary.csv', index=False)
    (output / 'gp_parameters.json').write_text(json.dumps(fitted_parameters, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refit', action='store_true', help='Optimize GP parameters again instead of using the archived parameters.')
    main(parser.parse_args().refit)
