"""Fit eight regression models using the same solvent partitions as the GP."""

import os

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor

from analysis import metric_tables

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260812
GRIDS = {
    "Decision_tree": [dict(max_depth=d, min_samples_leaf=n)
                      for d, n in product((2, 4, None), (1, 2, 4))],
    "Extra_trees": [dict(max_features=f, min_samples_leaf=n)
                    for f, n in product((0.5, 1.0), (1, 2, 4))],
    "Gradient_boosting": [dict(learning_rate=r, max_depth=d)
                          for r, d in product((0.05, 0.1), (1, 2, 3))],
    "kNN": [dict(n_neighbors=n, weights=w)
            for n, w in product((3, 5, 7, 9), ("uniform", "distance"))],
    "Kernel_ridge": [dict(alpha=a) for a in (0.01, 0.1, 1.0, 10.0, 100.0)],
    "Linear_SVR": [dict(C=c, epsilon=e)
                   for c, e in product((0.1, 1.0, 10.0, 100.0), (0.0, 0.1))],
    "Random_forest": [dict(max_features=f, min_samples_leaf=n)
                      for f, n in product((0.5, 1.0), (1, 2, 4))],
    "Ridge": [dict(alpha=a) for a in (0.01, 0.1, 1.0, 10.0, 100.0)],
}
ESTIMATORS = {
    "Decision_tree": (DecisionTreeRegressor, dict(random_state=42)),
    "Extra_trees": (ExtraTreesRegressor, dict(n_estimators=50, random_state=42, n_jobs=1)),
    "Gradient_boosting": (GradientBoostingRegressor, dict(n_estimators=100, random_state=42)),
    "kNN": (KNeighborsRegressor, {}),
    "Kernel_ridge": (KernelRidge, dict(kernel="linear")),
    "Linear_SVR": (LinearSVR, dict(random_state=42, dual=True, tol=1e-4, max_iter=2000000)),
    "Random_forest": (RandomForestRegressor, dict(n_estimators=50, random_state=42, n_jobs=1)),
    "Ridge": (Ridge, dict(random_state=42)),
}


def predict(model, parameters, x, y, train, test):
    mean = x[train].mean(axis=0)
    deviation = x[train].std(axis=0)
    retained = deviation > 1e-10
    inputs = (x[:, retained] - mean[retained]) / deviation[retained]
    target_mean = y[train].mean()
    target_scale = y[train].std()
    estimator, fixed = ESTIMATORS[model]
    fitted = estimator(**fixed, **parameters)
    fitted.fit(inputs[train], (y[train] - target_mean) / target_scale)
    return fitted.predict(inputs[test]) * target_scale + target_mean


def fit_fold(task):
    root, fold, retune = task
    cohort = pd.read_csv(root / "data/selected45.csv")
    descriptors = pd.read_csv(root / "data/descriptors.csv").set_index("CAS")
    x = descriptors.loc[cohort.CAS].to_numpy(float)
    y = cohort.Conversion_pct.to_numpy(float)
    train = np.array(fold["train_indices"])
    test = np.array(fold["test_indices"])
    archived = json.loads((root / "parameters/comparators.json").read_text())
    parameters = {row["model"]: row["parameters"] for row in archived
                  if row["repeat"] == fold["repeat"] and row["fold"] == fold["fold"]}
    inner = [(train[a], train[b])
             for a, b in KFold(3, shuffle=True, random_state=SEED).split(train)]
    predictions, choices, scores = [], [], []
    for model, grid in GRIDS.items():
        selected = parameters[model]
        if retune:
            losses = []
            for index, candidate in enumerate(grid):
                squared_error = 0.0
                for inner_train, inner_test in inner:
                    estimated = predict(model, candidate, x, y, inner_train, inner_test)
                    squared_error += np.square(np.clip(estimated, 0, 100) - y[inner_test]).sum()
                losses.append(squared_error)
                scores.append(dict(model=model, repeat=fold["repeat"], fold=fold["fold"],
                                   grid_index=index, parameters=json.dumps(candidate, sort_keys=True),
                                   inner_RMSE_pp=float(np.sqrt(squared_error / len(train)))))
            selected = grid[int(np.argmin(losses))]
        raw = predict(model, selected, x, y, train, test)
        choices.append(dict(model=model, repeat=fold["repeat"], fold=fold["fold"],
                            parameters=selected))
        for index, value in zip(test, raw):
            predictions.append(dict(model=model, repeat=fold["repeat"], fold=fold["fold"],
                                    ID=int(cohort.ID.iloc[index]), CAS=cohort.CAS.iloc[index],
                                    observed=float(y[index]), raw_mean=float(value),
                                    prediction=float(np.clip(value, 0, 100))))
    return predictions, choices, scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retune", action="store_true", help="Repeat inner three-fold parameter selection.")
    args = parser.parse_args()
    folds = json.loads((args.root / "data/folds.json").read_text())
    predictions, choices, scores = [], [], []
    tasks = [(args.root, fold, args.retune) for fold in folds]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for number, (rows, selected, trials) in enumerate(pool.map(fit_fold, tasks), 1):
            predictions.extend(rows)
            choices.extend(selected)
            scores.extend(trials)
            print(f"Regression comparison {number}/{len(folds)}", flush=True)
    output = args.root / "results"
    output.mkdir(exist_ok=True)
    comparators = pd.DataFrame(predictions).sort_values(["model", "repeat", "fold", "ID"])
    comparators.to_csv(output / "comparator_predictions.csv", index=False)
    gp = pd.read_csv(output / "validation_predictions.csv")
    combined = pd.concat([gp, comparators], ignore_index=True).sort_values(["model", "repeat", "fold", "ID"])
    combined.to_csv(output / "model_predictions.csv", index=False)
    repeats, summary = metric_tables(combined)
    summary = summary.sort_values("MAE_mean")
    repeats.to_csv(output / "model_repeat_metrics.csv", index=False)
    summary.to_csv(output / "model_summary.csv", index=False)
    choices.sort(key=lambda row: (row["model"], row["repeat"], row["fold"]))
    (output / "selected_hyperparameters.json").write_text(json.dumps(choices, indent=2) + "\n")
    if args.retune:
        pd.DataFrame(scores).to_csv(output / "inner_search_scores.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
