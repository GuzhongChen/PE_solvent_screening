# Solvent screening for polyethylene deconstruction

Code and data for *Deciphering Solvent Effects in Low-Temperature Polyolefin Deconstruction via Active-Learning and Mechanistic Elucidation*.

## Run

Use Python 3.12. Install the dependencies, then run the calculation and plotting steps with one command.

```sh
python -m pip install -r requirements.txt
python run.py
```

The outputs are Figure 1 and Figures M1, M3 and M4 in PDF, SVG, PNG and TIFF formats, their numerical data, and Tables M1 and M3–M10. Manuscript figure and table numbers are retained. Figures are written to `figures/`, plot data to `figure_data/`, tables to `tables/` and intermediate calculations to `results/`.

The default run recalculates GP predictions using the supplied fitted covariance parameters and fits the eight comparison regressors using their selected settings. `python run.py --refit` repeats GP optimization and regression parameter selection before drawing the figures. `--workers 4` sets the number of parallel workers for the regression comparison.

## Data

| File or directory | Contents |
| --- | --- |
| `data/selected45.csv` | The 30 initial solvents and three subsequent batches of five |
| `data/library259.csv` | Candidate solvent names, CAS numbers and SMILES |
| `data/unavailable_candidates.csv` | Candidates without a recorded response for reproducing the batch selections |
| `data/descriptors.csv` | Molecular descriptors calculated from SMILES |
| `data/folds.json` | The 50 training/test partitions |
| `parameters/` | Fitted GP covariance parameters and selected regression settings |
| `assets/` | Editable workflow artwork used in Figure 1a |

Solvents and isomers retain their individual CAS records. PE conversion is expressed in percent. Values below 5% are recorded as zero, and exactly 5% is retained. MAE, RMSE and predictive SD are expressed in percentage points. Each solvent pair is predicted after excluding both solvents and fitting the other 43. Shapley contributions use the four descriptor groups and the training molecules as background.

## Code

| Script | Calculation or figure |
| --- | --- |
| `src/descriptors.py` | Molecular graph descriptors |
| `src/gp.py` | Matérn-5/2 GP fitting and prediction |
| `src/analysis.py` | Cross-validation, learning curves, grouped Shapley contributions and solvent comparisons |
| `src/screening.py` | Candidate predictions, expected improvement and batch selection |
| `src/compare_models.py` | Eight regression methods on the same training/test partitions |
| `src/export_tables.py` | Figure data and SI tables |
| `src/plot_main.py` | Figure 1 |
| `src/plot_si.py` | Figures M1, M3 and M4 |
| `src/plot_style.py` | Shared figure styling and molecular drawings |
| `src/assemble_main.py` | Combines the Figure 1a artwork with panels b–f |

