"""Calculate the manuscript data and draw Figure 1 and Figures M1, M3 and M4."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refit', action='store_true',
                        help='Repeat GP optimization and regression parameter selection.')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    environment = os.environ.copy()
    for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        environment[variable] = '1'
    steps = [
        ('analysis.py', ['--refit'] if args.refit else []),
        ('screening.py', ['--refit'] if args.refit else []),
        ('compare_models.py', ['--workers', str(args.workers)]
         + (['--retune'] if args.refit else [])),
        ('export_tables.py', []),
        ('plot_main.py', []),
        ('plot_si.py', []),
    ]
    for script, options in steps:
        print(f'Running {script}', flush=True)
        subprocess.run([sys.executable, str(ROOT / 'src' / script), *options],
                       cwd=ROOT, env=environment, check=True)
    captions = json.loads((ROOT / 'data/figure_captions.json').read_text(encoding='utf-8'))
    (ROOT / 'figures/Captions.txt').write_text(
        '\n\n'.join(captions.values()) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
