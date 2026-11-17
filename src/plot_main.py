"""Draw Figure 1 from the calculated predictions and measured responses."""

from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.transforms import Affine2D
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from svgpath2mpl import parse_path

from assemble_main import assemble
from plot_style import C, LIB, OUT, SHORT, axat, color, endpoints_legend, fig, read, txt


def pair_molecule(figure, cas, x, y):
    width, height = 40, 38
    axis = axat(figure, x, y, width, height)
    axis.set_axis_off()
    axis.set(xlim=(0, 180), ylim=(140, 0))
    axis.set_aspect('equal')
    drawer = rdMolDraw2D.MolDraw2DSVG(180, 140)
    options = drawer.drawOptions()
    options.useBWAtomPalette()
    options.clearBackground = False
    options.bondLineWidth = 2.1
    options.padding = .18
    options.minFontSize = options.maxFontSize = 44
    drawer.DrawMolecule(Chem.MolFromSmiles(LIB.loc[cas, 'SMILES']))
    drawer.FinishDrawing()
    paths = [(element, parse_path(element.get('d')))
             for element in ET.fromstring(drawer.GetDrawingText()).iter()
             if element.tag.endswith('path') and element.get('d')]
    subscript = None
    if cas == '75-12-7':
        atom_paths = [path for element, path in paths
                      if element.get('class', '').startswith('atom-')]
        subscript = min(atom_paths, key=lambda path: path.get_extents().height)
    for element, path in paths:
        if path is subscript:
            bounds = path.get_extents()
            cx, cy = (bounds.x0 + bounds.x1) / 2, (bounds.y0 + bounds.y1) / 2
            path = Affine2D().translate(-cx, -cy).scale(1.3).translate(cx, cy).transform_path(path)
        style = dict(part.split(':', 1) for part in element.get('style', '').split(';') if ':' in part)
        line_width = float(style.get('stroke-width', '0').replace('px', ''))
        axis.add_patch(PathPatch(
            path, facecolor=style.get('fill', element.get('fill', '#000000')),
            edgecolor=style.get('stroke', 'none'), lw=line_width * width / 180,
            capstyle='round', joinstyle='round',
        ))


def main():
    figure = fig(504, 661)
    predictions = read('figure_data/Figure1b_predictions.csv')
    identities = read('figure_data/Figure1b_identity_means.csv')
    solvents = read('data/selected45.csv').set_index('CAS')
    txt(figure, 3, 146, 'b  Predictions for test solvents', 10, 'bold')
    b = axat(figure, 38, 177, 182, 112)
    colors = [C['blue'] if solvents.loc[cas, 'Class'] == 'Halogenated'
              else C['purple'] if solvents.loc[cas, 'Class'] == 'Amides' else C['teal']
              for cas in predictions.CAS]
    b.scatter(predictions.observed, predictions.prediction, s=5, c=colors,
              alpha=.18, edgecolors='none')
    colors = [C['blue'] if row.Class == 'Halogenated'
              else C['purple'] if row.Class == 'Amides' else C['teal']
              for row in identities.itertuples()]
    b.scatter(identities.observed, identities.prediction, s=14, c=colors,
              edgecolors='white', lw=.35, zorder=3)
    b.plot([-5, 105], [-5, 105], ls='--', c='#888888', lw=.65)
    b.set(xlim=(-5, 105), ylim=(-5, 105), xticks=[0, 50, 100], yticks=[0, 50, 100],
          xlabel='Measured PE conversion (%)', ylabel='Predicted PE conversion (%)')
    models = read('figure_data/Figure1d_all_regressors.csv').set_index('model')
    metric = models.loc['Matern52_GP']
    b.text(.035, .96, f'R² = {metric.R2_mean:.3f}\nMAE = {metric.MAE_mean:.1f} pp',
           transform=b.transAxes, va='top', fontsize=8.3)
    txt(figure, 39, 162, '45 solvents × 10 repeats = 450 predictions', 8.3)

    txt(figure, 244, 146, 'c  Initial EI ranking before round 1', 10, 'bold')
    top = read('figure_data/Figure1c_and_M3_top10.csv')
    c = axat(figure, 324, 177, 118, 112)
    ei = axat(figure, 462, 177, 34, 112)
    for index, row in enumerate(top.itertuples()):
        y = 9 - index
        tone = color(row.CAS)
        selected = bool(row.selected_in_first_batch)
        c.errorbar(row.prediction, y,
                   xerr=[[row.prediction - row.lower90_pct], [row.upper90_pct - row.prediction]],
                   fmt='o', mfc=tone if selected else 'white', mec=tone, c=tone,
                   ms=3.5, lw=.7, capsize=1.3)
        ei.barh(y, row.EI_pp, height=.55, color=tone if selected else 'white',
                edgecolor=tone, lw=.6)
    names = [f'{index + 1}  {SHORT.get(row.CAS, row.Name)}'
             for index, row in enumerate(top.itertuples())]
    c.set(xlim=(-3, 103), ylim=(-.6, 9.6), xticks=[0, 50, 100], yticks=range(10),
          yticklabels=names[::-1], xlabel='Predicted PE (%)')
    c.spines['left'].set_visible(False)
    c.tick_params(axis='y', length=0, pad=4)
    ei.set(xlim=(0, 1.5), ylim=(-.6, 9.6), xticks=[0, 1.5], yticks=[], xlabel='EI (pp)')
    ei.spines['left'].set_visible(False)
    txt(figure, 250, 163, 'Filled: round 1; whiskers: 90% intervals', 8.3)

    txt(figure, 3, 329, 'd  Regression model comparison', 10, 'bold')
    comparison = read('figure_data/Figure1d_display.csv').sort_values('MAE_mean')
    labels = {'Matern52_GP': 'GP', 'Random_forest': 'Random forest', 'Extra_trees': 'Extra trees',
              'Gradient_boosting': 'Gradient boosting', 'Decision_tree': 'Decision tree',
              'kNN': 'kNN', 'Linear_SVR': 'Linear SVR', 'Ridge': 'Ridge / linear KRR'}
    d = axat(figure, 92, 358, 127, 105)
    for index, row in enumerate(comparison.itertuples()):
        y = 7 - index
        tone = C['blue'] if row.model == 'Matern52_GP' else '#A4B8C9'
        d.barh(y, row.MAE_mean, height=.55, color=tone)
        d.errorbar(row.MAE_mean, y, xerr=row.MAE_std, fmt='none',
                   ecolor='#333333', lw=.6, capsize=1.5)
    d.set(xlim=(0, 32), ylim=(-.6, 7.6), yticks=range(8),
          yticklabels=[labels[name] for name in comparison.model.iloc[::-1]],
          xticks=[0, 10, 20, 30], xlabel='Test MAE (pp)')
    d.tick_params(axis='y', length=0, pad=4)
    d.spines['left'].set_visible(False)
    txt(figure, 39, 346, 'Same training/test sets; mean ± SD, 10 repeats', 8.3)

    txt(figure, 244, 329, 'e  Learning curve on the same test solvents', 10, 'bold')
    learning = read('figure_data/Figure1e_repeat_metrics.csv')
    summary = read('results/learning_summary.csv').set_index('model').loc[['n18', 'n27', 'n36']]
    e = axat(figure, 286, 358, 214, 105)
    for _, group in learning.groupby('repeat'):
        values = group.set_index('model').loc[['n18', 'n27', 'n36']]
        e.plot([18, 27, 36], values.MAE, c=C['blue'], alpha=.12, lw=.6)
    e.errorbar([18, 27, 36], summary.MAE_mean, yerr=summary.MAE_std, c=C['blue'],
               fmt='o-', lw=1.1, ms=3.6, capsize=2)
    e.set(xlim=(16, 38), ylim=(11, 24), xticks=[18, 27, 36], yticks=[12, 18, 24],
          xlabel='Training solvents (from 45 total)', ylabel='Test MAE (pp)')
    txt(figure, 285, 346, '9 test solvents; mean ± SD, 10 repeats', 8.3)

    txt(figure, 3, 501, 'f  Measured and predicted conversion for five solvent pairs', 10, 'bold')
    txt(figure, 39, 517, 'Both solvents are excluded from fitting; the GP uses the other 43 solvents', 8.3)
    pairs = read('figure_data/Figure1f_pairs.csv')
    f = axat(figure, 38, 548, 462, 54)
    f.set(xlim=(-.6, 11.4), ylim=(-6, 112), yticks=[0, 50, 100], xticks=[],
          ylabel='PE conversion (%)')
    group_names = ['C–H substitution', 'C₂ chlorination', 'C₂ saturation',
                   'Amide substitution', 'Positional isomers']
    for group_index, (_, group) in enumerate(pairs.groupby('pair', sort=False)):
        for member_index, row in enumerate(group.itertuples()):
            x = group_index * 2.4 + member_index * 1.05
            tone = color(row.CAS)
            f.plot([x, x], [row.observed, row.prediction], c='#999999', lw=.7)
            f.scatter(x, row.observed, s=16, marker='s', c=tone, zorder=3)
            f.scatter(x, row.prediction, s=22, facecolors='white', edgecolors=tone, lw=.8, zorder=4)
            center = 38 + 462 * (x + .6) / 12
            pair_molecule(figure, row.CAS, center - 20, 605)
            txt(figure, center, 647, SHORT[row.CAS], 8.3, ha='center')
        txt(figure, 38 + 462 * (group_index * 2.4 + 1.125) / 12, 536,
            group_names[group_index], 8.3, ha='center')
        if group_index < 4:
            f.axvline(group_index * 2.4 + 1.725, c='#DDDDDD', lw=.6)
    figure.legend(handles=endpoints_legend(), loc='upper right', bbox_to_anchor=(.993, .227),
                  ncol=2, fontsize=8.3, handletextpad=.3, columnspacing=.8)
    for axis in [b, c, ei, d, e, f]:
        axis.tick_params(axis='both', labelsize=8.3)
        axis.xaxis.label.set_fontsize(8.3)
        axis.yaxis.label.set_fontsize(8.3)
    OUT.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        figure.savefig(directory / 'plots.pdf', metadata={'CreationDate': None, 'ModDate': None})
        figure.savefig(directory / 'plots.svg', metadata={'Date': None})
        assemble(directory / 'plots.svg', directory / 'plots.pdf', OUT)
    plt.close(figure)


if __name__ == '__main__':
    main()
