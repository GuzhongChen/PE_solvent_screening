"""Draw Supporting Information Figures M1, M3 and M4."""

from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

from analysis import GROUPS
from plot_style import (
    C, axat, color, endpoints_legend, fig, label, mol, read, save, title, txt,
)


def axis_fonts(axis):
    axis.tick_params(axis='both', labelsize=7.3)
    axis.xaxis.label.set_size(7.5)
    axis.yaxis.label.set_size(7.5)


def m1():
    figure = fig(504, 410)
    title(figure, 3, 3, 'Predictions before each round and measured conversions')
    figure.legend(handles=endpoints_legend(), loc='upper right',
                  bbox_to_anchor=(.99, .955), ncol=2, fontsize=7.3)
    selected = read('figure_data/Figure_M1.csv')
    for batch, group in selected.groupby('batch', sort=True):
        y = 42 + (batch - 1) * 122
        title(figure, 3, y - 17, f'{"abc"[batch - 1]}  Round {batch}: {30 + 5 * (batch - 1)} training solvents')
        axis = axat(figure, 167, y + 6, 243, 77)
        axis.set_label('abc'[batch - 1])
        for number, row in enumerate(group.itertuples()):
            yy = 4 - number
            shade = color(row.CAS)
            axis.errorbar(row.prediction, yy,
                          xerr=[[row.prediction - row.lower90_pct],
                                [row.upper90_pct - row.prediction]],
                          fmt='o', mfc='white', mec=shade, c=shade,
                          ms=4, lw=.8, capsize=2)
            axis.scatter(row.observed_pct, yy, marker='s', s=16, c=shade, zorder=4)
            text_y = y + 6 + 77 * (4.65 - yy) / 5.3
            txt(figure, 437, text_y, f'{row.prediction:.1f}', 7.5, ha='center', va='center')
            txt(figure, 481, text_y, f'{row.observed_pct:g}', 7.5, ha='center', va='center')
        txt(figure, 437, y - 6, 'Predicted', 7.3, 'bold', ha='center')
        txt(figure, 481, y - 6, 'Measured', 7.3, 'bold', ha='center')
        axis.set(xlim=(-3, 103), ylim=(-.65, 4.65), xticks=[0, 25, 50, 75, 100],
                 yticks=range(5), yticklabels=[label(cas) for cas in group.CAS.iloc[::-1]],
                 xlabel='PE conversion (%)')
        axis_fonts(axis)
        axis.tick_params(axis='y', length=0, pad=5)
        axis.spines['left'].set_visible(False)
    save(figure, 'Figure_M1')


def m3():
    figure = fig(504, 462)
    pool = read('figure_data/Figure_M3_all229.csv')
    top = read('figure_data/Figure1c_and_M3_top10.csv')
    title(figure, 3, 3, 'a  Initial predictions for all 229 untested structures')
    axis = axat(figure, 43, 33, 270, 105)
    axis.set_label('a')
    points = axis.scatter(
        pool.prediction, pool.sd_latent, c=pool.EI_pp,
        cmap=LinearSegmentedColormap.from_list('EI', ['#E9F1F7', C['blue']]),
        s=12, edgecolors='#6E7B86', lw=.2)
    chosen = pool[pool.selected_in_first_batch]
    axis.scatter(chosen.prediction, chosen.sd_latent, s=43,
                 facecolors='none', edgecolors=C['orange'], lw=1)
    labeled = set()
    for row in chosen.itertuples():
        location = (round(row.prediction, 4), round(row.sd_latent, 4))
        if location in labeled:
            continue
        labeled.add(location)
        same = chosen[(chosen.prediction.round(4) == location[0])
                      & (chosen.sd_latent.round(4) == location[1])]
        offset = {'67-66-3': (6, -10), '127-18-4': (-16, 9),
                  '56-23-5': (5, -10)}.get(row.CAS, (5, 5))
        axis.annotate(', '.join(str(int(rank)) for rank in same.eligible_rank),
                      (row.prediction, row.sd_latent), xytext=offset,
                      textcoords='offset points', fontsize=7.3)
    axis.set(xlim=(-3, 103), xticks=[0, 25, 50, 75, 100], ylim=(0, 37),
             xlabel='Predicted PE conversion (%)', ylabel='SD of predicted mean (pp)')
    axis_fonts(axis)
    color_axis = axat(figure, 333, 40, 7, 89)
    colorbar = figure.colorbar(points, cax=color_axis)
    colorbar.set_label('EI (pp)', fontsize=7.5)
    colorbar.ax.tick_params(labelsize=7.3, length=2)
    txt(figure, 389, 43, 'Orange rings', 7.5, 'bold', color=C['orange'])
    txt(figure, 389, 60, 'Round-1 members', 7.3)
    txt(figure, 389, 82, 'Numbers: initial EI rank', 7.3)
    txt(figure, 389, 108, 'Initial training set: 30', 7.3)
    title(figure, 3, 179, 'b  Leading candidates: prediction, expected gain and selection order')
    txt(figure, 13, 202, 'Rank', 7.3, 'bold')
    txt(figure, 75, 202, 'Candidate', 7.3, 'bold')
    txt(figure, 486, 198, 'Round-1\norder', 7.3, 'bold', ha='center')
    prediction = axat(figure, 277, 221, 132, 204)
    prediction.set_label('b')
    improvement = axat(figure, 430, 221, 37, 204)
    improvement.set_label('b_EI')
    for number, row in enumerate(top.itertuples()):
        yy = 9 - number
        shade = color(row.CAS)
        selected = row.selected_in_first_batch
        prediction.errorbar(row.prediction, yy,
                            xerr=[[row.prediction - row.lower90_pct],
                                  [row.upper90_pct - row.prediction]],
                            fmt='o', mfc=shade if selected else 'white',
                            mec=shade, c=shade, ms=3.8, lw=.8, capsize=1.8)
        improvement.barh(yy, row.EI_pp, height=.52, color=shade if selected else 'white',
                         edgecolor=shade, lw=.6)
        y = 221 + 204 * (9.6 - yy) / 10.2
        txt(figure, 18, y, str(number + 1), 7.3,
            'bold' if selected else 'normal', ha='center', va='center')
        mol(figure, row.CAS, 35, y - 12, 36, 24, atom_font_px=40)
        txt(figure, 77, y, label(row.CAS), 7.3,
            'bold' if selected else 'normal', va='center')
        txt(figure, 486, y, str(int(row.first_batch_slot)) if selected else '–',
            7.3, ha='center', va='center', color=shade)
    prediction.set(xlim=(-3, 103), ylim=(-.6, 9.6), xticks=[0, 50, 100],
                   yticks=[], xlabel='Predicted PE (%)')
    prediction.spines['left'].set_visible(False)
    improvement.set(xlim=(0, 1.5), ylim=(-.6, 9.6), xticks=[0, 1.5], yticks=[], xlabel='EI (pp)')
    improvement.spines['left'].set_visible(False)
    axis_fonts(prediction)
    axis_fonts(improvement)
    txt(figure, 277, 205, '90% prediction intervals', 7.3)
    save(figure, 'Figure_M3')


def m4():
    figure = fig(504, 585)
    solvents = read('data/selected45.csv')
    endpoints = read('figure_data/Figure1b_identity_means.csv').set_index('CAS')
    contributions = read('figure_data/Figure_M4_contributions.csv')
    title(figure, 3, 3, 'Descriptor contributions and conversion predictions')
    txt(figure, 3, 18, 'Four descriptor groups; all 16 combinations; 36 training solvents as background', 7.3)
    order = pd.concat([solvents[solvents.Class == 'Halogenated'],
                       solvents[solvents.Class == 'Amides'],
                       solvents[~solvents.Class.isin(['Halogenated', 'Amides'])]])
    matrix = contributions.pivot(index='CAS', columns='group', values='contribution_pp')
    matrix = matrix.reindex(index=order.CAS, columns=list(GROUPS)).to_numpy()
    maximum = 5 * np.ceil(np.max(np.abs(matrix)) / 5)
    heatmap = axat(figure, 164, 65, 205, 462)
    heatmap.set_label('a')
    image = heatmap.imshow(
        matrix, cmap=LinearSegmentedColormap.from_list('signed', [C['orange'], 'white', C['blue']]),
        vmin=-maximum, vmax=maximum, aspect='auto', interpolation='nearest')
    heatmap.set(yticks=np.arange(45), yticklabels=[label(cas) for cas in order.CAS],
                xticks=np.arange(4), xticklabels=['Element\ncounts', 'Bond\nenvironments',
                                                'Functional\ngroups', 'C–H/C–X\nindicators'])
    heatmap.xaxis.tick_top()
    heatmap.tick_params(axis='both', length=0, pad=4, labelsize=7.3)
    for spine in heatmap.spines.values():
        spine.set_visible(False)
    for y in np.arange(.5, 45):
        heatmap.axhline(y, c='white', lw=.35)
    for x in np.arange(.5, 4):
        heatmap.axvline(x, c='white', lw=.35)
    for boundary in [sum(solvents.Class == 'Halogenated') - .5,
                     sum(solvents.Class.isin(['Halogenated', 'Amides'])) - .5]:
        heatmap.axhline(boundary, c='#555555', lw=.65)
    comparison = axat(figure, 404, 65, 96, 462)
    comparison.set_label('b')
    comparison.set(xlim=(-5, 105), ylim=(44.5, -.5), xticks=[0, 50, 100], yticks=[])
    comparison.xaxis.tick_top()
    comparison.tick_params(axis='x', labelsize=7.3)
    for spine in comparison.spines.values():
        spine.set_visible(False)
    for value in [0, 50, 100]:
        comparison.axvline(value, c='#DDDDDD', lw=.5)
    for number, cas in enumerate(order.CAS):
        row = endpoints.loc[cas]
        comparison.plot([row.observed, row.prediction], [number, number], c='#999999', lw=.55)
        comparison.scatter(row.observed, number, s=12, marker='s', c=color(cas))
        comparison.scatter(row.prediction, number, s=16, facecolors='white', edgecolors=color(cas), lw=.7)
    txt(figure, 452, 32, 'PE conversion (%)', 7.5, ha='center')
    figure.legend(handles=endpoints_legend(), loc='lower right', bbox_to_anchor=(.99, .02),
                  ncol=2, fontsize=7.3, handletextpad=.2, columnspacing=.5)
    color_axis = axat(figure, 177, 547, 180, 7)
    colorbar = figure.colorbar(image, cax=color_axis, orientation='horizontal',
                             ticks=np.linspace(-maximum, maximum, 5))
    colorbar.set_label('Contribution to GP mean (pp)', fontsize=7.5, labelpad=3)
    colorbar.ax.tick_params(labelsize=7.3, length=2, pad=2)
    save(figure, 'Figure_M4')


def main():
    for draw in (m1, m3, m4):
        draw()


if __name__ == '__main__':
    main()
