"""Shared layout, colors and molecular drawings for the manuscript figures."""

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
import pandas as pd
from PIL import Image
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from svgpath2mpl import parse_path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures'
PALETTE = json.loads((ROOT / 'data/palette.json').read_text(encoding='utf-8'))
C = PALETTE['swatches']
COL = {cas: C[name] for cas, name in PALETTE['solvent_colors'].items()}
LIB = pd.read_csv(ROOT / 'data/library259.csv').set_index('CAS')
SHORT = {
    '75-34-3': '1,1-DCA',
    '107-06-2': '1,2-DCA',
    '67-66-3': 'CHCl₃',
    '56-23-5': 'CCl₄',
    '71-55-6': '1,1,1-TCA',
    '79-00-5': '1,1,2-TCA',
    '79-34-5': 'TeCA',
    '76-01-7': 'PentaCA',
    '127-18-4': 'PCE',
    '79-01-6': 'TCE',
    '75-09-2': 'CH₂Cl₂',
    '75-11-6': 'CH₂I₂',
    '75-12-7': 'Formamide',
    '123-39-7': 'NMF',
    '68-12-2': 'DMF',
}
NAME = {
    '68-12-2': 'DMF',
    '123-39-7': 'N-Methylformamide',
    '75-12-7': 'Formamide',
    '75-09-2': 'CH₂Cl₂',
    '67-66-3': 'CHCl₃',
    '56-23-5': 'CCl₄',
    '141-62-8': 'Decamethyltetrasiloxane',
    '680-31-9': 'HMPA',
    '5614-37-9': 'Cyclopentyl methyl ether',
    '872-50-4': 'NMP',
    '67-68-5': 'DMSO',
    '108-29-2': 'γ-Valerolactone',
    '107-46-0': 'Hexamethyldisiloxane',
    '108-10-1': 'Methyl isobutyl ketone',
    '75-85-4': 'tert-Amyl alcohol',
    '107-83-5': '2-Methylpentane',
    '102-76-1': 'Triacetin',
    '111-69-3': 'Adiponitrile',
    '107-15-3': 'Ethylenediamine',
    '120-51-4': 'Benzyl benzoate',
}
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans', 'Arial'],
    'font.size': 7.0, 'axes.labelsize': 7.0, 'xtick.labelsize': 6.7,
    'ytick.labelsize': 6.7, 'axes.linewidth': .6,
    'axes.spines.top': False, 'axes.spines.right': False,
    'svg.fonttype': 'none', 'pdf.fonttype': 42,
    'text.color': '#222222', 'axes.edgecolor': '#555555',
    'xtick.color': '#555555', 'ytick.color': '#555555',
    'legend.frameon': False, 'savefig.facecolor': 'white',
    'mathtext.default': 'regular',
})


def read(name):
    return pd.read_csv(ROOT / name)


def color(cas):
    return COL.get(cas, C['teal'])


def label(cas):
    return NAME.get(cas, LIB.loc[cas, 'Name'])


def fig(width, height):
    return plt.figure(figsize=(width / 72, height / 72))


def txt(figure, x, y, text, size=7, weight='normal', ha='left', va='top', **kwargs):
    width, height = figure.get_size_inches() * 72
    return figure.text(x / width, 1 - y / height, text, fontsize=size,
                       weight=weight, ha=ha, va=va, **kwargs)


def axat(figure, x, y, width, height):
    fw, fh = figure.get_size_inches() * 72
    axis = figure.add_axes([x / fw, 1 - (y + height) / fh, width / fw, height / fh])
    axis.tick_params(length=2.5, pad=2)
    axis.patch.set_facecolor('none')
    return axis


def title(figure, x, y, text):
    return txt(figure, x, y, text, 8.4, 'bold')


def mol(figure, cas, x, y, width=36, height=25, atom_font_px=30):
    axis = axat(figure, x, y, width, height)
    axis.set_axis_off()
    axis.set(xlim=(0, 180), ylim=(100, 0))
    axis.set_aspect('equal')
    molecule = Chem.MolFromSmiles(LIB.loc[cas, 'SMILES'])
    drawer = rdMolDraw2D.MolDraw2DSVG(180, 100)
    options = drawer.drawOptions()
    options.useBWAtomPalette()
    options.clearBackground = False
    options.bondLineWidth = 2.1
    options.padding = .07
    options.minFontSize = atom_font_px
    options.maxFontSize = atom_font_px
    drawer.DrawMolecule(molecule)
    drawer.FinishDrawing()
    for element in ET.fromstring(drawer.GetDrawingText()).iter():
        if element.tag.endswith('path') and element.get('d'):
            style = dict(item.split(':', 1) for item in element.get('style', '').split(';') if ':' in item)
            line_width = float(style.get('stroke-width', '0').replace('px', ''))
            axis.add_patch(PathPatch(parse_path(element.get('d')),
                                     facecolor=style.get('fill', '#000000'),
                                     edgecolor=style.get('stroke', 'none'),
                                     lw=line_width * width / 180,
                                     capstyle='round', joinstyle='round'))
    return axis


def endpoints_legend():
    return [
        Line2D([], [], ls='none', marker='s', ms=3.6, color=C['blue'], label='Measured'),
        Line2D([], [], ls='none', marker='o', ms=3.6, mfc='white', mec=C['blue'], label='Predicted'),
    ]


def save(figure, name):
    OUT.mkdir(exist_ok=True)
    figure.savefig(OUT / f'{name}.pdf', metadata={'CreationDate': None, 'ModDate': None})
    figure.savefig(OUT / f'{name}.svg', metadata={'Date': None})
    figure.savefig(OUT / f'{name}.png', dpi=600)
    image = Image.open(OUT / f'{name}.png').convert('RGB')
    image.save(OUT / f'{name}.tiff', dpi=(600, 600), compression='tiff_lzw')
    plt.close(figure)
