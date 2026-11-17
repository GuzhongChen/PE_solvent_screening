"""Combine the workflow panel and the numerical panels of Figure 1."""

from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET

import pymupdf as fitz
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
NS = 'http://www.w3.org/2000/svg'
ET.register_namespace('', NS)
ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')


def assemble(bottom_svg, bottom_pdf, output):
    canvas = ET.Element(f'{{{NS}}}svg', {
        'width': '504pt', 'height': '712pt', 'viewBox': '0 0 504 712',
    })
    ET.SubElement(canvas, f'{{{NS}}}rect', {'width': '504', 'height': '712', 'fill': 'white'})
    panel = ET.parse(ROOT / 'assets/Figure1a.svg').getroot()
    panel.attrib.update(x='0', y='0', width='504', height='184')
    canvas.append(panel)
    bottom = ET.SubElement(canvas, f'{{{NS}}}svg', {
        'x': '0', 'y': '184', 'width': '504', 'height': '528',
        'viewBox': '0 133 504 528', 'overflow': 'hidden',
    })
    for element in ET.parse(bottom_svg).getroot():
        bottom.append(deepcopy(element))
    ET.ElementTree(canvas).write(output / 'Figure1.svg', encoding='utf-8', xml_declaration=True)

    with fitz.open(ROOT / 'assets/Figure1a.pdf') as workflow, fitz.open(bottom_pdf) as plots:
        document = fitz.open()
        page = document.new_page(width=504, height=712)
        page.show_pdf_page(fitz.Rect(0, 0, 504, 184), workflow, 0)
        page.show_pdf_page(fitz.Rect(0, 184, 504, 712), plots, 0,
                           clip=fitz.Rect(0, 133, 504, 661))
        document.save(output / 'Figure1.pdf', garbage=4, deflate=True, no_new_id=True)
        pixels = page.get_pixmap(dpi=600, alpha=False)
        image = Image.frombytes('RGB', (pixels.width, pixels.height), pixels.samples)
        image.save(output / 'Figure1.png', dpi=(600, 600))
        image.save(output / 'Figure1.tiff', dpi=(600, 600), compression='tiff_lzw')
        document.close()
