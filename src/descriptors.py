"""Molecular graph descriptors used for solvent prediction."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
MOTIFS = [
    'amide', 'amide_NH2', 'amide_NH1', 'amide_NH0', 'formyl_amide',
    'primary_amine', 'secondary_amine', 'tertiary_amine', 'alcohol',
    'phenol', 'ether', 'carbonyl', 'ester', 'nitrile', 'sulfoxide',
    'sulfone', 'aromatic_nitrogen',
]
NAMES = (
    ['n_C', 'n_N', 'n_O', 'n_S', 'n_P', 'n_Si', 'n_F', 'n_Cl', 'n_Br', 'n_I']
    + [f'{element}_{bond}' for element in ('Cl', 'Br', 'I')
       for bond in ('sp3', 'sp2_nonaromatic', 'aryl')]
    + ['carbon_aromatic', 'carbon_sp2_nonaromatic']
    + MOTIFS + ['H_available', 'X_available']
)
SMARTS = json.loads((ROOT / 'data/structural_SMARTS.json').read_text())
PATTERNS = {name: Chem.MolFromSmarts(query) for name, query in SMARTS.items()}


def molecular_inputs(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    atoms = list(molecule.GetAtoms())
    values = [sum(atom.GetAtomicNum() == number for atom in atoms)
              for number in (6, 7, 8, 16, 15, 14, 9, 17, 35, 53)]
    for halogen in (17, 35, 53):
        counts = [0, 0, 0]
        for atom in atoms:
            if atom.GetAtomicNum() != halogen:
                continue
            for carbon in atom.GetNeighbors():
                if carbon.GetAtomicNum() != 6:
                    continue
                if carbon.GetIsAromatic():
                    counts[2] += 1
                elif carbon.GetHybridization() == Chem.HybridizationType.SP3:
                    counts[0] += 1
                elif carbon.GetHybridization() == Chem.HybridizationType.SP2:
                    counts[1] += 1
        values.extend(counts)
    values.extend([
        sum(atom.GetAtomicNum() == 6 and atom.GetIsAromatic() for atom in atoms),
        sum(atom.GetAtomicNum() == 6 and not atom.GetIsAromatic()
            and atom.GetHybridization() == Chem.HybridizationType.SP2 for atom in atoms),
    ])
    values.extend(len(molecule.GetSubstructMatches(PATTERNS[name])) for name in MOTIFS)
    values.extend([
        any(atom.GetAtomicNum() == 6 and atom.GetTotalNumHs(includeNeighbors=True) > 0
            for atom in atoms),
        any(atom.GetAtomicNum() == 6 and any(neighbor.GetAtomicNum() in (17, 35, 53)
            for neighbor in atom.GetNeighbors()) for atom in atoms),
    ])
    return np.asarray(values, dtype=float)


def descriptor_frame(solvents):
    frame = pd.DataFrame([molecular_inputs(smiles) for smiles in solvents.SMILES], columns=NAMES)
    frame.insert(0, 'CAS', solvents.CAS.to_numpy())
    return frame
