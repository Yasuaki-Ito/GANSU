#!/usr/bin/env python3
# GANSU: GPU Accelerated Numerical Simulation Utility
# Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
# SPDX-License-Identifier: BSD-3-Clause
"""Generate the additional validation-benchmark geometries as XYZ.

The auto-fragment vs whole-molecule STEOM comparison uses the SAME geometry on
both sides, so the geometry quality cancels in the deviation; an ETKDG embed
with an MMFF relaxation is therefore adequate here. Writes into ../xyz/ next to
the existing butylbenzene.xyz. Requires RDKit (run wherever it is installed).
"""
import os
from rdkit import Chem
from rdkit.Chem import AllChem

MOLS = {
    "acetophenone":  "CC(=O)c1ccccc1",   # aryl ketone, localized      -> small error
    "cyclohexanone": "O=C1CCCCC1",        # aliphatic n->pi*, localized -> small error
    "anisaldehyde":  "COc1ccc(C=O)cc1",   # push-pull OMe...CHO         -> intermediate
    "styrene":       "C=Cc1ccccc1",       # ring+vinyl conjugation      -> intermediate
}

here = os.path.dirname(os.path.abspath(__file__))
outdir = os.path.join(here, "..", "xyz")
os.makedirs(outdir, exist_ok=True)

for name, smi in MOLS.items():
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xC0FFEE
    AllChem.EmbedMolecule(mol, params)
    AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    conf = mol.GetConformer()
    path = os.path.join(outdir, name + ".xyz")
    with open(path, "w") as f:
        f.write("%d\n%s (RDKit ETKDGv3 + MMFF)\n" % (mol.GetNumAtoms(), name))
        for atom in mol.GetAtoms():
            p = conf.GetAtomPosition(atom.GetIdx())
            f.write("%-2s %12.6f %12.6f %12.6f\n" % (atom.GetSymbol(), p.x, p.y, p.z))
    print("%-14s %2d atoms -> %s" % (name, mol.GetNumAtoms(), path))
