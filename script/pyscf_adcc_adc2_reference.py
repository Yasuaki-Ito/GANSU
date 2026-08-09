#!/usr/bin/env python
"""M8 external ADC(2) reference via PySCF's native EE-ADC(2).

Uses pyscf.adc with method_type="ee" -- no compiled extension is needed (unlike
adcc, which fails to build without Python dev headers). pyscf's ADC(2) is the
canonical (non-CVS) ADC(2); with a converged RHF reference this is an
implementation-independent check of the GANSU numbers.

Matches the GANSU whole-molecule ADC(2) run: cc-pVDZ, frozen core, 5 lowest
excitations. Geometries are read from ../xyz/<Molecule>.xyz so the reference and
GANSU use identical coordinates.

    python pyscf_adcc_adc2_reference.py        # pyscf already installed
"""
import os
from pyscf import gto, scf, adc

HERE = os.path.dirname(os.path.abspath(__file__))
XYZDIR = os.path.join(HERE, "..", "xyz")
HARTREE2EV = 27.211386245988

for name in ["Formaldehyde", "C2H4", "H2O"]:
    xyz = os.path.join(XYZDIR, name + ".xyz")
    mol = gto.M(atom=xyz, basis="cc-pvdz", unit="Angstrom", verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.kernel()

    # frozen core = number of 1s-like core orbitals (C,N,O -> 1 each; H -> 0)
    ncore = int(sum(1 for z in mol.atom_charges() if z > 2))

    myadc = adc.ADC(mf, frozen=ncore)
    myadc.method = "adc(2)"
    myadc.method_type = "ee"
    myadc.conv_tol = 1e-8
    e = myadc.kernel(nroots=5)[0]  # excitation energies, Hartree

    print("==== %s  EE-ADC(2)/cc-pVDZ  (frozen_core=%d) ====" % (name, ncore))
    for i, ev in enumerate(e):
        print("  %d  %14.8f  %8.4f" % (i, ev, ev * HARTREE2EV))
    print()
