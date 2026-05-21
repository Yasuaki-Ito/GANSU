#!/usr/bin/env python3
"""
PySCF reference: state-averaged CIS NTO occupations + active-space size.

Compared against GANSU `--post_hf_method cis_nto` output. The two should agree
to ~1e-6 in NTO occupations (the only meaningful difference is CIS Davidson
convergence; the post-CIS linear algebra is well-conditioned).

Usage (run inside WSL Ubuntu — see [[pyscf-run-locally]]):
    wsl python3 script/pyscf_cis_nto_reference.py xyz/H2O.xyz cc-pVDZ 8
    wsl python3 script/pyscf_cis_nto_reference.py xyz/N2.xyz  cc-pVDZ 8
    wsl python3 script/pyscf_cis_nto_reference.py xyz/large_molecular/Benzene.xyz cc-pVDZ 10

Notes:
  * Uses `cart=True` (GANSU is Cartesian GTO; required for d+ basis). See
    [[pyscf-cartesian]].
  * Outputs descending-sorted occupations + counts above the 1e-3 ORCA default.
"""

import sys
import numpy as np


def read_xyz(path):
    with open(path) as f:
        natoms = int(f.readline().strip())
        f.readline()
        atoms = []
        for _ in range(natoms):
            parts = f.readline().split()
            atoms.append(f"{parts[0]} {parts[1]} {parts[2]} {parts[3]}")
    return "; ".join(atoms)


def main(xyz_path: str, basis: str, n_states_cis: int,
         o_thresh: float = 1e-3, v_thresh: float = 1e-3) -> None:
    from pyscf import gto, scf, tdscf

    mol = gto.M(atom=read_xyz(xyz_path), basis=basis, cart=True, unit="Angstrom")
    mf  = scf.RHF(mol)
    mf.kernel()

    td = tdscf.TDA(mf)
    td.nstates = n_states_cis
    td.kernel()

    nocc = mol.nelectron // 2
    nvir = mol.nao - nocc

    # PySCF TDA stores (X, Y) per root with Y=0 for TDA. X has shape (nocc, nvir).
    X = np.stack([xy[0] for xy in td.xy], axis=0).astype(np.float64)
    # L2-normalize each root (PySCF normalizes, but be defensive).
    norms = np.linalg.norm(X.reshape(X.shape[0], -1), axis=1, keepdims=True)
    X = X / norms[..., None]

    weights = np.full(n_states_cis, 1.0 / n_states_cis, dtype=np.float64)
    rho_o = np.einsum("n,nia,nja->ij", weights, X, X)
    rho_v = np.einsum("n,nia,nib->ab", weights, X, X)

    occ_vals = np.linalg.eigvalsh(0.5 * (rho_o + rho_o.T))[::-1]
    vir_vals = np.linalg.eigvalsh(0.5 * (rho_v + rho_v.T))[::-1]

    n_act_o = int(np.sum(occ_vals > o_thresh))
    n_act_v = int(np.sum(vir_vals > v_thresh))

    print(f"--- PySCF CIS NTO reference -----------------------------------------")
    print(f"  xyz             = {xyz_path}")
    print(f"  basis           = {basis} (cart=True)")
    print(f"  n_states_cis    = {n_states_cis}")
    print(f"  nocc            = {nocc}")
    print(f"  nvir            = {nvir}")
    print(f"  trace(rho_o)    = {rho_o.trace():.10f}")
    print(f"  trace(rho_v)    = {rho_v.trace():.10f}")
    print(f"  o_thresh        = {o_thresh}")
    print(f"  v_thresh        = {v_thresh}")
    print(f"  n_act_occ       = {n_act_o} / {nocc}")
    print(f"  n_act_vir       = {n_act_v} / {nvir}")
    print(f"  excitation E    (eV) = {' '.join(f'{e*27.2114:.4f}' for e in td.e[:n_states_cis])}")
    print("  Occupied NTO occupations (desc):")
    for k, v in enumerate(occ_vals):
        tag = " [ACTIVE]" if v > o_thresh else ""
        print(f"    [{k:3d}] n = {v:.10f}{tag}")
    print("  Virtual NTO occupations (desc):")
    for k, v in enumerate(vir_vals):
        tag = " [ACTIVE]" if v > v_thresh else ""
        print(f"    [{k:3d}] n = {v:.10f}{tag}")


if __name__ == "__main__":
    if len(sys.argv) not in (4, 5, 6):
        print(__doc__)
        sys.exit(1)
    xyz   = sys.argv[1]
    basis = sys.argv[2]
    n_cis = int(sys.argv[3])
    o_th  = float(sys.argv[4]) if len(sys.argv) >= 5 else 1e-3
    v_th  = float(sys.argv[5]) if len(sys.argv) >= 6 else 1e-3
    main(xyz, basis, n_cis, o_th, v_th)
