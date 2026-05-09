"""
Detailed 3c derivative test: check per-center contributions.
"""
import numpy as np
from pyscf import gto, df
from pyscf.gto.basis import parse_nwchem

with open('/tmp/cc-pvdz-rifit.nwchem') as f:
    raw = f.read()
aux_dict = {'O': parse_nwchem.parse(raw, 'O'), 'H': parse_nwchem.parse(raw, 'H')}

mol = gto.M(atom='O 0 0 0.127; H 0 0.758 -0.509; H 0 -0.758 -0.509',
            basis='sto-3g', unit='Angstrom')
auxmol = df.addons.make_auxmol(mol, auxbasis=aux_dict)
nao, naux = mol.nao, auxmol.nao

# Print AO and aux slices
aoslices = mol.aoslice_by_atom()
auxslices = auxmol.aoslice_by_atom()
print("AO slices (atom, start, end):")
for i in range(mol.natm):
    p0, p1 = aoslices[i, 2], aoslices[i, 3]
    print(f"  {mol.atom_symbol(i):2s}: AO {p0}..{p1-1}")
print("Aux slices:")
for i in range(mol.natm):
    q0, q1 = auxslices[i, 2], auxslices[i, 3]
    print(f"  {mol.atom_symbol(i):2s}: aux {q0}..{q1-1}")

# Derivative integrals
int3c_ip1 = df.incore.aux_e2(mol, auxmol, intor='int3c2e_ip1').reshape(3, nao, nao, naux)
int3c_ip2 = df.incore.aux_e2(mol, auxmol, intor='int3c2e_ip2').reshape(3, nao, nao, naux)

# Numerical derivative w.r.t. O_z
def get_int3c(dz):
    m = gto.M(atom=f'O 0 0 {0.127+dz}; H 0 0.758 -0.509; H 0 -0.758 -0.509',
              basis='sto-3g', unit='Angstrom')
    am = df.addons.make_auxmol(m, auxbasis=aux_dict)
    return df.incore.aux_e2(m, am, intor='int3c2e').reshape(nao, nao, naux)

delta = 1e-5
bohr = 1.8897259886
deriv_num = (get_int3c(+delta) - get_int3c(-delta)) / (2 * delta * bohr)

# For O atom: numerical = d(μν|P)/dR_O = contributions from ALL centers on O
# ip1[z,μ,ν,P] = -d(μν|P)/dR_μ  (center of μ)
# ip2[z,μ,ν,P] = -d(μν|P)/dR_P  (center of P)
# For atom A:
#   d(μν|P)/dR_A = -ip1 if μ on A, + (-ip2) if P on A, + (translational for ν on A)
#   = -ip1[μ∈A] - ip2[P∈A] + (ip1+ip2)[ν∈A by trans. inv.]

# Actually per-center:
# d(μν|P)/dR_μ = -ip1[z,μ,ν,P]
# d(μν|P)/dR_ν = -ip1[z,ν,μ,P]  (by symmetry μ↔ν)
# d(μν|P)/dR_P = -ip2[z,μ,ν,P]
# Translational invariance: d/dR_μ + d/dR_ν + d/dR_P = 0

# Test: for specific (μ,ν,P), compute per-center derivatives
print("\n=== Per-center derivative test ===")
print(f"{'(mu,nu,P)':>12s} {'atom_mu':>7s} {'atom_nu':>7s} {'atom_P':>7s} {'num':>12s} {'-ip1':>12s} {'-ip1(nu)':>12s} {'-ip2':>12s} {'sum':>12s}")

# Pick elements with large derivatives
flat = np.abs(deriv_num).flatten()
top_idx = np.argsort(flat)[-10:]
for idx in reversed(top_idx):
    mu, nu, P = np.unravel_index(idx, (nao, nao, naux))

    # Which atom?
    atom_mu = -1
    atom_nu = -1
    atom_P = -1
    for a in range(mol.natm):
        p0, p1 = aoslices[a, 2], aoslices[a, 3]
        if p0 <= mu < p1: atom_mu = a
        if p0 <= nu < p1: atom_nu = a
        q0, q1 = auxslices[a, 2], auxslices[a, 3]
        if q0 <= P < q1: atom_P = a

    num = deriv_num[mu, nu, P]
    d_mu = -int3c_ip1[2, mu, nu, P]    # d/dR_μ
    d_nu = -int3c_ip1[2, nu, mu, P]    # d/dR_ν (swap μ↔ν)
    d_P = -int3c_ip2[2, mu, nu, P]     # d/dR_P

    # Total for O atom (atom 0): sum contributions from all centers on O
    total_O = 0.0
    if atom_mu == 0: total_O += d_mu
    if atom_nu == 0: total_O += d_nu
    if atom_P == 0: total_O += d_P

    print(f"  ({mu},{nu},{P:2d})  O({atom_mu})  O({atom_nu})  O({atom_P})  {num:12.6e}  {d_mu:12.6e}  {d_nu:12.6e}  {d_P:12.6e}  O_total:{total_O:12.6e}  match:{abs(num-total_O)<1e-4}")
