"""
Test a single 3-center integral derivative numerically.
Use PySCF with GANSU's exact aux basis (nwchem format).
Compare d(μν|P)/dR_A with finite difference.
"""
import numpy as np
from pyscf import gto, df
from pyscf.gto.basis import parse_nwchem

with open('/tmp/cc-pvdz-rifit.nwchem') as f:
    raw = f.read()
auxbasis_o = parse_nwchem.parse(raw, 'O')
auxbasis_h = parse_nwchem.parse(raw, 'H')
aux_dict = {'O': auxbasis_o, 'H': auxbasis_h}

def make_mol(dz=0.0):
    return gto.M(
        atom=f'O 0 0 {0.127+dz}; H 0 0.758 -0.509; H 0 -0.758 -0.509',
        basis='sto-3g', unit='Angstrom'
    )

mol = make_mol()
auxmol = df.addons.make_auxmol(mol, auxbasis=aux_dict)
nao = mol.nao
naux = auxmol.nao
print(f"nao={nao}, naux={naux}")

# 3-center integral
int3c = df.incore.aux_e2(mol, auxmol, intor='int3c2e')  # (nao, nao, naux)
print(f"(0,0|0) = {int3c[0,0,0]:.10e}")
print(f"(0,1|0) = {int3c[0,1,0]:.10e}")

# Analytical derivative (ip1 = -d/dR_1)
int3c_ip1 = df.incore.aux_e2(mol, auxmol, intor='int3c2e_ip1')  # (3, nao, nao, naux)
print(f"\nAnalytical: -d(0,0|0)/dR_O_z via ip1 = {int3c_ip1[2,0,0,0]:.10e}")
print(f"  so d(0,0|0)/dR_O_z = {-int3c_ip1[2,0,0,0]:.10e}")

# Numerical derivative: d(μν|P)/dR_O_z by finite difference
delta = 1e-6  # Angstrom
mol_p = make_mol(+delta)
mol_m = make_mol(-delta)
auxmol_p = df.addons.make_auxmol(mol_p, auxbasis=aux_dict)
auxmol_m = df.addons.make_auxmol(mol_m, auxbasis=aux_dict)
int3c_p = df.incore.aux_e2(mol_p, auxmol_p, intor='int3c2e')
int3c_m = df.incore.aux_e2(mol_m, auxmol_m, intor='int3c2e')

# Convert delta to bohr
bohr = 1.8897259886
delta_bohr = delta * bohr

deriv_num = (int3c_p - int3c_m) / (2 * delta_bohr)
print(f"\nNumerical: d(0,0|0)/dR_O_z = {deriv_num[0,0,0]:.10e}")
print(f"Analytical:                  = {-int3c_ip1[2,0,0,0]:.10e}")
print(f"Diff: {deriv_num[0,0,0] - (-int3c_ip1[2,0,0,0]):.2e}")

# Find largest derivatives
print(f"\n=== Largest (μ,ν|P) derivatives w.r.t. O_z ===")
print(f"{'(mu,nu,P)':>12s} {'numerical':>14s} {'analytical':>14s} {'ratio':>10s}")
flat = np.abs(deriv_num).flatten()
top_idx = np.argsort(flat)[-20:]
for idx in reversed(top_idx):
    mu, nu, P = np.unravel_index(idx, (nao, nao, naux))
    num = deriv_num[mu,nu,P]
    ana = -int3c_ip1[2,mu,nu,P]
    ratio = ana/num if abs(num) > 1e-12 else float('nan')
    print(f"  ({mu},{nu},{P}) {num:14.8e} {ana:14.8e} {ratio:10.4f}")

# Also check ip2 (aux center derivative)
int3c_ip2 = df.incore.aux_e2(mol, auxmol, intor='int3c2e_ip2')
print(f"\n=== Aux center derivative d(0,0|P)/dR_O_z (O has aux functions) ===")
# O is atom 0, aux functions on O
auxslices = auxmol.aoslice_by_atom()
q0, q1 = auxslices[0, 2], auxslices[0, 3]
print(f"O aux functions: {q0}..{q1}")
for P in range(q0, min(q0+3, q1)):
    num = deriv_num[0,0,P]
    ana_ip2 = -int3c_ip2[2,0,0,P]
    ana_ip1 = -int3c_ip1[2,0,0,P]  # μ on O
    # By translational invariance: d/dR_O = d/dR_μ + d/dR_P (if both on O)
    # = -ip1 - ip2 ... no: ip1 is for μ center, ip2 is for P center
    # if μ on O and P on O: total = -ip1 - ip2 (both centers on same atom)
    # But numerical deriv captures the total
    print(f"  P={P}: num={num:14.8e}  -ip1(mu)={ana_ip1:14.8e}  -ip2(aux)={ana_ip2:14.8e}  -ip1-ip2={-(int3c_ip1[2,0,0,P]+int3c_ip2[2,0,0,P]):14.8e}")
