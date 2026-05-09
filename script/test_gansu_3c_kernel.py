"""
Generate test case for GANSU 3c gradient kernel.
Print the expected values that GANSU should produce.
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

int3c_ip1 = df.incore.aux_e2(mol, auxmol, intor='int3c2e_ip1').reshape(3, nao, nao, naux)
int3c_ip2 = df.incore.aux_e2(mol, auxmol, intor='int3c2e_ip2').reshape(3, nao, nao, naux)

aoslices = mol.aoslice_by_atom()
auxslices = auxmol.aoslice_by_atom()

D = np.eye(nao)  # identity density for testing
d_bar = np.zeros(naux)
d_bar[0] = 1.0  # single aux function

# Coulomb 3c gradient with unit D and d_bar[0]=1:
# grad_A = Σ_{μν} d_bar_P D_μν d(μν|P)/dR_A
# With D=I, d_bar[0]=1: grad_A = Σ_μ d(μ,μ|0)/dR_A

print("=== Expected Coulomb 3c gradient with D=I, d_bar[0]=1 ===")
print("(Only contribution from P=0, diagonal μ=ν)")
for iatm in range(mol.natm):
    p0, p1 = aoslices[iatm, 2], aoslices[iatm, 3]
    q0, q1 = auxslices[iatm, 2], auxslices[iatm, 3]
    gz = 0.0
    # μ on atom: -ip1[z, μ, ν, 0] × D[μ,ν] × d_bar[0]
    # With D=I: only μ=ν contributes
    for mu in range(p0, p1):
        gz -= int3c_ip1[2, mu, mu, 0]
    # ν on atom: -ip1[z, ν, μ, 0] × D[μ,ν]
    for nu in range(p0, p1):
        gz -= int3c_ip1[2, nu, nu, 0]  # swap μ↔ν, but D=I so same as above
    # P on atom: only if P=0 is on this atom
    if q0 <= 0 < q1:
        for mu in range(nao):
            gz -= int3c_ip2[2, mu, mu, 0]
    print(f"  {mol.atom_symbol(iatm):2s}: dz = {gz:.10e}")

# Now with actual D and d_bar
mf = gto.M(atom='O 0 0 0.127; H 0 0.758 -0.509; H 0 -0.758 -0.509',
           basis='sto-3g', unit='Angstrom')
from pyscf import scf
mf_obj = scf.RHF(mf).density_fit(auxbasis=aux_dict)
mf_obj.kernel()
D = mf_obj.make_rdm1()

int2c = auxmol.intor('int2c2e')
int3c_val = df.incore.aux_e2(mol, auxmol, intor='int3c2e').reshape(nao*nao, naux)
L = np.linalg.cholesky(int2c)
B = np.linalg.inv(L) @ int3c_val.T
w = B @ D.flatten()
d_bar = np.linalg.solve(L.T, w)

print(f"\n=== Full Coulomb 3c gradient (PySCF reference) ===")
D3 = np.outer(d_bar, D.flatten()).reshape(naux, nao, nao)  # D3[P,μ,ν] = d_bar[P] D[μ,ν]

grad_3c = np.zeros((mol.natm, 3))
for iatm in range(mol.natm):
    p0, p1 = aoslices[iatm, 2], aoslices[iatm, 3]
    q0, q1 = auxslices[iatm, 2], auxslices[iatm, 3]
    for xyz in range(3):
        # μ on atom
        grad_3c[iatm, xyz] -= np.einsum('Puv,uvP->', D3[:, p0:p1, :], int3c_ip1[xyz, p0:p1, :, :])
        # ν on atom
        grad_3c[iatm, xyz] -= np.einsum('Puv,vuP->', D3[:, :, p0:p1], int3c_ip1[xyz, p0:p1, :, :])
        # P on atom
        grad_3c[iatm, xyz] -= np.einsum('Puv,uvP->', D3[q0:q1, :, :], int3c_ip2[xyz, :, :, q0:q1])

print("  PySCF Coulomb 3c (dz):")
for i in range(mol.natm):
    print(f"    {mol.atom_symbol(i):2s}: {grad_3c[i,2]:.10e}")

# Now what GANSU's kernel convention would give:
# GANSU kernel computes shift formula = -d/dR (not +d/dR)
# For atom A (μ center): kernel adds D3[P,μ,ν] × (-d(μν|P)/dR_A)
# For atom B (ν center): kernel adds D3[P,μ,ν] × (-d(μν|P)/dR_B)
# For atom C (P center): kernel adds D3[P,μ,ν] × (d(μν|P)/dR_A + d(μν|P)/dR_B)  [translational inv]
#
# PySCF: -ip1 = d/dR_μ, -ip2 = d/dR_P
# GANSU shift formula = +2α I(l+1) - l I(l-1) = d/dx = -d/dA_x
# So GANSU d/dA output = -d(μν|P)/dR_A = ip1[μ,ν,P]
#
# GANSU 3c contribution for atom X:
#   grad[X] += Σ D3[P,μ,ν] × kernel_dA (if μ on X)
#            += Σ D3[P,μ,ν] × kernel_dB (if ν on X)
#            += Σ D3[P,μ,ν] × (-(kernel_dA+kernel_dB)) (if P on X)
#
# kernel_dA = ip1[z,μ,ν,P] = -d/dR_μ = -(- ip1) wait...
# PySCF ip1 = -d/dR_1 = -d/dR_μ
# GANSU shift = -d/dR_A = -d/dR_μ = same as PySCF ip1!
#
# So: kernel_dA = PySCF_ip1[z,μ,ν,P]
# GANSU contribution:
#   grad[X] += D3 × ip1[μ] (μ on X)    → PySCF would write: += D3 × ip1 = -D3 × (-ip1) = -D3 × d/dR_μ
#   grad[X] += D3 × kernel_dB (ν on X)  → kernel_dB = ip1[z,ν,μ,P]? or different?
#
# For d/dB (ν center), GANSU uses: 2β I(...,l2+1,...) - l2 I(...,l2-1,...)
# This gives d/dx_B = -d/dR_B.
# PySCF's ν center derivative: -ip1[z,ν,μ,P] (swap and negate) = d/dR_ν
#
# So GANSU dB = -d/dR_B, PySCF dR_ν = -ip1[ν,μ,P]
# GANSU dB should equal PySCF ip1[ν,μ,P]?
# Let me just check numerically.

print(f"\n=== GANSU would compute (for D3=d_bar⊗D): ===")
print(f"  If kernel uses shift formula and atomicAdd:")
print(f"  grad[A] += D3 × shift_dA + (P on A: D3 × (-(shift_dA+shift_dB)))")
print(f"  shift_dA ≈ ip1[μ,ν,P], shift_dB ≈ ip1[ν,μ,P]")

# Simulate GANSU kernel behavior
grad_gansu = np.zeros((mol.natm, 3))
for mu in range(nao):
    for nu in range(nao):
        for P in range(naux):
            d3 = D3[P, mu, nu]  # D3_eff in GANSU (positive d_bar × D)
            if abs(d3) < 1e-15: continue

            # GANSU shift formula: dA = ip1[z,μ,ν,P], dB = ip1[z,ν,μ,P]
            # (shift formula = -d/dR_center, same sign as PySCF ip1)
            dA_z = int3c_ip1[2, mu, nu, P]  # -d/dR_μ
            dB_z = int3c_ip1[2, nu, mu, P]  # -d/dR_ν

            # Find atoms
            atom_mu = [a for a in range(mol.natm) if aoslices[a,2] <= mu < aoslices[a,3]][0]
            atom_nu = [a for a in range(mol.natm) if aoslices[a,2] <= nu < aoslices[a,3]][0]
            atom_P = [a for a in range(mol.natm) if auxslices[a,2] <= P < auxslices[a,3]][0]

            grad_gansu[atom_mu, 2] += d3 * dA_z
            grad_gansu[atom_nu, 2] += d3 * dB_z
            grad_gansu[atom_P, 2] += d3 * (-(dA_z + dB_z))

print(f"\n  Simulated GANSU 3c (dz):")
for i in range(mol.natm):
    print(f"    {mol.atom_symbol(i):2s}: {grad_gansu[i,2]:.10e}")

print(f"\n  PySCF 3c (dz):")
for i in range(mol.natm):
    print(f"    {mol.atom_symbol(i):2s}: {grad_3c[i,2]:.10e}")

print(f"\n  Diff (GANSU - PySCF):")
for i in range(mol.natm):
    print(f"    {mol.atom_symbol(i):2s}: {grad_gansu[i,2]-grad_3c[i,2]:.10e}")
