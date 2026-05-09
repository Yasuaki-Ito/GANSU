"""Check Exchange 3c gradient contribution sign."""
import numpy as np
from pyscf import gto, scf, df
from pyscf.gto.basis import parse_nwchem

with open('/tmp/cc-pvdz-rifit.nwchem') as f:
    raw = f.read()
aux_dict = {'O': parse_nwchem.parse(raw, 'O'), 'H': parse_nwchem.parse(raw, 'H')}

mol = gto.M(atom='O 0 0 0.127; H 0 0.758 -0.509; H 0 -0.758 -0.509',
            basis='sto-3g', unit='Angstrom')
auxmol = df.addons.make_auxmol(mol, auxbasis=aux_dict)
nao, naux = mol.nao, auxmol.nao

mf = scf.RHF(mol).density_fit(auxbasis=aux_dict)
mf.kernel()
D = mf.make_rdm1()

int2c = auxmol.intor('int2c2e')
int3c = df.incore.aux_e2(mol, auxmol, intor='int3c2e').reshape(nao*nao, naux)
L = np.linalg.cholesky(int2c)
B = np.linalg.inv(L) @ int3c.T  # (naux, nao²)
w = B @ D.flatten()

# D3 = wD - 0.5 DBD
B3d = B.reshape(naux, nao, nao)
D3_coulomb = np.outer(w, D.flatten())  # (naux, nao²)
D3_exchange = np.zeros((naux, nao*nao))
for P in range(naux):
    dbd = D @ B3d[P] @ D
    D3_exchange[P] = -0.5 * dbd.flatten()

D3_full = D3_coulomb + D3_exchange

# Z = L^{-T} D3
Z_coulomb = np.linalg.solve(L.T, D3_coulomb)
Z_exchange = np.linalg.solve(L.T, D3_exchange)
Z_full = np.linalg.solve(L.T, D3_full)

# 3c gradient: Σ Z[Q,μν] d(μν|Q)/dR_A
# Kernel returns -d/dR → contribution = Z × (-d/dR)
int3c_ip1 = df.incore.aux_e2(mol, auxmol, intor='int3c2e_ip1').reshape(3, nao, nao, naux)
int3c_ip2 = df.incore.aux_e2(mol, auxmol, intor='int3c2e_ip2').reshape(3, nao, nao, naux)

aoslices = mol.aoslice_by_atom()
auxslices = auxmol.aoslice_by_atom()

def compute_3c_grad(Z_mat):
    """Compute 3c gradient contribution from Z (naux, nao²)."""
    Z3d = Z_mat.reshape(naux, nao, nao)
    grad = np.zeros((mol.natm, 3))
    for iatm in range(mol.natm):
        p0, p1 = aoslices[iatm, 2], aoslices[iatm, 3]
        q0, q1 = auxslices[iatm, 2], auxslices[iatm, 3]
        for xyz in range(3):
            # GANSU convention: kernel returns -d/dR, so contribution = Z × (-d/dR)
            # = Z × ip1 (since ip1 = -d/dR)
            # μ on atom
            grad[iatm, xyz] += np.einsum('Quv,uvQ->', Z3d[:, p0:p1, :], int3c_ip1[xyz, p0:p1, :, :])
            # ν on atom: use ip1[ν,μ,P] = -d(νμ|P)/dR_ν = -d(μν|P)/dR_ν
            grad[iatm, xyz] += np.einsum('Quv,vuQ->', Z3d[:, :, p0:p1], int3c_ip1[xyz, p0:p1, :, :])
            # P on atom: translational invariance → -(dA+dB)
            grad[iatm, xyz] -= np.einsum('Quv,uvQ->', Z3d[q0:q1, :, :], int3c_ip1[xyz, :, :, q0:q1])
            grad[iatm, xyz] -= np.einsum('Quv,vuQ->', Z3d[q0:q1, :, :], int3c_ip1[xyz, :, :, q0:q1])
            # Hmm, translational invariance: d/dC = -(d/dA + d/dB)
            # GANSU kernel: dC = -(dA + dB), and dC is also -d/dR_C
            # So contribution for P on atom = Z × dC = Z × (-(dA+dB))
            # = -Z × dA - Z × dB  where dA,dB are shift formula results = -d/dR
            # Wait, I need to think about this differently.
            # Let's just use ip2 directly for P center.

    # Redo with correct ip2
    grad2 = np.zeros((mol.natm, 3))
    for iatm in range(mol.natm):
        p0, p1 = aoslices[iatm, 2], aoslices[iatm, 3]
        q0, q1 = auxslices[iatm, 2], auxslices[iatm, 3]
        for xyz in range(3):
            # μ on atom: d(μν|Q)/dR_A = -ip1[μ,ν,Q]
            # GANSU kernel gives -d/dR = ip1. Contribution = Z × ip1 = Z × (-d/dR)
            # But we want total = Z × d/dR → need to negate?
            # Actually:
            #   dE/dR = Σ dE/dB × dB/dR = Σ D3 × L^{-1} dV/dR = Σ Z × dV/dR
            #   dV/dR_A for μ on A: = -ip1
            #   So contribution = Z × (-ip1) = -Z × ip1
            grad2[iatm, xyz] -= np.einsum('Quv,uvQ->', Z3d[:, p0:p1, :], int3c_ip1[xyz, p0:p1, :, :])
            grad2[iatm, xyz] -= np.einsum('Quv,vuQ->', Z3d[:, :, p0:p1], int3c_ip1[xyz, p0:p1, :, :])
            grad2[iatm, xyz] -= np.einsum('Quv,uvQ->', Z3d[q0:q1, :, :], int3c_ip2[xyz, :, :, q0:q1])
    return grad2

g_coulomb_3c = compute_3c_grad(Z_coulomb)
g_exchange_3c = compute_3c_grad(Z_exchange)
g_full_3c = compute_3c_grad(Z_full)

print("=== 3c gradient (O dz) ===")
print(f"  Coulomb:  {g_coulomb_3c[0,2]:.6e}")
print(f"  Exchange: {g_exchange_3c[0,2]:.6e}")
print(f"  Full:     {g_full_3c[0,2]:.6e}")

# Reference
g_ref = mf.nuc_grad_method().kernel()
print(f"\n  Full RI-HF total (O dz): {g_ref[0,2]:.6e}")
