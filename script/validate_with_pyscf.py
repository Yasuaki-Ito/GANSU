"""
GANSU Validation Script: Generate reference values using PySCF

Usage:
    pip install pyscf
    python validate_with_pyscf.py

This script computes reference energies for various molecules, basis sets,
and methods using PySCF, to validate GANSU's results.

Note: For basis sets with d-functions or higher, cart=True is required
to match GANSU's Cartesian basis functions (6d, 10f).
"""

import numpy as np

try:
    from pyscf import gto, scf, mp, cc, ci, fci
except ImportError:
    print("PySCF is not installed. Install with: pip install pyscf")
    exit(1)


def print_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def run_rhf(mol_args, label, basis='sto-3g', cart=False):
    """Run RHF and return mf object and total energy."""
    mol = gto.M(**mol_args, basis=basis, cart=cart, unit='Angstrom')
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    print(f"  {label:30s}  RHF total = {mf.e_tot:.10f}")
    return mf


def run_uhf(mol_args, label, basis='sto-3g', cart=False):
    """Run UHF and return mf object and total energy."""
    mol = gto.M(**mol_args, basis=basis, cart=cart, unit='Angstrom')
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    print(f"  {label:30s}  UHF total = {mf.e_tot:.10f}")
    print(f"  {'':30s}  <S^2> = {mf.spin_square()[0]:.6f}")
    return mf


def run_rohf(mol_args, label, basis='sto-3g', cart=False):
    """Run ROHF and return mf object and total energy."""
    mol = gto.M(**mol_args, basis=basis, cart=cart, unit='Angstrom')
    mf = scf.ROHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    print(f"  {label:30s}  ROHF total = {mf.e_tot:.10f}")
    print(f"  {'':30s}  <S^2> = {mf.spin_square()[0]:.6f}")
    return mf


# ============================================================
#  Molecule definitions (matching GANSU xyz files exactly)
# ============================================================

H2 = dict(atom='H 0.0 0.0 0.0; H 0.0 0.0 0.7122', charge=0, spin=0)
H2O = dict(atom='O 0.0 0.0 0.127; H 0.0 0.758 -0.509; H 0.0 -0.758 -0.509', charge=0, spin=0)
NH3 = dict(atom='N 0.0 0.0 0.128; H 0.0 0.941 -0.298; H 0.815 -0.470 -0.298; H -0.815 -0.470 -0.298', charge=0, spin=0)
CH4 = dict(atom='C 0.0 0.0 0.0; H 0.625 -0.625 -0.625; H 0.625 0.625 0.625; H -0.625 0.625 -0.625; H -0.625 -0.625 0.625', charge=0, spin=0)
N2 = dict(atom='N 0.0 0.0 0.567; N 0.0 0.0 -0.567', charge=0, spin=0)
HF_mol = dict(atom='F 0.0 0.0 0.096; H 0.0 0.0 -0.860', charge=0, spin=0)
O2 = dict(atom='O 0.0 0.0 0.0; O 0.0 0.0 1.2172', charge=0, spin=2)  # triplet


# ============================================================
#  1. RHF / STO-3G
# ============================================================
print_header("RHF / STO-3G")

mf_h2  = run_rhf(H2, "H2")
mf_h2o = run_rhf(H2O, "H2O")
mf_nh3 = run_rhf(NH3, "NH3")
mf_ch4 = run_rhf(CH4, "CH4")
mf_n2  = run_rhf(N2, "N2")
mf_hf  = run_rhf(HF_mol, "HF")


# ============================================================
#  2. RHF / cc-pVDZ (Cartesian, with d-functions)
# ============================================================
print_header("RHF / cc-pVDZ (cart=True)")

mf_h2o_dz = run_rhf(H2O, "H2O", basis='cc-pvdz', cart=True)


# ============================================================
#  3. UHF / STO-3G  (O2 triplet)
# ============================================================
print_header("UHF / STO-3G")

mf_o2_uhf = run_uhf(O2, "O2 (triplet)")

# UHF stability analysis: initial solution is unstable, re-optimize
mo1 = mf_o2_uhf.stability()[0]
dm1 = mf_o2_uhf.make_rdm1(mo1, mf_o2_uhf.mo_occ)
mf_o2_uhf = scf.UHF(gto.M(**O2, basis='sto-3g', unit='Angstrom'))
mf_o2_uhf.conv_tol = 1e-12
mf_o2_uhf.kernel(dm1)
print(f"  {'O2 (triplet) after stability':30s}  UHF total = {mf_o2_uhf.e_tot:.10f}")


# ============================================================
#  4. ROHF / STO-3G  (O2 triplet)
# ============================================================
print_header("ROHF / STO-3G")

mf_o2_rohf = run_rohf(O2, "O2 (triplet)")


# ============================================================
#  5. Post-HF / H2O / STO-3G
# ============================================================
print_header("Post-HF / H2O / STO-3G")

# MP2
mp2 = mp.MP2(mf_h2o)
mp2.kernel()
print(f"  {'MP2':30s}  corr = {mp2.e_corr:.10f}  total = {mp2.e_tot:.10f}")

# MP3 (via CI module - PySCF doesn't have mp.MP3)
from pyscf.ci import cisd as cisd_mod
mp2_for_mp3 = mp.MP2(mf_h2o)
mp2_for_mp3.kernel()
# Use MP2 object to compute MP3 correction
e_mp3_corr = mp2_for_mp3.e_corr_mp3 if hasattr(mp2_for_mp3, 'e_corr_mp3') else None
if e_mp3_corr is None:
    # Compute MP3 manually via emp2 + emp3 from mp module
    try:
        from pyscf.mp.mp2 import _make_eris as make_eris
        e_mp3_corr = mp2_for_mp3.kernel_mp3()[0] if hasattr(mp2_for_mp3, 'kernel_mp3') else None
    except:
        e_mp3_corr = None
if e_mp3_corr is not None:
    e_mp3_total = mf_h2o.e_tot + e_mp3_corr
    print(f"  {'MP3':30s}  corr = {e_mp3_corr:.10f}  total = {e_mp3_total:.10f}")
else:
    e_mp3_corr = None
    e_mp3_total = None
    print(f"  {'MP3':30s}  NOT AVAILABLE in this PySCF version")

# CCSD
mycc = cc.CCSD(mf_h2o)
mycc.conv_tol = 1e-10
mycc.kernel()
print(f"  {'CCSD':30s}  corr = {mycc.e_corr:.10f}  total = {mycc.e_tot:.10f}")

# CCSD(T)
et = mycc.ccsd_t()
print(f"  {'CCSD(T)':30s}  (T) = {et:.10f}  total = {mycc.e_tot + et:.10f}")

# FCI
cisolver = fci.FCI(mf_h2o)
e_fci, _ = cisolver.kernel()
print(f"  {'FCI':30s}  total = {e_fci:.10f}  corr = {e_fci - mf_h2o.e_tot:.10f}")


# ============================================================
#  6. Post-HF / H2O / cc-pVDZ (cart=True)
# ============================================================
print_header("Post-HF / H2O / cc-pVDZ (cart=True)")

mp2_dz = mp.MP2(mf_h2o_dz)
mp2_dz.kernel()
print(f"  {'MP2':30s}  corr = {mp2_dz.e_corr:.10f}  total = {mp2_dz.e_tot:.10f}")


# ============================================================
#  Summary table (for copy-paste into test_validation.cu)
# ============================================================
print_header("Summary for test_validation.cu")

results = [
    ("H2_RHF_STO3G",         mf_h2.e_tot),
    ("H2O_RHF_STO3G",        mf_h2o.e_tot),
    ("NH3_RHF_STO3G",        mf_nh3.e_tot),
    ("CH4_RHF_STO3G",        mf_ch4.e_tot),
    ("N2_RHF_STO3G",         mf_n2.e_tot),
    ("HF_RHF_STO3G",         mf_hf.e_tot),
    ("H2O_RHF_ccpVDZ",       mf_h2o_dz.e_tot),
    ("O2_UHF_STO3G",         mf_o2_uhf.e_tot),
    ("O2_ROHF_STO3G",        mf_o2_rohf.e_tot),
    ("H2O_MP2_STO3G_corr",   mp2.e_corr),
    ("H2O_MP2_STO3G_total",  mp2.e_tot),
    ("H2O_MP3_STO3G_corr",   e_mp3_corr),
    ("H2O_MP3_STO3G_total",  e_mp3_total),
    ("H2O_CCSD_STO3G_corr",  mycc.e_corr),
    ("H2O_CCSD_STO3G_total", mycc.e_tot),
    ("H2O_CCSDT_STO3G_total", mycc.e_tot + et),
    ("H2O_FCI_STO3G_total",  e_fci),
    ("H2O_FCI_STO3G_corr",   e_fci - mf_h2o.e_tot),
    ("H2O_MP2_ccpVDZ_corr",  mp2_dz.e_corr),
    ("H2O_MP2_ccpVDZ_total", mp2_dz.e_tot),
]

print("\n// PySCF reference values (copy into test_validation.cu)")
for name, val in results:
    if val is not None:
        print(f"constexpr real_t REF_{name} = {val:.10f};")
    else:
        print(f"// REF_{name} = NOT AVAILABLE")

# ============================================================
#  CIS (TDA) excited state reference values
# ============================================================
print_header("CIS (TDA) Excited States")

from pyscf import tdscf

# H2 / STO-3G CIS
td_h2 = tdscf.TDA(mf_h2)
td_h2.nstates = 1
td_h2.kernel()
print(f"\nH2/STO-3G CIS excitation energies: {td_h2.e}")

# H2O / STO-3G CIS
td_h2o = tdscf.TDA(mf_h2o)
td_h2o.nstates = 5
td_h2o.kernel()
print(f"\nH2O/STO-3G CIS excitation energies: {td_h2o.e}")

# H2O / cc-pVDZ CIS
td_h2o_dz = tdscf.TDA(mf_h2o_dz)
td_h2o_dz.nstates = 5
td_h2o_dz.kernel()
print(f"\nH2O/cc-pVDZ CIS excitation energies: {td_h2o_dz.e}")

print("\n// CIS reference values")
for i, e in enumerate(td_h2.e):
    print(f"constexpr real_t REF_H2_CIS_STO3G_state{i+1} = {e:.10f};")
for i, e in enumerate(td_h2o.e):
    print(f"constexpr real_t REF_H2O_CIS_STO3G_state{i+1} = {e:.10f};")
for i, e in enumerate(td_h2o_dz.e):
    print(f"constexpr real_t REF_H2O_CIS_ccpVDZ_state{i+1} = {e:.10f};")
