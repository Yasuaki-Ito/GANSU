"""
Compare GANSU ADC(2) results with PySCF EE-ADC(2).
Usage: python compare_adc2_pyscf.py
"""

from pyscf import gto, scf, adc

def run_adc2(name, atom_str, basis='cc-pvdz', nroots=5):
    print(f"\n{'='*60}")
    print(f" {name} / {basis} — EE-ADC(2)")
    print(f"{'='*60}")

    mol = gto.M(atom=atom_str, basis=basis, verbose=4)
    mf = scf.RHF(mol).run()
    print(f"\nRHF energy: {mf.e_tot:.10f} Ha")

    myadc = adc.ADC(mf)
    myadc.method_type = 'ee'
    myadc.verbose = 4
    result = myadc.kernel(nroots=nroots)
    e_ee = result[0]

    print(f"\n--- {name} EE-ADC(2) Excitation Energies ---")
    print(f"{'State':>5}  {'E (Ha)':>14}  {'E (eV)':>12}")
    for i, e in enumerate(e_ee):
        print(f"{i+1:5d}  {e:14.8f}  {e*27.211386:12.6f}")

    return mf.e_tot, e_ee


# Benzene
benzene = """
C   0.000    1.387    0.000
C   1.201    0.693    0.000
C   1.201   -0.693    0.000
C   0.000   -1.387    0.000
C  -1.201   -0.693    0.000
C  -1.201    0.693    0.000
H   0.000    2.469    0.000
H   2.139    1.235    0.000
H   2.139   -1.235    0.000
H   0.000   -2.469    0.000
H  -2.139   -1.235    0.000
H  -2.139    1.235    0.000
"""

# Naphthalene
naphthalene = """
C  0.0  1.253   1.395
C  0.0  2.421   0.713
C  0.0  2.421  -0.713
C  0.0  1.253  -1.395
C  0.0 -1.253  -1.395
C  0.0 -2.421  -0.713
C  0.0 -2.421   0.713
C  0.0 -1.253   1.395
C  0.0  0.000   0.702
C  0.0  0.000  -0.702
H  0.0  1.244   2.478
H  0.0  3.367   1.240
H  0.0  3.367  -1.240
H  0.0  1.244  -2.478
H  0.0 -1.244  -2.478
H  0.0 -3.367  -1.240
H  0.0 -3.367   1.240
H  0.0 -1.244   2.478
"""

if __name__ == '__main__':
    e_rhf_b, e_adc2_b = run_adc2("Benzene", benzene)
    e_rhf_n, e_adc2_n = run_adc2("Naphthalene", naphthalene)

    print(f"\n{'='*60}")
    print(f" Summary")
    print(f"{'='*60}")
    print(f"\nBenzene RHF: {e_rhf_b:.10f}")
    print(f"Benzene ADC(2) excitations (Ha): {[f'{e:.8f}' for e in e_adc2_b]}")
    print(f"\nNaphthalene RHF: {e_rhf_n:.10f}")
    print(f"Naphthalene ADC(2) excitations (Ha): {[f'{e:.8f}' for e in e_adc2_n]}")
