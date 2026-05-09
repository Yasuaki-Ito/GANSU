"""
Verify ADC(2) reference values for all test molecules.
"""
from pyscf import gto, scf, adc

def run(name, atom_str, basis, nroots=3):
    mol = gto.M(atom=atom_str, basis=basis, verbose=0)
    mf = scf.RHF(mol).run()
    myadc = adc.ADC(mf)
    myadc.method_type = 'ee'
    myadc.verbose = 0
    result = myadc.kernel(nroots=nroots)
    e_ee = result[0]
    print(f"\n{name} / {basis}")
    print(f"  RHF energy: {mf.e_tot:.10f}")
    for i, e in enumerate(e_ee):
        print(f"  State {i+1}: {e:.10f} Ha = {e*27.211386:.6f} eV")
    return e_ee

# H2/STO-3G
run("H2", "H 0 0 0; H 0 0 0.74", "sto-3g", nroots=1)

# H2O/STO-3G
run("H2O", "O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587", "sto-3g", nroots=3)

# HF/cc-pVDZ
run("HF", "H 0 0 0; F 0 0 0.917", "cc-pvdz", nroots=3)

# H2O/cc-pVDZ (from user output)
run("H2O", "O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587", "cc-pvdz", nroots=3)
