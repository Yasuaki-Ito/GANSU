"""Emit a PySCF-bundled JK-fit auxiliary basis in GANSU's .gbs (Gaussian94)
form. Source data is the one PySCF ships (itself from Basis Set Exchange), so
no network access is needed and the numbers are already validated by PySCF.

Layout is matched to auxiliary_basis/cc-pvdz-rifit.gbs exactly:
    <El>     0
    <L>    <nprim>   1.00
    <exp>             <coeff>
    ...
    ****
Only general (uncontracted or contracted) shells are emitted; SP blocks do not
occur in JK-fit sets.
"""
import sys

from pyscf import gto

ANG = "SPDFGHIK"


def emit(basis_name, elements, out_path):
    lines = [
        "!" + "-" * 70,
        "! Generated from the PySCF-bundled copy of %s" % basis_name.upper(),
        "! (PySCF ships the Basis Set Exchange data; no network fetch needed)",
        "! Emitted for GANSU by scratchpad/mkjkfit.py",
        "!" + "-" * 70,
        "",
        "",
    ]
    n_ok = 0
    for el in elements:
        try:
            b = gto.basis.load(basis_name, el)
        except Exception:                      # element not in this set
            continue
        if not b:
            continue
        lines.append("%-5s 0" % el)
        for shell in b:
            l = shell[0]
            prims = shell[1:]
            # a shell entry is [l, [exp, c1(, c2...)], ...]; a block with more
            # than one contraction column becomes that many separate shells
            ncol = len(prims[0]) - 1
            for col in range(1, ncol + 1):
                keep = [p for p in prims if p[col] != 0.0]
                if not keep:
                    continue
                lines.append("%s   %2d   1.00" % (ANG[l], len(keep)))
                for p in keep:
                    lines.append("  %18.10f    %18.10f" % (p[0], p[col]))
        lines.append("****")
        n_ok += 1
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("%s: %d elements -> %s" % (basis_name, n_ok, out_path))


if __name__ == "__main__":
    els = ("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar "
           "K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr").split()
    emit(sys.argv[1], els, sys.argv[2])
