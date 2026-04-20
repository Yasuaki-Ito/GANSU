"""
GANSU CLI entry point.

After pip install: `gansu -x H2O.xyz -b cc-pvdz -m RHF --post_hf ccsd`
"""

import argparse
import sys
import gansu


def main():
    parser = argparse.ArgumentParser(
        prog="gansu",
        description="GANSU: GPU Accelerated Numerical Simulation Utility")
    parser.add_argument("-x", "--xyz", required=True, help="XYZ file path")
    parser.add_argument("-b", "--basis", default="sto-3g", help="Basis set name or path")
    parser.add_argument("-m", "--method", default="RHF", help="HF method (RHF, UHF, ROHF)")
    parser.add_argument("--post_hf", default="none", help="Post-HF method")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress output")
    parser.add_argument("--list-basis", action="store_true", help="List available basis sets")
    # Pass-through for any GANSU parameter
    parser.add_argument("-p", "--param", nargs=2, action="append", metavar=("KEY", "VALUE"),
                        help="Set arbitrary parameter (e.g. -p eri_method RI)")

    args = parser.parse_args()

    if args.list_basis:
        for b in gansu.list_basis_sets():
            print(b)
        return

    gansu.init(force_cpu=args.cpu)

    mol = gansu.Molecule(args.xyz, basis=args.basis)

    extra = {}
    if args.param:
        for k, v in args.param:
            extra[k] = v

    r = mol.run(method=args.method, post_hf=args.post_hf,
                quiet=args.quiet, **extra)

    if not args.quiet:
        print(f"\n=== GANSU Results ===")
        print(f"HF Total Energy:     {r.total_energy:.10f} Hartree")
        if r.post_hf_energy != 0:
            print(f"Post-HF Correlation: {r.post_hf_energy:.10f} Hartree")
            print(f"Total (HF+corr):     {r.total_energy + r.post_hf_energy:.10f} Hartree")

    gansu.finalize()


if __name__ == "__main__":
    main()
