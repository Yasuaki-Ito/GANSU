"""
Read PySCF's radc_ee.py to extract the EE-ADC(2) sigma vector formulas,
then build the full ADC(2) matrix numerically by probing the matvec.

Usage: python read_pyscf_radc_ee.py
"""
import numpy as np
from pyscf import gto, scf, adc, ao2mo
import os, inspect

# 1. Read the radc_ee.py source to find the sigma vector formulas
import pyscf.adc.radc_ee as radc_ee
src_file = radc_ee.__file__
print(f"radc_ee location: {src_file}")

with open(src_file) as f:
    lines = f.readlines()

print(f"Total lines: {len(lines)}")

# Print lines with einsum that relate to the sigma vector
print("\n=== Key einsum lines in radc_ee.py ===")
for i, line in enumerate(lines):
    s = line.strip()
    if 'einsum' in s and ('s1' in s or 's2' in s or 'sigma' in s.lower() or 'r1' in s or 'r2' in s):
        # Print context
        print(f"  {i+1:4d}: {lines[i].rstrip()}")

# Print function definitions
print("\n=== Function definitions ===")
for i, line in enumerate(lines):
    if line.strip().startswith('def '):
        print(f"  {i+1:4d}: {line.rstrip()}")

# Print the first 500 lines to see the structure
print("\n=== First 500 lines of radc_ee.py ===")
for i, line in enumerate(lines[:500]):
    print(f"  {i+1:4d}: {line.rstrip()}")
