"""Read PySCF radc_ee.py sigma_ function (lines 800-1100)."""
import pyscf.adc.radc_ee as radc_ee

with open(radc_ee.__file__) as f:
    lines = f.readlines()

print(f"Total lines: {len(lines)}")
print("\n=== Lines 800-1100 of radc_ee.py ===")
for i in range(799, min(1100, len(lines))):
    print(f"  {i+1:4d}: {lines[i].rstrip()}")
