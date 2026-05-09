"""
Understand PySCF's M11 (get_imds) to fix GANSU's M11 formula.
"""
import numpy as np
import inspect
from pyscf import gto, scf, adc, ao2mo
from pyscf.adc import radc_ee

np.set_printoptions(precision=10, linewidth=200, suppress=True)

# Print get_imds source
src = inspect.getsource(radc_ee.get_imds)
print("=== get_imds SOURCE ===")
print(src)
print("=== END ===")
