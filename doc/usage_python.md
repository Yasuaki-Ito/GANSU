# GANSU Python API Usage Guide

## Setup

### Build the shared library

```bash
cd GANSU/build
cmake ..
make gansu_shared
```

This produces `libgansu.so` in the build directory.

### Set environment variable

```bash
export GANSU_LIB=/path/to/build/libgansu.so
```

Or add to your `.bashrc` / `.zshrc` for persistent use.

### Import

```python
import sys
sys.path.insert(0, "/path/to/GANSU/python")
import gansu
```

---

## Quick Start

```python
import gansu

gansu.init()

# Create molecule and run RHF + MP2
mol = gansu.Molecule("h2o.xyz", basis="cc-pvdz")
result = mol.run(method="RHF", post_hf="mp2")

print(f"HF energy:  {result.total_energy:.8f} Hartree")
print(f"MP2 corr:   {result.post_hf_energy:.8f} Hartree")
print(f"Total:      {result.total_energy + result.post_hf_energy:.8f} Hartree")

gansu.finalize()
```

---

## Preparing XYZ Files

XYZ files can be created inline in Python:

```python
# H2O
with open("h2o.xyz", "w") as f:
    f.write("""3
Water
O   0.000   0.000   0.127
H   0.000   0.758  -0.509
H   0.000  -0.758  -0.509
""")

# H2 at a given bond length
def write_h2(path, R):
    with open(path, "w") as f:
        f.write(f"2\nH2 R={R}\nH 0 0 0\nH 0 0 {R}\n")

# Benzene
with open("benzene.xyz", "w") as f:
    f.write("""12
Benzene
C   0.000   1.387   0.000
C   1.201   0.693   0.000
C   1.201  -0.693   0.000
C   0.000  -1.387   0.000
C  -1.201  -0.693   0.000
C  -1.201   0.693   0.000
H   0.000   2.469   0.000
H   2.138   1.235   0.000
H   2.138  -1.235   0.000
H   0.000  -2.469   0.000
H  -2.138  -1.235   0.000
H  -2.138   1.235   0.000
""")
```

---

## API Reference

### Module Functions

```python
gansu.init(force_cpu=False)     # Initialize (call once)
gansu.finalize()                # Cleanup (call at end)
gansu.list_basis_sets()         # List available basis sets
```

### `gansu.Molecule`

```python
mol = gansu.Molecule(
    xyz_path,                   # Path to XYZ file
    basis="sto-3g",             # Basis set name or .gbs path
    **kwargs                    # Additional parameters
)
```

### `mol.run()`

```python
result = mol.run(
    method="RHF",               # HF method: "RHF", "UHF", "ROHF"
    post_hf="none",             # Post-HF: "mp2", "ccsd", "fci", etc.
    quiet=True,                 # Suppress GANSU output (default True)
    **kwargs                    # Additional GANSU parameters
)
```

### `gansu.Result` Properties

| Property | Type | Description |
|----------|------|-------------|
| `total_energy` | `float` | HF total energy (Hartree) |
| `post_hf_energy` | `float` | Post-HF correlation energy |
| `correlation_energy` | `float` | Alias for post_hf_energy |
| `nuclear_repulsion_energy` | `float` | Nuclear repulsion energy |
| `num_basis` | `int` | Number of basis functions |
| `num_electrons` | `int` | Number of electrons |
| `num_atoms` | `int` | Number of atoms |
| `orbital_energies` | `np.ndarray` | Orbital energies (nao,) |
| `mo_coefficients` | `np.ndarray` | MO coefficients (nao, nao) |
| `ccsd_1rdm_mo` | `np.ndarray` | CCSD 1-RDM in MO basis (nao, nao) |
| `excited_state_report` | `str` | Formatted excited state table |

---

## Examples

### 1. HF Energy with Different Basis Sets

```python
import gansu

gansu.init()

with open("h2o.xyz", "w") as f:
    f.write("3\nWater\nO 0 0 0.127\nH 0 0.758 -0.509\nH 0 -0.758 -0.509\n")

for basis in ["sto-3g", "cc-pvdz", "cc-pvtz"]:
    r = gansu.Molecule("h2o.xyz", basis=basis).run()
    print(f"{basis:10s}  E = {r.total_energy:.8f}  nao = {r.num_basis}")

gansu.finalize()
```

### 2. Post-HF Method Comparison

```python
import gansu

gansu.init()

with open("h2o.xyz", "w") as f:
    f.write("3\nWater\nO 0 0 0.127\nH 0 0.758 -0.509\nH 0 -0.758 -0.509\n")

for method in ["none", "mp2", "mp3", "ccsd", "ccsd_t", "fci"]:
    mol = gansu.Molecule("h2o.xyz", basis="sto-3g")
    r = mol.run(post_hf=method)
    E = r.total_energy + r.post_hf_energy
    label = method.upper() if method != "none" else "HF"
    print(f"{label:10s}  E = {E:.8f}")

gansu.finalize()
```

### 3. H2 Dissociation Curve

```python
import gansu
import numpy as np

gansu.init()

print(f"{'R (A)':>8s}  {'RHF':>14s}  {'CCSD':>14s}  {'FCI':>14s}")
print("-" * 56)

for R in np.linspace(0.4, 5.0, 24):
    with open("h2.xyz", "w") as f:
        f.write(f"2\nH2\nH 0 0 0\nH 0 0 {R}\n")

    r_hf   = gansu.Molecule("h2.xyz", basis="cc-pvdz").run()
    r_ccsd = gansu.Molecule("h2.xyz", basis="cc-pvdz").run(post_hf="ccsd")
    r_fci  = gansu.Molecule("h2.xyz", basis="cc-pvdz").run(post_hf="fci")

    e_hf   = r_hf.total_energy
    e_ccsd = r_ccsd.total_energy + r_ccsd.post_hf_energy
    e_fci  = r_fci.total_energy + r_fci.post_hf_energy

    print(f"{R:8.3f}  {e_hf:14.8f}  {e_ccsd:14.8f}  {e_fci:14.8f}")

gansu.finalize()
```

### 4. Excited States

```python
import gansu

gansu.init()

with open("h2o.xyz", "w") as f:
    f.write("3\nWater\nO 0 0 0.127\nH 0 0.758 -0.509\nH 0 -0.758 -0.509\n")

mol = gansu.Molecule("h2o.xyz", basis="cc-pvdz")
r = mol.run(post_hf="eom_ccsd", n_excited_states="5")
print(r.excited_state_report)

gansu.finalize()
```

### 5. Orbital Energies and MO Coefficients

```python
import gansu
import numpy as np

gansu.init()

with open("h2o.xyz", "w") as f:
    f.write("3\nWater\nO 0 0 0.127\nH 0 0.758 -0.509\nH 0 -0.758 -0.509\n")

r = gansu.Molecule("h2o.xyz", basis="cc-pvdz").run()

print("Orbital energies (Hartree):")
eps = r.orbital_energies
nocc = r.num_electrons // 2
for i, e in enumerate(eps):
    label = "occ" if i < nocc else "vir"
    print(f"  MO {i+1:3d} ({label})  {e:12.6f}")

print(f"\nHOMO-LUMO gap: {(eps[nocc] - eps[nocc-1]) * 27.2114:.2f} eV")

gansu.finalize()
```

### 6. CCSD 1-RDM (Correlation Density)

```python
import gansu
import numpy as np

gansu.init()

with open("h2o.xyz", "w") as f:
    f.write("3\nWater\nO 0 0 0.127\nH 0 0.758 -0.509\nH 0 -0.758 -0.509\n")

r = gansu.Molecule("h2o.xyz", basis="sto-3g").run(post_hf="ccsd_density")

D = r.ccsd_1rdm_mo
print(f"1-RDM shape: {D.shape}")
print(f"Trace(D) = {np.trace(D):.6f}  (should be {r.num_electrons})")
print(f"Natural occupations: {np.sort(np.linalg.eigvalsh(D))[::-1]}")

gansu.finalize()
```

### 7. Potential Energy Scan (Bond Stretching)

```python
import gansu
import numpy as np

gansu.init()

# OH stretch in water: fix one H, move the other
distances = np.linspace(0.8, 2.5, 20)
energies = []

for d in distances:
    with open("h2o_scan.xyz", "w") as f:
        f.write(f"3\nH2O OH scan\nO 0 0 0\nH 0 0.758 -0.509\nH 0 {-0.758*d/0.958:.6f} {-0.509*d/0.958:.6f}\n")
    r = gansu.Molecule("h2o_scan.xyz", basis="cc-pvdz").run(post_hf="mp2")
    E = r.total_energy + r.post_hf_energy
    energies.append(E)
    print(f"d={d:.3f} A  E={E:.8f}")

gansu.finalize()
```

### 8. Basis Set Convergence Study

```python
import gansu

gansu.init()

with open("h2o.xyz", "w") as f:
    f.write("3\nWater\nO 0 0 0.127\nH 0 0.758 -0.509\nH 0 -0.758 -0.509\n")

print(f"{'Basis':>12s}  {'nao':>5s}  {'HF':>14s}  {'CCSD':>14s}  {'Corr':>12s}")
print("-" * 65)

for basis in ["sto-3g", "3-21g", "6-31g", "6-31g_st", "cc-pvdz", "cc-pvtz"]:
    try:
        r = gansu.Molecule("h2o.xyz", basis=basis).run(post_hf="ccsd")
        e_hf = r.total_energy
        e_corr = r.post_hf_energy
        print(f"{basis:>12s}  {r.num_basis:5d}  {e_hf:14.8f}  {e_hf+e_corr:14.8f}  {e_corr:12.8f}")
    except Exception as e:
        print(f"{basis:>12s}  FAILED: {e}")

gansu.finalize()
```

### 9. Context Manager

```python
import gansu

with gansu.session():  # init + finalize automatically
    r = gansu.Molecule("h2o.xyz", basis="cc-pvdz").run(post_hf="mp2")
    print(f"E = {r.total_energy + r.post_hf_energy:.8f}")
```

### 10. CPU-Only Mode

```python
import gansu

gansu.init(force_cpu=True)  # No GPU used

r = gansu.Molecule("h2o.xyz", basis="sto-3g").run(post_hf="ccsd")
print(f"E = {r.total_energy + r.post_hf_energy:.8f}")

gansu.finalize()
```

---

## Full Parameter Reference

All parameters accepted by `mol.run(**kwargs)` and `gansu.Molecule(..., **kwargs)` are the same as the CLI parameters. See [parameters.md](parameters.md) for the complete list.

```python
# Example: passing arbitrary parameters
r = mol.run(
    method="RHF",
    post_hf="adc2",
    n_excited_states="10",
    adc2_solver="schur_omega",
    initial_guess="sad",
)
```

---

## Available Basis Sets

No `gansu.init()` required — works immediately:

```python
import gansu
print(gansu.list_basis_sets())
# ['3-21g', '4-31g', '6-311g_st_st', '6-31g', '6-31g_st',
#  'ano-rcc-mb', 'cc-pvdz', 'cc-pvqz', 'cc-pvtz', 'sto-3g']
```

From CLI:
```bash
gansu --list-basis
```

---

## Plotting Example (matplotlib)

```python
import gansu
import numpy as np
import matplotlib.pyplot as plt

gansu.init()

Rs = np.linspace(0.4, 5.0, 30)
methods = {"RHF": "none", "MP2": "mp2", "CCSD": "ccsd", "FCI": "fci"}
results = {m: [] for m in methods}

for R in Rs:
    with open("h2.xyz", "w") as f:
        f.write(f"2\nH2\nH 0 0 0\nH 0 0 {R}\n")
    for label, post_hf in methods.items():
        r = gansu.Molecule("h2.xyz", basis="cc-pvdz").run(post_hf=post_hf)
        results[label].append(r.total_energy + r.post_hf_energy)

gansu.finalize()

plt.figure(figsize=(8, 6))
for label, Es in results.items():
    plt.plot(Rs, Es, "o-", label=label)
plt.xlabel("H-H distance (A)")
plt.ylabel("Total energy (Hartree)")
plt.title("H2 dissociation curve")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("h2_dissociation.png", dpi=150)
plt.show()
```
