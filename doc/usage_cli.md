# GANSU CLI Usage Guide

## Quick Start

```bash
# Build
cd GANSU && mkdir build && cd build
cmake .. && make

# Run
./gansu -x ../xyz/H2O.xyz -g sto-3g -m RHF
```

## Preparing Input Files

### XYZ File Format

XYZ files specify molecular geometry. Format: number of atoms, comment line, then atom coordinates in Angstrom.

**H2 molecule** (`h2.xyz`):
```
2
Hydrogen molecule
H  0.000  0.000  0.000
H  0.000  0.000  0.740
```

**H2O molecule** (`h2o.xyz`):
```
3
Water molecule
O   0.000   0.000   0.127
H   0.000   0.758  -0.509
H   0.000  -0.758  -0.509
```

**NH3 molecule** (`nh3.xyz`):
```
4
Ammonia
N   0.000   0.000   0.117
H   0.000   0.939  -0.273
H   0.813  -0.470  -0.273
H  -0.813  -0.470  -0.273
```

**CH4 molecule** (`ch4.xyz`):
```
5
Methane
C   0.000   0.000   0.000
H   0.629   0.629   0.629
H  -0.629  -0.629   0.629
H  -0.629   0.629  -0.629
H   0.629  -0.629  -0.629
```

**Benzene** (`benzene.xyz`):
```
12
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
```

### Basis Sets

Basis sets are specified by name. GANSU automatically resolves the path.

| Name | Description | Typical use |
|------|-------------|-------------|
| `sto-3g` | Minimal basis | Quick tests |
| `3-21g` | Split-valence | Preliminary calculations |
| `6-31g` | Split-valence | General use |
| `6-31g_st` | With polarization (6-31G*) | Better accuracy |
| `cc-pvdz` | Correlation-consistent DZ | Post-HF calculations |
| `cc-pvtz` | Correlation-consistent TZ | High-accuracy post-HF |
| `cc-pvqz` | Correlation-consistent QZ | Benchmark calculations |

---

## Basic Usage

### Syntax
```
./gansu -x <xyz_file> -g <basis> [options]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `-x` | XYZ file path | (required) |
| `-g` | Basis set name or path | (required) |
| `-m` | HF method: `RHF`, `UHF`, `ROHF` | `RHF` |
| `-r` | Run type: `energy`, `gradient`, `optimize`, `hessian` | `energy` |
| `-c` | Molecular charge | `0` |
| `--post_hf_method` | Post-HF method (see below) | `none` |
| `--n_excited_states` | Number of excited states | `3` |
| `--eri_method` | ERI method: `stored`, `RI`, `direct`, `hash` | `stored` |
| `--convergence_method` | SCF convergence: `DIIS`, `SOSCF`, etc. | `DIIS` |
| `--initial_guess` | Initial guess: `core`, `gwh`, `sad`, `minao` | `core` |
| `--optimizer` | Geometry optimizer (see below) | `bfgs` |
| `--cpu` | Force CPU-only execution | (off) |
| `--list-basis` | List available basis sets and exit | |
| `-p` | Parameter recipe file | (none) |

### Post-HF Methods

| Value | Method |
|-------|--------|
| `none` | HF only |
| `mp2` | MP2 |
| `mp3` | MP3 |
| `mp4` | MP4 |
| `cc2` | CC2 |
| `ccsd` | CCSD |
| `ccsd_t` | CCSD(T) |
| `ccsd_density` | CCSD + Lambda + 1-RDM |
| `fci` | Full CI |
| `cis` | CIS excited states |
| `adc2` | ADC(2) excited states |
| `adc2x` | ADC(2)-x excited states |
| `eom_mp2` | EOM-MP2 excited states |
| `eom_cc2` | EOM-CC2 excited states |
| `eom_ccsd` | EOM-CCSD excited states |

### Optimizers

`bfgs`, `dfp`, `sr1`, `gdiis`, `newton`, `cg-fr`, `cg-pr`, `cg-hs`, `cg-dy`, `sd`

---

## Examples

### 1. Basic HF Energy

```bash
cat > h2o.xyz << 'EOF'
3
Water
O   0.000   0.000   0.127
H   0.000   0.758  -0.509
H   0.000  -0.758  -0.509
EOF

./gansu -x h2o.xyz -g sto-3g
```

### 2. Different Basis Sets

```bash
./gansu -x h2o.xyz -g sto-3g        # Minimal basis
./gansu -x h2o.xyz -g cc-pvdz       # Double-zeta
./gansu -x h2o.xyz -g cc-pvtz       # Triple-zeta
```

### 3. Post-HF Correlation Energy

```bash
# MP2
./gansu -x h2o.xyz -g cc-pvdz --post_hf_method mp2

# CCSD
./gansu -x h2o.xyz -g cc-pvdz --post_hf_method ccsd

# CCSD(T)
./gansu -x h2o.xyz -g cc-pvdz --post_hf_method ccsd_t

# Full CI (exact, small molecules only)
./gansu -x h2o.xyz -g sto-3g --post_hf_method fci
```

### 4. Excited State Calculations

```bash
# CIS — 10 excited states
./gansu -x h2o.xyz -g cc-pvdz --post_hf_method cis --n_excited_states 10

# ADC(2) — accurate excited states
./gansu -x h2o.xyz -g cc-pvdz --post_hf_method adc2

# EOM-CCSD — gold standard for excited states
./gansu -x h2o.xyz -g cc-pvdz --post_hf_method eom_ccsd

# Triplet states
./gansu -x h2o.xyz -g cc-pvdz --post_hf_method adc2 --spin_type triplet
```

### 5. Energy Gradient

```bash
./gansu -x h2o.xyz -g cc-pvdz -r gradient
```

### 6. Geometry Optimization

```bash
# Create a distorted H2
cat > h2_stretched.xyz << 'EOF'
2
Stretched H2
H  0.000  0.000  0.000
H  0.000  0.000  1.500
EOF

# Optimize with BFGS (default)
./gansu -x h2_stretched.xyz -g cc-pvdz -r optimize

# Optimize with Newton-Raphson (uses analytical Hessian)
./gansu -x h2_stretched.xyz -g cc-pvdz -r optimize --optimizer newton
```

### 7. Vibrational Frequencies (Hessian)

```bash
./gansu -x h2o.xyz -g cc-pvdz -r hessian
```

### 8. UHF for Open-Shell Systems

```bash
# Hydrogen atom (1 electron)
cat > h_atom.xyz << 'EOF'
1
Hydrogen atom
H  0.000  0.000  0.000
EOF

./gansu -x h_atom.xyz -g cc-pvdz -m UHF
```

### 9. Charged Molecules

```bash
# OH- anion
cat > oh.xyz << 'EOF'
2
Hydroxide
O  0.000  0.000  0.000
H  0.000  0.000  0.970
EOF

./gansu -x oh.xyz -g cc-pvdz -c -1
```

### 10. RI Approximation for Large Molecules

```bash
# Explicit auxiliary basis
./gansu -x benzene.xyz -g cc-pvdz --eri_method ri -ag ../auxiliary_basis/cc-pvdz-rifit.gbs

# Auto-generated auxiliary basis
./gansu -x benzene.xyz -g cc-pvdz --eri_method ri
```

### 11. CCSD Density Matrix (for DMET)

```bash
./gansu -x h2o.xyz -g cc-pvdz --post_hf_method ccsd_density
```

### 12. CPU-Only Mode

```bash
./gansu -x h2o.xyz -g sto-3g --cpu
./gansu -x h2o.xyz -g cc-pvdz --post_hf_method ccsd --cpu
```

### 13. Using a Parameter Recipe File

```bash
cat > recipe.txt << 'EOF'
method = RHF
convergence_method = DIIS
initial_guess = sad
eri_method = stored
EOF

./gansu -p recipe.txt -x h2o.xyz -g cc-pvdz --post_hf_method mp2
```

---

### 14. List Available Basis Sets

```bash
./gansu --list-basis
```

---

## Full Parameter Reference

See [parameters.md](parameters.md) for the complete list of all parameters, types, and defaults.

---

## Tips

- **Basis set names are case-insensitive**: `cc-pvdz`, `CC-PVDZ`, `cc-pVDZ` all work.
- **SAD initial guess** (`--initial_guess sad`) is recommended for faster SCF convergence on molecules larger than H2.
- **RI approximation** (`--eri_method ri`) dramatically reduces memory usage for larger molecules. Performance is comparable to stored ERI for small molecules.
- **GPU acceleration** is automatic when an NVIDIA GPU is detected. Use `--cpu` to force CPU execution for benchmarking or debugging.
