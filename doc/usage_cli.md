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

By default GANSU uses **Cartesian** Gaussians (6 d-functions, 10 f, 15 g). To match the **spherical-harmonic** convention used by ORCA / PySCF / NWChem for cc-pVnZ and similar basis sets (5 d, 7 f, 9 g), add `--use_spherical 1`. See [Example 2b](#2b-spherical-harmonic-basis).

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
| `--use_spherical` | Spherical-harmonic basis (5D/7F/9G) instead of Cartesian (6D/10F/15G) | `0` |
| `--num_gpus` | Number of GPUs for multi-GPU RI (-1 = auto-detect all) | `-1` |
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
| `dmet` | DMET-CCSD (fragment-based correlation, multi-GPU, auto X-H bond fragmentation) |
| `dmet_ccsd_t` | DMET-CCSD(T) — same fragment construction plus perturbative triples per fragment |
| `dlpno_ccsd` | DLPNO-CCSD (closed-shell RHF, local-correlation CCSD, requires RI) |
| `dlpno_ccsd_t` | DLPNO-CCSD(T) — perturbative triples on per-triple TNO bases, multi-GPU batched |
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

### 2b. Spherical-Harmonic Basis

By default GANSU uses Cartesian Gaussians (6D/10F/15G). Add `--use_spherical 1`
to use pure spherical harmonics (5D/7F/9G, Molden ordering), matching the
ORCA / PySCF / NWChem convention for cc-pVnZ and similar basis sets.

```bash
# Cartesian (default) vs spherical RHF energy
./gansu -x h2o.xyz -g cc-pvdz                       # Cartesian (25 basis functions)
./gansu -x h2o.xyz -g cc-pvdz --use_spherical 1     # Spherical (24 basis functions)

# Works across stored / RI / multi-GPU and gradients + optimization
./gansu -x benzene.xyz -g cc-pvdz --use_spherical 1
./gansu -x h2o.xyz -g cc-pvdz --eri_method ri -ag cc-pvdz-rifit.gbs \
        -r optimize --use_spherical 1 --num_gpus 4
```

Spherical basis is supported for RHF/UHF/ROHF energy (stored, RI, multi-GPU
distributed RI), RI post-HF (MP2/CCSD/CIS/EOM/ADC(2)/DLPNO/STEOM), THC, all
initial guesses (core/gwh/sad/minao for RHF; core/gwh/sad for UHF/ROHF), ECP,
analytical gradient and geometry optimization, analytical Hessian and
vibrational frequencies, and Molden export. Direct-SCF/Hash ERIs and the
(experimental) MP2 gradient remain Cartesian-only and raise a clear error under
`--use_spherical 1`.

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

#### Multi-node / multi-GPU Full-CI

The Full-CI vector grows factorially and quickly exceeds a single GPU's memory
(e.g. C₂/6-31g has ~3.4×10⁸ determinants and does not fit one GPU). GANSU ships a
distributed solver (`fci_mpi`) that shards the FCI vector across MPI ranks — one
GPU per rank — using NCCL collectives for the Davidson sigma builds (alpha-string
row-block partitioning, with AllGather/ReduceScatter overlapped against compute).

Build with MPI enabled (requires the MPI dev headers and NCCL):

```bash
cd build
cmake .. -DENABLE_MPI=ON
make
```

Launch with `mpirun` through the `script/gansu_mpi.sh` wrapper, which pins each
rank to exactly one GPU (`CUDA_VISIBLE_DEVICES=$LOCAL_RANK`) so every rank sees
its GPU as device 0:

```bash
# 2 ranks → 2 GPUs
mpirun -np 2 --bind-to none ../script/gansu_mpi.sh \
       ./gansu -x ../xyz/C2.xyz -g 6-31g --post_hf_method fci
```

The dispatch is automatic: with more than one rank the distributed `fci_mpi`
solver runs; with `-np 1` (or a non-MPI build) the single-GPU `fci()` path is
used and results are byte-identical to before. The two paths agree to machine
precision (H₂O/sto-3g cross-check: single-GPU vs. 2-rank FCI energy differ by
~1×10⁻¹³ Ha).

> [!NOTE]
> `script/gansu_mpi.sh` resolves the node-local rank from Open MPI, MVAPICH2,
> Slurm (`srun`), or MPICH/Intel MPI environment variables, so the same wrapper
> works across launchers and scales to multiple nodes unchanged (one GPU per rank).

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

### 4b. DMET-CCSD (Fragment-Based Correlation)

DMET-CCSD partitions the molecule into atom-localized fragments, builds an embedding cluster (fragment AOs + Schmidt-decomposed bath orbitals) for each, and runs CCSD per cluster. Total correlation is reconstructed by democratic projection. Multi-GPU fragment parallelism gives near-linear speedup with the number of unique fragments.

```bash
# Auto fragment detection (recommended): each heavy atom + nearest H atoms
# Benzene → 6 CH fragments, 2 unique by symmetry
./gansu -x ../xyz/Benzene.xyz -g sto-3g --eri_method ri \
    -ag ../auxiliary_basis/cc-pvdz-rifit.gbs \
    --post_hf_method dmet --num_gpus 4

# DMET-CCSD(T) — perturbative triples added per fragment at converged μ_DMET
./gansu -x ../xyz/Benzene.xyz -g sto-3g --eri_method ri \
    -ag ../auxiliary_basis/cc-pvdz-rifit.gbs \
    --post_hf_method dmet_ccsd_t --num_gpus 4

# Manual fragment specification (atom indices, 0-indexed)
./gansu -x ../xyz/Benzene.xyz -g sto-3g --post_hf_method dmet \
    --dmet_fragments "{0,6} {1,7} {2,8} {3,9} {4,10} {5,11}"

# Vayesta-compatible loose tolerance (faster bisection, ~1% N deviation)
./gansu -x ../xyz/Benzene.xyz -g sto-3g --post_hf_method dmet \
    --dmet_n_tol 4.2e-3

# Per-fragment debug output (max|f_ov|, ε spectrum, D_cluster eigenvalues)
GANSU_DMET_VERBOSE=1 ./gansu ../xyz/Benzene.xyz -g sto-3g --post_hf_method dmet
```

The output reports both T-amplitude democratic energy and the standard QC-DMET (Vayesta-convention) energy:

```
---- DMET-CCSD Summary ----
  Chemical potential μ_DMET (CCSD-relaxed): -1.144e-03 Ha
  Total DMET-CCSD correlation energy (T-amp democratic): -0.3043 Ha
  Total DMET-CCSD correlation energy (DMET, Vayesta):    -0.4938 Ha
  HF energy:                              -227.8926 Ha
  DMET-CCSD total energy (DMET):          -228.3864 Ha
```

### 4c. DLPNO-CCSD / DLPNO-CCSD(T) (Local Correlation)

DLPNO-CCSD and DLPNO-CCSD(T) (Riplinger & Neese, 2013/2016) exploit four nested locality layers — occupied LMO localization (Pipek-Mezey), projected atomic orbitals + per-LMO atom domains, per-pair PNO truncation, and strong/weak-pair partitioning — to achieve near-canonical CCSD(T) accuracy at near-linear scaling. GANSU implements the **closed-shell RHF** variant with GPU-accelerated per-triple kernels (cuBLAS batched DGEMM, memory-aware chunked flush) and multi-GPU per-triple parallelism. The RI back-end is required.

```bash
# DLPNO-CCSD on water hexamer (normal preset)
./gansu -x ../xyz/large_molecular/water_hexamer.xyz -g cc-pvdz \
    --eri_method ri -ag ../auxiliary_basis/cc-pvdz-rifit.gbs \
    --post_hf_method dlpno_ccsd --dlpno_preset normal --num_gpus 8

# DLPNO-CCSD(T) — adds perturbative triples (TNO basis, PySCF-equivalent 6-W formula)
./gansu -x ../xyz/large_molecular/water_hexamer.xyz -g cc-pvdz \
    --eri_method ri -ag ../auxiliary_basis/cc-pvdz-rifit.gbs \
    --post_hf_method dlpno_ccsd_t --dlpno_preset normal --num_gpus 8

# Tighter accuracy (closer to canonical, larger PNO/TNO basis)
./gansu ... --post_hf_method dlpno_ccsd_t --dlpno_preset tight

# Per-section profile (DLPNO-(T) triple loop breakdown)
./gansu ... --post_hf_method dlpno_ccsd_t --dlpno_verbose 2
```

ORCA-compatible truncation presets: `loose` / `normal` (default) / `tight` / `very_tight`. See [DLPNO-CCSD / DLPNO-CCSD(T) parameters](parameters.md#dlpno-ccsd--dlpno-ccsdt-parameters) for the underlying `t_cut_*` cutoffs and fine-tuning options.

### 4d. Exporting Localized Orbitals

The Pipek-Mezey localization from the DLPNO pipeline can be repurposed to write occupied LMOs to a Molden file for visualization in Avogadro / Jmol / VMD / MOrbVis. Works for any SCF method (RHF / UHF / ROHF) — DLPNO is not actually run.

```bash
# Closed-shell: 1 localization, writes occupied LMOs + canonical virtuals
./gansu -x ../xyz/Benzene.xyz -g cc-pvdz --export_lmo_molden 1
# → output_lmo.molden

# UHF: α and β occupied blocks localized independently
./gansu -x ../xyz/radical.xyz -g cc-pvdz -m UHF --export_lmo_molden 1
```

For ROHF, doubly-occupied and singly-occupied subspaces are localized separately so closed-shell core and open-shell electrons do not mix. Pass `--export_molden 1 --export_lmo_molden 1` together to emit both canonical (`output.molden`) and localized (`output_lmo.molden`) files in a single run.

### 5. Energy Gradient

```bash
# Stored-ERI analytical gradient
./gansu -x h2o.xyz -g cc-pvdz -r gradient

# RI-native analytical gradient (3c/2c integral derivatives), single- or multi-GPU
./gansu -x h2o.xyz -g cc-pvdz --eri_method ri -ag cc-pvdz-rifit.gbs -r gradient
./gansu -x h2o.xyz -g cc-pvdz --eri_method ri -ag cc-pvdz-rifit.gbs -r gradient --num_gpus 4

# Spherical-basis gradient
./gansu -x h2o.xyz -g cc-pvdz --eri_method ri -ag cc-pvdz-rifit.gbs -r gradient --use_spherical 1
```

Analytical gradients are available for RHF/UHF (stored, Direct) and RHF (RI,
single- and multi-GPU), in both Cartesian and spherical bases. Post-HF gradients
(MP2/CCSD/...) are not yet supported.

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
