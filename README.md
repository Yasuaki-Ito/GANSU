# GANSU 

## Overview
GANSU (GPU Accelerated Numerical Simulation Utility) is an open-source quantum chemistry software designed for high-performance computations on modern computing architectures. This software aims to accelerate quantum chemistry simulations using advanced computational techniques such as GPU parallelization and efficient algorithms.

GANSU also supports a **CPU-only mode** for systems without NVIDIA GPUs, providing the same functionality using Eigen and OpenMP parallelization.

GANSU provides both a **C++ CLI** and a **Python API** for flexible usage.

## Features
* Hartree-Fock Methods: Includes RHF, UHF, and ROHF implementations.
* Parallel computing: Accelerates almost all operations on the GPU, achieving true speedup through custom implementations from scratch.
* Multi-GPU support: Distributed RI-HF across multiple GPUs via NCCL, with per-GPU B-matrix construction and AllReduce-based Fock build.
* CPU backend: Full CPU-only execution via `--cpu` flag (Eigen + OpenMP), supporting all HF methods, post-HF, gradient, Hessian, and geometry optimization.
* ECP support: Effective Core Potentials for heavy elements (LANL2DZ, cc-pVnZ-PP basis sets).
* Python API: Call GANSU from Python via `import gansu` with automatic basis set resolution.
* C API: Stable ABI for external bindings (`libgansu.so`).
* Flexible Input Options: Supports standard file formats such as XYZ and Gaussian basis set files.
* The numerical calculations in this software are performed using 64-bit double precision floating-point arithmetic.


### Supported computations
* Hartree-Fock methods
    * Restricted Hartree-Fock (RHF)
    * Unrestricted Hartree-Fock (UHF)
    * Restricted Open-Shell Hartree-Fock (ROHF)
    * ERI storage methods
      * Stored (full nao⁴ tensor)
      * Hash (sparse COO/hash table with Compact, Indexed, and Fullscan Fock construction, full post-HF support)
      * RI approximation (Density Fitting) (RHF, UHF, ROHF)
      * Semi-Direct-RI (RHF, UHF) — recomputes B matrix each iteration, J/K via BLAS
      * Direct-RI (RHF) — on-the-fly contraction without storing B matrix
      * Hash-RI (RHF) — 3-center ERIs in sparse COO, B matrix built on-demand from COO
      * Direct SCF (RHF)
* Post-Hartree-Fock methods
    * Møller-Plesset Perturbation Theory (RMP2, SCS-MP2, SOS-MP2, LT-MP2, LT-SOS-MP2, RMP3, RMP4, UMP2, UMP3)
    * Coupled Cluster (RCC2, RCCSD, RCCSD(T))
    * CCSD Lambda equations and 1-RDM (relaxed correlation density)
    * Density Matrix Embedding Theory with CCSD solver (DMET-CCSD, DMET-CCSD(T)) — fragment-based correlation, semi-canonical CCSD with f_ov support, μ-bisection density consistency, multi-GPU fragment parallelism, automatic X-H bond fragment detection, optional perturbative triples per fragment
    * Domain-based Local Pair Natural Orbital Coupled Cluster (DLPNO-CCSD, DLPNO-CCSD(T)) — Pipek-Mezey occupied localization, PAO + per-LMO domain, pair natural orbitals with PNO truncation, weak-pair MP2 reduction, multi-GPU per-triple parallelism with batched cuBLAS DGEMM kernels (RHF closed-shell, requires RI)
    * Full Configuration Interaction (RFCI)
    * RI support for all post-HF methods (AO ERI reconstructed from B matrix, nao⁴ intermediate skipped via direct MO ERI construction)
    * Semi-Direct RI and Direct-RI MP2 (B matrix built on-the-fly, no persistent naux×nao² storage)
    * RI CIS with B-matrix based sigma vector (no nmo⁴ MO ERI, O(naux×nmo²) memory)
* Excited state methods
    * Configuration Interaction Singles (CIS)
    * Algebraic Diagrammatic Construction (ADC(2), SOS-ADC(2), ADC(2)-x)
    * Equation-of-Motion MP2 (EOM-MP2)
    * Equation-of-Motion CC2 (EOM-CC2)
    * Equation-of-Motion CCSD (EOM-CCSD)
    * Singlet and triplet excited states (CIS, ADC(2), ADC(2)-x)
    * Oscillator strengths for all singlet excited state methods
    * RI support for all excited state methods
* Initial Guess
    * Core Hamiltonian (RHF, UHF, ROHF)
    * Generalized Wolfsberg-Helmholz (GWH) (RHF, UHF, ROHF)
    * Superposition of Atomic Densities (SAD) (RHF, UHF, ROHF)
    * Given density matrix (RHF, UHF, ROHF)
    * MINAO (RHF) — Projection of minimal ANO basis occupations (ANO-RCC-MB)
* Convergence algorithms
    * Damping (RHF, UHF, ROHF)
    * Optimal Damping (RHF, UHF, ROHF)
    * DIIS (RHF, UHF, ROHF)
    * SOSCF (RHF) — DIIS→Second-Order SCF automatic switching
    * ADIIS (RHF) — Augmented DIIS (JCP 132, 054109 (2010))
    * EDIIS (RHF) — Energy DIIS (JCP 116, 8255 (2002))
    * AEDIIS (RHF) — Automatic EDIIS→ADIIS→DIIS switching
* Molecular integrals
    * Overlap integrals
      * McMurchie-Davidson algorithm (s-, p-, d-, f-, g-, h-, and i-orbitals)
    * Kinetic energy and nuclear attraction integrals
      * McMurchie-Davidson algorithm (s-, p-, d-, f-, g-, h-, and i-orbitals)
      * Obara-Saika algorithm (s-, p-, d-, and f-orbitals)
    * Electron repulsion integrals
      * Rys quadrature (shell quartets involving d-orbitals or higher)
      * McMurchie-Davidson algorithm (fallback for all angular momenta)
      * Head-Gordon-Pople algorithm (s- and p-orbital only quartets)
      * Schwarz Screening
    * Electron repulsion integrals for density fitting (RI approximation)
      * McMurchie-Davidson algorithm (s-, p-, d-, f-, and g-orbitals)
      * Head-Gordon-Pople algorithm (s-, p-, and d-orbitals for basis functions, s-, p-, d-, f-, and g-orbitals for auxiliary basis functions)
    * Boys function
* Charge analysis
    * Mulliken population analysis (RHF, UHF, ROHF)
* Bond order analysis
    * Mayer bond order (RHF, UHF, ROHF)
    * Wiberg bond order (RHF, UHF, ROHF)
* Energy Gradient
    * Analytical energy gradient (RHF, UHF)
* Energy Hessian
    * Analytical Hessian (RHF) — skeleton (1e/2e/Vnn) + CPHF response
    * Vibrational frequency analysis (harmonic, with translation/rotation projection)
* Geometry Optimization
    * Quasi-Newton methods: BFGS, DFP, SR1
    * Newton-Raphson with analytical Hessian
    * Conjugate gradient methods: Fletcher-Reeves, Polak-Ribière, Hestenes-Stiefel, Dai-Yuan
    * GDIIS (Geometry DIIS)
    * Steepest Descent
    * Armijo backtracking line search (Quasi-Newton, Conjugate Gradient, Steepest Descent)
    * Translation and rotation projection
* Export
    * Export wave function information in the Molden format for visualization
        * Tested by [MOrbVis](https://yasuaki-ito.github.io/morbvis/), [Avogadro](https://avogadro.cc/), and [Pegamoid](https://github.com/Jellby/Pegamoid)
      ![Orbital renderred by MOrbVis](/doc/images/orbital.png)
      *Resulting molecular orbital of Benzene by MOrbVis*
    * Export Pipek-Mezey localized occupied orbitals (LMOs) in Molden format (`--export_lmo_molden`, RHF / UHF / ROHF)
* Effective Core Potentials (ECP)
    * LANL2DZ, cc-pVnZ-PP basis sets for heavy elements
    * GPU-accelerated ECP integral computation
* Multi-GPU (`--num_gpus`)
    * Distributed RI-HF with NCCL AllReduce
    * Per-GPU independent B-matrix construction (chunked 3-center ERI + L⁻¹ DGEMM)
    * Supports stored RI and Direct-RI modes
* CPU-only Backend (`--cpu`)
    * All HF methods (RHF, UHF, ROHF) with all ERI storage methods
    * All post-HF methods (MP2, MP3, MP4, CC2, CCSD, CCSD(T), FCI)
    * All excited state methods (CIS, ADC(2), ADC(2)-x, EOM-MP2, EOM-CC2, EOM-CCSD)
    * Analytical energy gradient (RHF, UHF) and geometry optimization
    * Analytical Hessian (RHF) with CPHF response, vibrational frequencies, and Newton-Raphson optimization
    * Population/bond order analysis and Molden export
    * Parallelized with OpenMP



### Todo / Not Implemented yet
* Convergence algorithms
  * Optimal Damping (UHF)
  * ADIIS/EDIIS/AEDIIS (UHF, ROHF)
* Initial Guess
  * Random
  * Load the precomputed coefficients/Fock matrix
* Excited State Methods
  * Time-Dependent Hartree-Fock (TDHF)
* Energy Gradient
  * RI-native gradient (3-center integral derivatives)
  * Post-HF energy gradient (MP2, CCSD, DLPNO, etc.)
* DLPNO methods
  * UHF / ROHF DLPNO (currently RHF closed-shell only)
  * Non-RI DLPNO (currently requires RI)
  * DLPNO gradient / dipole / response
* Energy Hessian
  * Analytical h1ao/s1ao derivatives (currently uses finite differences)
  * UHF Hessian
* Multi-GPU
  * UHF/ROHF distributed Fock build
  * Multi-GPU post-HF methods
* Density Functional Theory (DFT)
* GPU implementation
  * Total spin (UHF)
* Charge analysis
    * Lowdin population analysis
    * Hirshfeld population analysis

## Installation

### Quick install via pip (recommended)

```bash
pip install gansu
```

That's it — no CUDA toolkit required on the user side, no compilation, no `cmake`. The PyPI wheel ships a thin Python wrapper (~2.6 MB) and declares the CUDA runtime libraries (`nvidia-cublas-cu12`, `nvidia-cusolver-cu12`, `nvidia-nccl-cu12`, etc.) as dependencies, so they are pulled in automatically. The GPU-accelerated shared library `libgansu.so` (~460 MB, multi-arch fatbin covering SM 8.0–12.0) is downloaded from the matching [GitHub Release](https://github.com/Yasuaki-Ito/GANSU/releases/latest) on first use and cached under `~/.cache/gansu/<version>/` with SHA-256 verification.

**System requirements:**
* Linux x86_64 (manylinux_2_28+, i.e. Ubuntu 18.04+, Debian 10+, RHEL 8+ and equivalents)
* Python 3.10+
* NVIDIA driver ≥ 525.60.13 (CUDA 12.x runtime compatible)
* NVIDIA GPU with Compute Capability 8.0+ (Ampere / Ada / Hopper / Blackwell)

**Quick test:**
```python
import gansu

gansu.init()
m = gansu.Molecule("H2O.xyz", basis="sto-3g")
print(f"RHF energy: {m.run(method='RHF').total_energy:.6f} Ha")
gansu.finalize()
```

#### Direct URL install (e.g. for pinning a specific release)

```bash
pip install https://github.com/Yasuaki-Ito/GANSU/releases/download/v2026.5.9/gansu-2026.5.9-py3-none-manylinux_2_28_x86_64.whl
```

#### Offline / airgapped install

Download both the wheel and the matching shared library from a release page on a machine that has internet access, then on the target machine:

```bash
pip install ./gansu-2026.5.9-py3-none-manylinux_2_28_x86_64.whl
export GANSU_LIB=/path/to/libgansu-2026.5.9-linux-x86_64.so
python -c "import gansu; gansu.init()"
```

`GANSU_LIB` short-circuits the auto-download so the loader uses the explicit local copy.

### Build from source

The remaining sections — Prerequisites, Directory Structure, and Build instructions — are for developers and users who want to build GANSU themselves (e.g. to enable extended angular-momentum support, modify the source, or run on a non-`x86_64` Linux platform). Pure users can stop here and skip ahead to [Usage](#usage).

#### Prerequisites

##### GPU mode (default)
* Hardware
  * NVIDIA GPU with CUDA Compute Capability 8.0, 8.6, 9.0 or later
  * x86_64 / ARM architecture
* Software
  * C++ 17 or later
  * CMake 3.31 or later
  * NVIDIA CUDA Toolkit 12.9 or later
  * cuBLAS 12.9 or later
  * cuSOLVER 11.7 or later
  * [Eigen](https://eigen.tuxfamily.org/) 3.4+ (automatically downloaded via CMake FetchContent)
  * [OpenBLAS](https://www.openblas.net/) (optional but recommended, `sudo apt install libopenblas-dev` on Ubuntu) — automatically detected by CMake; significantly accelerates CPU-mode computation
  * [NCCL](https://developer.nvidia.com/nccl) (optional, required for multi-GPU support) — `sudo apt install libnccl-dev` on Ubuntu, or included in CUDA Toolkit

##### CPU-only mode (`--cpu`)
When a GPU is available, pass `--cpu` to force CPU execution. All features are supported with OpenMP parallelization.
No additional dependencies beyond the GPU mode prerequisites are required.


### Directory Structure

#### Top-level directory structure
```
.
├─ basis/
├─ auxiliary_basis/
├─ doc/
│   └─ html/
├─ include/
├─ parameter_recipe/
├─ python/
│   └─ gansu/
├─ script/
├─ src/
│   └─ boys/
├─ test/
├─ xyz/
│   ├─ large_molecular/
│   ├─ larger_molecular/
│   ├─ monatomic/
│   └─ optimization/
├─ CMakeLists.txt
├─ pyproject.toml
├─ LICENSE
├─ doc/parameters.md
└─ README.md
```

#### Description of the directories and files
| File/Directory | Description |
| --- | --- |
| `basis/` | Contains the basis set files (e.g., sto-3g.gbs) downloaded from [Basis Set Exchange](https://www.basissetexchange.org/), and the precomputed density matrix cache files (e.g., sto-3g.sad) for SAD |
| `auxiliary_basis/` | Contains the auxiliary basis set files (e.g., cc-pvdz-rifit.gbs) downloaded from [Basis Set Exchange](https://www.basissetexchange.org/) |
| `doc/` | Contains document materials |
| `doc/html/` | Contains the Doxygen-generated documentation |
| `include/` | Contains the header files |
| `parameter_recipe/` | Contains the parameter recipes for convenience |
| `python/gansu/` | Python package (`import gansu`) |
| `script/` | Script files |
| `src/` | Contains the source files |
| `src/boys/` | Contains a precomputed file for the Boys function |
| `test/` | Contains the test files |
| `xyz/` | Contains the XYZ files (e.g., H2O.xyz) |
| `xyz/large_molecular/` | Contains the XYZ files for large molecules (e.g., fullerene.xyz). RI approximation (density fitting) may be necessary for them. |
| `xyz/larger_molecular/` | Contains the XYZ files for larger molecules (e.g., C720.xyz). Direct-SCF may be necessary for them. |
| `xyz/monatomic/` | Contains the XYZ files for monatomic molecules (e.g., H.xyz) |
| `xyz/optimization/` | Contains the XYZ files with distorted geometries for geometry optimization tests (e.g., H2_stretched.xyz) |
| `CMakeLists.txt` | CMake configuration file |
| `pyproject.toml` | Python package configuration for pip |
| `LICENSE` | License file |
| `doc/parameters.md` | Parameter overview and description |
| `README.md` | Project overview and installation instructions |

### Build instructions
1. Copy the source code.
``` bash
git clone https://github.com/Yasuaki-Ito/GANSU.git
```

2. Create a build directory and configure the build using CMake:
``` bash
cd GANSU
mkdir build
cd build
cmake ..
```

To enable multi-GPU support (requires NCCL):
``` bash
cmake .. -DENABLE_MULTI_GPU=ON
```

3. Build the software using the generated Makefile:
``` bash
make
```
4. Run the H2 molecule example:
``` bash
./gansu -x ../xyz/H2.xyz -g sto-3g -m RHF
```

> [!NOTE]
> To enable support for higher angular momentum orbitals in the RI approximation, uncomment the relevant lines in CMakeLists.txt. Be aware that doing so may result in a substantially longer compilation time.

### Usage

#### Quick Examples

```bash
# RHF energy
./gansu -x ../xyz/H2O.xyz -g sto-3g

# CCSD correlation energy
./gansu -x ../xyz/H2O.xyz -g cc-pvdz --post_hf_method ccsd

# Geometry optimization
./gansu -x ../xyz/optimization/H2_stretched.xyz -g cc-pvdz -r optimize

# RI approximation for large molecules
./gansu -x ../xyz/large_molecular/fullerene.xyz -g sto-3g --eri_method ri -ag ../auxiliary_basis/cc-pvdz-rifit.gbs

# Multi-GPU RI-HF (auto-detect GPUs)
./gansu -x ../xyz/large_molecular/fullerene.xyz -g sto-3g --eri_method ri -ag ../auxiliary_basis/cc-pvdz-rifit.gbs --num_gpus 4

# DMET-CCSD (auto fragment detection by X-H bonds; benzene → 6 CH fragments)
./gansu -x ../xyz/Benzene.xyz -g sto-3g --eri_method ri -ag ../auxiliary_basis/cc-pvdz-rifit.gbs --post_hf_method dmet --num_gpus 4

# DMET-CCSD(T) (perturbative triples per fragment)
./gansu -x ../xyz/Benzene.xyz -g sto-3g --eri_method ri -ag ../auxiliary_basis/cc-pvdz-rifit.gbs --post_hf_method dmet_ccsd_t --num_gpus 4

# DLPNO-CCSD / DLPNO-CCSD(T) — local correlation, scales to ~100 atoms with RI
./gansu -x ../xyz/large_molecular/water_hexamer.xyz -g cc-pvdz --eri_method ri -ag ../auxiliary_basis/cc-pvdz-rifit.gbs --post_hf_method dlpno_ccsd --dlpno_preset normal
./gansu -x ../xyz/large_molecular/water_hexamer.xyz -g cc-pvdz --eri_method ri -ag ../auxiliary_basis/cc-pvdz-rifit.gbs --post_hf_method dlpno_ccsd_t --dlpno_preset normal

# Export Pipek-Mezey localized orbitals for visualization (Avogadro/Jmol/VMD)
./gansu -x ../xyz/Benzene.xyz -g cc-pvdz --export_lmo_molden 1

# CPU-only mode
./gansu -x ../xyz/H2O.xyz -g sto-3g --cpu

# List available basis sets
./gansu --list-basis
```

> [!TIP]
> Basis sets can be specified by name (e.g., `-g cc-pvdz`) or by full path (e.g., `-g ../basis/cc-pvdz.gbs`).

#### Python API

```python
import gansu

gansu.init()
r = gansu.Molecule("H2O.xyz", basis="cc-pvdz").run(post_hf="ccsd")
print(f"Energy: {r.total_energy + r.post_hf_energy:.8f} Hartree")
gansu.finalize()
```

#### Documentation

- **[CLI Usage Guide](doc/usage_cli.md)** — Full command-line reference with copy-paste examples
- **[Python API Guide](doc/usage_python.md)** — Python interface with scripting examples
- **[C API Reference](doc/usage_c_api.md)** — Language-agnostic C interface (C/Rust/Julia/JS/...)
- **[Parameters](doc/parameters.md)** — Complete parameter list and defaults


## License [![BSD 3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-orange)](https://opensource.org/licenses/BSD-3-Clause)


GANSU (GPU Accelerated Numerical Simulation Utility)

Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited All rights reserved.

This software is licensed under the BSD 3-Clause License.
You may obtain a copy of the license in the LICENSE file
located in the root directory of this source tree or at:
https://opensource.org/licenses/BSD-3-Clause

## Citation
The journal article reference describing GANSU is:

* Yasuaki Ito, Satoki Tsuji, Koji Nakano, and Akihiko Kasagi, GANSU: A GPU-Native Quantum Chemistry Framework for Efficient Hartree–Fock and Post-HF Calculations. Eng, vol. 7, no. 5, 205, 2026. ([DOI](https://doi.org/10.3390/eng7050205))

## Publications
  1. Satoki Tsuji, Yasuaki Ito, Haruto Fujii, Nobuya Yokogawa, Kanta Suzuki, Koji Nakano, Victor Parque, Akihiko Kasagi, Dynamic Schwarz Screening for GPU-Accelerated Fock Matrix Computation, Concurrency and Computation: Practice and Experience, vol. 38, no. 9, e70583, 2026. ([DOI](https://doi.org/10.1002/cpe.70583))
  1. Yasuaki Ito, Satoki Tsuji, Koji Nakano, and Akihiko Kasagi, GANSU: A GPU-Native Quantum Chemistry Framework for Efficient Hartree–Fock and Post-HF Calculations. Eng, vol. 7, 205, 2026. ([DOI](https://doi.org/10.3390/eng7050205))
  1. Nobuya Yokogawa, Yasuaki Ito, Satoki Tsuji, Haruto Fujii, Kanta Suzuki, Koji Nakano, Victor Parque, Akihiko Kasagi, GPU-Accelerated One-Electron Integral Computation for Quantum Chemistry, Concurrency and Computation: Practice and Experience, vol. 38, no. 5, e70628, 2026. ([DOI](https://doi.org/10.1002/cpe.70628))
  1. Kanta Suzuki, Yasuaki Ito, Haruto Fujii, Nobuya Yokogawa, Satoki Tsuji, Koji Nakano, Victor Parque, Akihiko Kasagi, Efficient GPU Implementations of Three-Center Two-Electron Repulsion Integrals, Concurrency and Computation: Practice and Experience, vol. 37, no. 25-26, e70328, 2025. ([DOI](https://doi.org/10.1002/cpe.70328))
  1. Satoki Tsuji, Yasuaki Ito, Nobuya Yokogawa, Kanta Suzuki, Koji Nakano, Victor Parque and Akihiko Kasagi, A GPU Implementation of the Second- and Third-Order Møller-Plesset Perturbation Theory, in Proc. of International Symposium on Computing and Networking Workshops, pp. 90-96, Yamagata, Yamagata, November 2025. ([DOI](https://doi.ieeecomputersociety.org/10.1109/CANDARW68385.2025.00023))
  1. Hong Gao, Yasuaki Ito, Koji Nakano, Satoshi Imamura, Akihiko Kasagi and Satoki Tsuji, Fully GPU-Accelerated Full Configuration Interaction for Exact Molecular Ground-state Energy Calculation, in Proc. of International Symposium on Computing and Networking, pp. 48-57, Yamagata, Yamagata, November 2025. ([DOI](https://doi.ieeecomputersociety.org/10.1109/CANDAR68384.2025.00014))
  1. Kanta Suzuki, Yasuaki Ito, Nobuya Yokogawa, Satoki Tsuji, Koji Nakano, Victor Parque and Akihiko Kasagi, GPU Acceleration of RI-RMP2 Correlation Energy Computation, in Proc. of International Symposium on Computing and Networking, pp. 174-180, Yamagata, Yamagata, November 2025. ([DOI](https://doi.ieeecomputersociety.org/10.1109/CANDAR68384.2025.00031))
  1. Satoki Tsuji, Yasuaki Ito, Haruto Fujii, Nobuya Yokogawa, Kanta Suzuki, Koji Nakano, Victor Parque, Akihiko Kasagi, GPU-Accelerated Fock Matrix Computation with Efficient Reduction, Applied Sciences, vol. 15, no. 9, 4779, April 2025. ([DOI](https://doi.org/10.3390/app15094779))
  1. Haruto Fujii, Yasuaki Ito, Nobuya Yokogawa, Kanta Suzuki, Satoki Tsuji, Koji Nakano, Victor Parque, Akihiko Kasagi, Efficient GPU Implementation of the McMurchie-Davidson Method for Shell-Based ERI Computations, Applied Sciences, vol. 15, no. 5, 2572, February 2025. ([DOI](https://doi.org/10.3390/app15052572))
  1. Satoki Tsuji, Yasuaki Ito, Koji Nakano, Akihiko Kasagi, GPU Acceleration of the Boys Function Evaluation in Computational Quantum Chemistry, Concurrency and Computation: Practice and Experience, vol. 37, no. 2, e8328, 2025. ([DOI](https://doi.org/10.1002/cpe.8328))
 1. Kanta Suzuki, Yasuaki Ito, Haruto Fujii, Nobuya Yokogawa, Satoki Tsuji, Koji Nakano and Akihiko Kasagi, GPU Acceleration of Head-Gordon-Pople Algorithm, in Proc. of International Symposium on Computing and Networking, pp. 115-124, Naha, Okinawa, November 2024. ([DOI](https://doi.ieeecomputersociety.org/10.1109/CANDAR64496.2024.00021)) 
 1. Nobuya Yokogawa, Yasuaki Ito, Satoki Tsuji, Haruto Fujii, Kanta Suzuki, Koji Nakano and Akihiko Kasagi, Parallel GPU Computation of Nuclear Attraction Integrals in Quantum Chemistry, in Proc. of International Symposium on Computing and Networking Workshops, pp. 163-169, Naha, Okinawa, November 2024. ([DOI](https://doi.org/10.1109/CANDARW64572.2024.00033)) 
 1. Satoki Tsuji, Yasuaki Ito, Haruto Fujii, Nobuya Yokogawa, Kanta Suzuki, Koji Nakano and Akihiko Kasagi, Dynamic Screening of Two-Electron Repulsion Integrals in GPU Parallelization, in Proc. of International Symposium on Computing and Networking Workshops, pp. 211-217, Naha, Okinawa, November 2024.  ([DOI](https://doi.org/10.1109/CANDARW64572.2024.00041)) 
 1. Haruto Fujii, Yasuaki Ito, Nobuya Yokogawa, Kanta Suzuki, Satoki Tsuji, Koji Nakano, and Akihiko Kasagi, A GPU Implementation of McMurchie-Davidson Algorithm for Two-Electron Repulsion Integral Computation, in Proc. of 15th International Conference of Parallel Processing and Applied Mathematics (PPAM 2024), LNCS 15579, pp. 210-224, 2025. ([DOI](https://doi.org/10.1007/978-3-031-85697-6_14))
 1. Yasuaki Ito, Satoki Tsuji, Haruto Fujii, Kanta Suzuki, Nobuya Yokogawa, Koji Nakano, Akihiko Kasagi, Introduction to Computational Quantum Chemistry for Computer Scientists, in Proc. of International Parallel and Distributed Processing Symposium Workshops, pp. 273-282, May 2024. ([DOI](https://doi.ieeecomputersociety.org/10.1109/IPDPSW63119.2024.00066))
 1. Satoki Tsuji, Yasuaki Ito, Koji Nakano, Akihiko Kasagi, Efficient GPU-Accelerated Bulk Evaluation of the Boys Function for Quantum Chemistry, in Proc. of International Symposium on Computing and Networking Workshops, pp. 49-58, Matsue, Shimane, November 2023. ([DOI](https://doi.org/10.1109/CANDAR60563.2023.00014))

## Additional resources (Japanese)
A series of articles explaining how to use GANSU in Japanese is available on Zenn:

* [GANSUで始める量子化学計算 記事一覧](https://zenn.dev/comp_lab/articles/29e73268f402b6)
