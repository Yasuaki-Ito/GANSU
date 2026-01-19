# GANSU (GPU Accelerated Numerical Simulation Utility)

## Overview
GANSU (GPU Accelerated Numerical Simulation Utility) is an open-source quantum chemistry software designed for high-performance computations on modern computing architectures. This software aims to accelerate quantum chemistry simulations using advanced computational techniques such as GPU parallelization and efficient algorithms.

## Features
* Hartree-Fock Methods: Includes RHF, UHF, and ROHF implementations.
* Parallel computing: Accelerates almost all operations on the GPU, achieving true speedup through custom implementations from scratch.
* Flexible Input Options: Supports standard file formats such as XYZ and Gaussian basis set files.
* etc.


### Supported computations
* Hartree-Fock
    * Restricted Hartree-Fock (RHF)
    * Unrestricted Hartree-Fock (UHF)
    * Restricted Open-Shell Hartree-Fock (ROHF)
* Initial Guess
    * Core Hamiltonian (RHF, UHF, ROHF)
    * Generalized Wolfsberg-Helmholz (GWH) @cite Wolfsberg1952 (RHF, UHF, ROHF)
    * Superposition of Atomic Densities (SAD) @cite Lenthe2006 (RHF, UHF, ROHF)
    * Given density matrix (RHF, UHF, ROHF)
* Convergence algorithms
    * Damping @cite Zerner1979 (RHF, UHF, ROHF)
    * Optimal Damping @cite Cances2000 (RHF, ROHF)
    * DIIS @cite Pulay1980, @cite Pulay1982 (RHF, UHF, ROHF)
* Molecular integrals
    * Overlap integrals (s-, p-, d-, and f-orbitals)
    * Kinetic energy and nuclear attraction integrals  (s-, p-, d-, and f-orbitals)
      * McMurchie-Davidson algorithm @cite McMurchie1978, @cite Yokogawa2024
      * Obara-Saika algorithm @cite Obara1988, @cite Yokogawa2024
    * Electron repulsion integrals
      * McMurchie-Davidson algorithm (s-, p-, d-, f-, g-, h-, i-orbitals) @cite McMurchie1978, @cite Fujii2024 @cite Fujii2025
      * Head-Godon-Pople algorithm (s- and p-orbitals) @cite Martin1988, @cite Suzuki2024
    * Boys function @cite Tsuji2023, @cite Tsuji2025
    * Swartz Screening @cite Gill1994, @cite Tsuji2024
* Charge analysis
    * Mulliken population analysis @cite Mulliken1955  (RHF, UHF, ROHF)
* Export
    * Export wave function information in the Molden format for visualization
        * Tested by [Avogadro](https://avogadro.cc/) @cite Hanwell2012 and [Pegamoid](https://github.com/Jellby/Pegamoid) 

### Todo / Not Implemented yet
* Molecular integrals
  * ERI memory is reduced from \f$ M^4 \f$ to \f$ {1 \over 8}M^4 \f$
* Convergence algorithms
  * Optimal Damping (UHF)
  * EDIIS
  * ADIIS
* Initial Guess
  * Random
  * Load the precomputed coefficients/Fock matrix
* Density Fitting
* Direct SCF
* Post-Hartree-Fock methods
  * Integral Transformation (AO -> MO)
  * Configuration Interaction (CI)
  * Coupled Cluster (CC)
  * Moller-Plesset Perturbation Theory (MP)
* Density Functional Theory (DFT)
* GPU implementation
  * Total spin \f$ \langle S^2 \rangle \f$ (UHF)
* Charge analysis
    * Lowdin population analysis @cite Lowdin1950
    * Hirshfeld population analysis @cite Hirshfeld1977


## Installation

### Prerequisites
* Hardware
  * NVIDIA GPU with CUDA Compute Capability 8.0 or later
  * x86_64 architecture
* Software
  * C++ 17 or later
  * CMake 3.31 or later
  * NVIDIA CUDA Toolkit 11.2 or later
  * cuBLAS 11.4 or later
  * cuSOLVER 11.1 or later


### Directory Structure

#### Top-level directory structure
```
.
├─ basis/
├─ doc/
├─ html/
├─ include/
├─ parameter_recipe/
├─ src/
│   └─ boys/
├─ xyz/
│   └─ monatomic/
├─ CMakeLists.txt
├─ Doxyfile
├─ License.txt
└─ README.md
```

#### Description of the directories and files
| File/Directory | Description |
| --- | --- |
| `basis/` | Contains the basis set files (e.g., sto-3g.gbs) downloaded from [Basis Set Exchange](https://www.basissetexchange.org/) @cite BasisSetExchange |
| `doc/` | Contains document materials for Doxygen |
| `html/` | Contains the Doxygen-generated documentation |
| `include/` | Contains the header files |
| `parameter_recipe/` | Contains the parameter recipes for convinience |
| `src/` | Contains the source files |
| `src/boys/` | Contains a precomputed file for the Boys function |
| `xyz/` | Contains the XYZ files (e.g., H2O.xyz) |
| `xyz/monatomic/` | Contains the XYZ files for monatomic molecules (e.g., H.xyz) |
| `CMakeLists.txt` | CMake configuration file |
| `Doxyfile` | Doxygen configuration file |
| `License.txt` | License file |
| `README.md` | Project overview and installation instructions |

### Build instructions
1. Copy the source code.
2. Create a build directory and configure the build using CMake:
``` bash
mkdir build && cd build
cmake ..
```
3. Build the software using the generated Makefile:
``` bash
make
```
4. Run the H2 molecule example:
``` bash
./HF_main -x ../xyz/H2.xyz -g ../basis/sto-3g.gbs -m RHF
```

### Usage
Usage
```
./HF_main [options]
```

Please see \ref parameters for [options].

Short options:

| Short option | Long option | Description |
| --- | --- | --- |
| `-m` | `--method` | Method (RHF, UHF, ROHF) |
| `-v` | `--verbose` | Verbose mode |
| `-p` | `--parameter_file` | Parameter recipe file |
| `-x` | `--xyzfilename` | XYZ file |
| `-g` | `--gbsfilename` | Gaussian basis set file |
| `-c` | `--charge` | Charge of the molecule |

#### Examples

##### Example 1: H2 molecule
``` bash
./HF_main -x ../xyz/H2.xyz -g ../basis/sto-3g.gbs -m RHF
```

##### Example 2: H2O molecule using a parameter recipe file
Command-line option "-p" specifies the parameter recipe file that contains pre-defined parameters for the calculation.

How to give the parameter recipe file:
``` bash
./HF_main -p ../parameter_recipe/RHF_OptimalDamping.txt -x ../xyz/H2.xyz -g ../basis/cc-pvdz.gbs
```
This command is equivalent to the following command:
``` bash
./HF_main --parameter_file ../parameter_recipe/RHF_OptimalDamping.txt --xyzfilename ../xyz/H2.xyz --gbsfilename ../basis/cc-pvdz.gbs
```



The contents of the parameter recipe file RHF_OptimalDamping.txt are as follows:
```
xyzfilename = ../xyz/H2O.xyz
gbsfilename = ../basis/sto-3g.gbs
method = RHF
convergence_method = OptimalDamping
```



\note
Parameters in the recipe file are overwritten by the command-line options.



## License 
[![BSD 3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-orange)](https://opensource.org/licenses/BSD-3-Clause)


GANSU: GPU Accelerated Numerical Simulation Utility

Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited All rights reserved.

This software is licensed under the BSD 3-Clause License.
You may obtain a copy of the license in the LICENSE file
located in the root directory of this source tree or at:
https://opensource.org/licenses/BSD-3-Clause
