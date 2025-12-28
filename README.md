# GANSU 

## Overview
GANSU (GPU Accelerated Numerical Simulation Utility) is an open-source quantum chemistry software designed for high-performance computations on modern computing architectures. This software aims to accelerate quantum chemistry simulations using advanced computational techniques such as GPU parallelization and efficient algorithms.

## Features
* Hartree-Fock Methods: Includes RHF, UHF, and ROHF implementations.
* Parallel computing: Accelerates almost all operations on the GPU, achieving true speedup through custom implementations from scratch.
* Flexible Input Options: Supports standard file formats such as XYZ and Gaussian basis set files.
* The numerical calculations in this software are performed using 64-bit double precision floating-point arithmetic.
* etc.


### Supported computations
* Hartree-Fock methods
    * Restricted Hartree-Fock (RHF) 
    * Unrestricted Hartree-Fock (UHF)
    * Restricted Open-Shell Hartree-Fock (ROHF)
    * RI approximation (Density Fitting) (RHF, UHF, ROHF)
    * Direct SCF (RHF) 
* Post-Hartree-Fock methods
    * Møller-Plesset Perturbation Theory (RI-RMP2)
* Initial Guess
    * Core Hamiltonian (RHF, UHF, ROHF) 
    * Generalized Wolfsberg-Helmholz (GWH) (RHF, UHF, ROHF)
    * Superposition of Atomic Densities (SAD) (RHF, UHF, ROHF)
    * Given density matrix (RHF, UHF, ROHF)
* Convergence algorithms
    * Damping (RHF, UHF, ROHF)
    * Optimal Damping (RHF, ROHF)
    * DIIS (RHF, UHF, ROHF)
* Molecular integrals
    * Overlap integrals
      * McMurchie-Davidson algorithm (s-, p-, d-, f-, g-, h-, and i-orbitals)
    * Kinetic energy and nuclear attraction integrals 
      * McMurchie-Davidson algorithm (s-, p-, d-, f-, g-, h-, and i-orbitals)
      * Obara-Saika algorithm (s-, p-, d-, and f-orbitals)
    * Electron repulsion integrals
      *  McMurchie-Davidson algorithm (s-, p-, d-, f-, g-, h-, and i-orbitals)
      *  Head-Godon-Pople algorithm (s- and p-orbitals)
      * Swartz Screening
    * Electron repulsion integrals for density fitting (RI approximation)
      * McMurchie-Davidson algorithm (s-, p-, d-, f-, and g-orbitals)
      * Head-Godon-Pople algorithm (s-, p-, and d-orbitals for basis functions, s-, p-, d-, f-, and g-orbitals for auxiliary basis functions)
    * Boys function
* Charge analysis
    * Mulliken population analysis (RHF, UHF, ROHF)
* Bond order analysis
    * Mayer bond order (RHF, UHF, ROHF)
    * Wiberg bond order (RHF, UHF, ROHF)
* Export
    * Export wave function information in the Molden format for visualization
        * Tested by [Avogadro](https://avogadro.cc/) and [Pegamoid](https://github.com/Jellby/Pegamoid) 
      ![Orbital](/doc/images/orbital.png)
      *Resulting molecular orbital of Benzene*
    
### Todo / Not Implemented yet
* Convergence algorithms
  * Optimal Damping (UHF)
  * EDIIS
  * ADIIS
* Initial Guess
  * Random
  * Load the precomputed coefficients/Fock matrix
* Post-Hartree-Fock methods
  * Configuration Interaction (CI)
  * Coupled Cluster (CC)
* Excited State Methods
  * Algebraic Diagrammatic Construction (ADC)
  * Equation-of-Motion Coupled Cluster (EOM-CC)
  * Time-Dependent Hartree-Fock (TDHF)
* Density Functional Theory (DFT)
* GPU implementation
  * Total spin (UHF)
* Charge analysis
    * Lowdin population analysis
    * Hirshfeld population analysis

## Installation

### Prerequisites
* Hardware
  * NVIDIA GPU with CUDA Compute Capability 8.0, 8.6, 9.0 or later
  * x86_64 / ARM architecture
* Software
  * C++ 17 or later
  * CMake 3.31 or later
  * NVIDIA CUDA Toolkit 12.9 or later
  * cuBLAS 12.9 or later
  * cuSOLVER 11.7 or later


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
├─ script/
├─ src/
│   └─ boys/
├─ test/
├─ xyz/
│   ├─ large_molecular/
│   └─ monatomic/
├─ CMakeLists.txt
├─ LICENSE
├─ Parameter.md
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
| `script/` | Script files |
| `src/` | Contains the source files |
| `src/boys/` | Contains a precomputed file for the Boys function |
| `test/` | Contains the test files |
| `xyz/` | Contains the XYZ files (e.g., H2O.xyz) |
| `xyz/large_molecular/` | Contains the XYZ files for large molecules (e.g., fullerene.xyz). RI approximation (density fitting) is neccesary for them. |
| `xyz/monatomic/` | Contains the XYZ files for monatomic molecules (e.g., H.xyz) |
| `CMakeLists.txt` | CMake configuration file |
| `LICENSE` | License file |
| `Parameter.md` | Parameter overview and description |
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
3. Build the software using the generated Makefile:
``` bash
make
```
4. Run the H2 molecule example:
``` bash
./HF_main -x ../xyz/H2.xyz -g ../basis/sto-3g.gbs -m RHF
```

> [!NOTE]
> To enable support for higher angular momentum orbitals in the RI approximation, uncomment the relevant lines in CMakeLists.txt. Be aware that doing so may result in a substantially longer compilation time.

### Usage
Usage
```
./HF_main [options]
```

Please see [Parameters](/Parameter.md) for options.

Short options:

| Short option | Long option | Description |
| --- | --- | --- |
| `-m` | `--method` | Method (RHF, UHF, ROHF) |
| `-v` | `--verbose` | Verbose mode |
| `-p` | `--parameter_file` | Parameter recipe file |
| `-x` | `--xyzfilename` | XYZ file |
| `-g` | `--gbsfilename` | Gaussian basis set file |
| `-ag` | `--auxiliary_gbsfilename` | Gaussian auxiliary basis set file |
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
./HF_main -p ../parameter_recipe/RHF_OptimalDamping.txt -x ../xyz/H2O.xyz -g ../basis/cc-pvdz.gbs
```
This command is equivalent to the following command:
``` bash
./HF_main --parameter_file ../parameter_recipe/RHF_OptimalDamping.txt --xyzfilename ../xyz/H2O.xyz --gbsfilename ../basis/cc-pvdz.gbs
```



The contents of the parameter recipe file RHF_OptimalDamping.txt are as follows:
```
xyzfilename = ../xyz/H2O.xyz
gbsfilename = ../basis/sto-3g.gbs
method = RHF
convergence_method = OptimalDamping
```
> [!NOTE]
> Parameters in the recipe file are overwritten by the command-line options.


##### Example 3: Fullerene (C60) molecule using RI approximation
If the molecule is large, it is recommended to use the RI approximation (density fitting) to reduce the memory usage. The auxiliary basis set file is required for the RI approximation.

``` bash
./HF_main -x ../xyz/large_molecular/fullerene.xyz -g ../basis/sto-3g.gbs --eri_method ri -ag ../auxiliary_basis/cc-pvdz-rifit.gbs
```


## License [![BSD 3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-orange)](https://opensource.org/licenses/BSD-3-Clause)


GANSU (GPU Accelerated Numerical Simulation Utility)

Copyright (c) 2025, Hiroshima University and Fujitsu Limited All rights reserved.

This software is licensed under the BSD 3-Clause License.
You may obtain a copy of the license in the LICENSE file
located in the root directory of this source tree or at:
https://opensource.org/licenses/BSD-3-Clause

## Publications
  1. Kanta Suzuki, Yasuaki Ito, Nobuya Yokogawa, Satoki Tsuji, Koji Nakano, Victor Parque and Akihiko Kasagi, GPU Acceleration of RI-RMP2 Correlation Energy Computation, in Proc. of International Symposium on Computing and Networking, pp. 174-180, Yamagata, Yamagata, November 2025.([DOI](https://doi.ieeecomputersociety.org/10.1109/CANDAR68384.2025.00031))
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
