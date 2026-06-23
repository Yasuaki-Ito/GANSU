# Source Code for

# GPU-Accelerated Full Configuration Interaction for 1.7 Trillion Determinants

This repository contains the source code accompanying the manuscript:

**"GPU-Accelerated Full Configuration Interaction for 1.7 Trillion Determinants"**

The implementation provides a distributed-memory Full Configuration Interaction (FCI) solver accelerated by NVIDIA GPUs using CUDA, cuBLAS, NCCL, and MPI.

---

# Software Requirements

The code has been tested with the following software versions:

| Software | Version       |
| -------- | ------------- |
| CUDA     | 12.9          |
| NCCL     | 2.29          |
| OpenMPI  | 4.1.x         |
| CMake    | 3.20 or later |
| Python   | 3.10.x        |

---

# Tested Hardware

| GPU         | Memory | CUDA | NCCL |
| ----------- | ------ | ---- | ---- |
| NVIDIA A100 | 80 GB  | 12.9 | 2.29 |
| NVIDIA A100 | 20 GB  | 13.1 | 2.29 |
| NVIDIA H200 | 141 GB | 12.9 | 2.28 |
| NVIDIA V100 | 16 GB  | 12.0 | 2.28 |

---

# Environment Setup

## NCCL

```bash
wget https://developer.download.nvidia.com/compute/redist/nccl/v2.29.3/nccl_2.29.3-1+cuda12.9_x86_64.txz

mkdir -p $HOME/local/nccl

tar -xvf nccl_2.29.3-1+cuda12.9_x86_64.txz \
    -C $HOME/local/nccl \
    --strip-components=1

export NCCL_HOME=$HOME/local/nccl
export CPATH=$NCCL_HOME/include:$CPATH
export LIBRARY_PATH=$NCCL_HOME/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$NCCL_HOME/lib:$LD_LIBRARY_PATH
```

## OpenMPI

```bash
wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.6.tar.gz

tar -zxvf openmpi-4.1.6.tar.gz

cd openmpi-4.1.6

mkdir -p $HOME/local/openmpi

./configure --prefix=$HOME/local/openmpi

make -j
make install

export PATH=$HOME/local/openmpi/bin:$PATH
export LD_LIBRARY_PATH=$HOME/local/openmpi/lib:$LD_LIBRARY_PATH
export MANPATH=$HOME/local/openmpi/share/man:$MANPATH
```

## Python Environment

```bash
python -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt
```

---

## Build

```bash
cd lib

sh build.sh
```

Successful compilation generates the shared library required by the distributed FCI solver.

---

# Running Calculations

## Example 1: Standard Execution

```bash
mpirun -n <nprocs> python fci.py --mol C2 --basis sto3g
```
where <nprocs> is the number of MPI processes (GPUs) used for the calculation.

A convenience script providing the same functionality is also included:
```bash
./mpifci.sh C2 sto3g
```

## Example 2: Calculation from FCIDUMP Integrals

```bash
./run_gpufci_from_integral.sh
```

The provided example uses the FCIDUMP integrals for the 2Fe2S system. 
Please update the path specified by the --fcidump option.

---

# Input Parameters

The main driver supports the following command-line arguments:

| Argument      | Description                                     | Default |
| ------------- | ----------------------------------------------- | ------- |
| `--mol`       | Molecule name                                   | None    |
| `--basis`     | Basis set                                       | `sto3g` |
| `--dist`      | Bond distance                                   | None    |
| `--max_cycle` | Maximum Davidson iterations                     | 100     |
| `--max_space` | Maximum Davidson subspace dimension             | 12      |
| `--filename`  | Output filename                                 | None    |
| `--incpu`     | Memory mode (`0=GPU`, `1=hybrid`)               | 0       |
| `--debugmode` | Debug level (`0=none`, `1=basic`, `2=detailed`) | 1       |
| `--chunksize` | Tile size used in σ = Hc evaluation             | 256     |

---

# Repository Contents

The repository includes the source code and example files used to run and validate the GPU-accelerated FCI calculations described in the manuscript.

Content	Location
Source code for the GPU-accelerated FCI implementation	Root directory
Example molecular input files	./share
Example FCIDUMP integral files	./share/sqd_data_repository/integrals
Example execution scripts	Root directory
Example output logs	./logs/examples

---

# Reproducibility

The example inputs provided in ./share and ./share/sqd_data_repository/integrals can be used to reproduce representative calculations and verify the functionality of the implementation.

Numerical results may exhibit minor variations across different GPU architectures, CUDA toolchains, MPI implementations, and NCCL versions. In all tested environments, the observed deviations were below 10⁻¹¹ Hartree.


---

# Contact

Email: [gao.hong@fujitsu.com](mailto:gao.hong@fujitsu.com)
