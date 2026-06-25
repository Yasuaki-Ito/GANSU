# GPU-Accelerated Full Configuration Interaction for 1.7 Trillion Determinants

This directory contains the implementation accompanying the manuscript entitled
"GPU-Accelerated Full Configuration Interaction for 1.7 Trillion Determinants".

The implementation provides a distributed-memory Full Configuration Interaction (FCI) solver accelerated by NVIDIA GPUs using CUDA, cuBLAS, NCCL, and MPI.

---

# Software Requirements

The implementation has been tested with the following software versions:

| Software | Version       |
| -------- | ------------- |
| CUDA     | 12.9          |
| NCCL     | 2.29          |
| OpenMPI  | 4.1.x         |
| CMake    | 3.20 or later |
| Python   | 3.10.x        |

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

Please update the environment variables in `lib/build.sh` to match your local environment before building.
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
where \<nprocs\> is the number of MPI processes (GPUs) used for the calculation.

A convenience script providing the same functionality is also included:
```bash
./run_gpufci.sh C2 sto3g
```

## Example 2: Calculation from FCIDUMP Integrals

```bash
./run_gpufci_from_fcidump.sh <molecule> <basis>
```

Please update the path specified by the --fcidump option.

The example script expects FCIDUMP integral files as input.
Representative FCIDUMP integral files used in this work were obtained from the SQD data repository:

https://github.com/jrm874/sqd_data_repository/tree/main/integrals

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

## Directory Structure

| Directory/File | Description |
|----------------|-------------|
| `README.md ` | This document | 
| `geometry/` | Molecular geometries used in this work |
| `lib/` | CUDA/C++ implementation |
| `results/` | Output directory for benchmark data and user calculations |
| `results/calculated_energy/` | Reference energies reported in the manuscript |
| `common.py` | Utility functions for molecular geometry processing and common routines |
| `fci.py` | Main driver for distributed GPU-accelerated FCI calculations |
| `fci_from_fcidump.py` | Driver for FCI calculations from FCIDUMP integral files |
| `requirements.txt` | Python package requirements |
| `run_gpufci.sh` | Example script for running FCI calculations from molecular geometries |
| `run_gpufci_from_fcidump.sh` | Example script for running FCI calculations from FCIDUMP integral files |
---

