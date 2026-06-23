#######################################################

# Copyright Fujitsu Limited 2026. All rights reserved.

#######################################################

import os
import sys
import time
import json
import argparse
import ctypes
import cupy as cp

import numpy as np
import pyscf
from functools import reduce
from mpi4py import MPI
from pyscf import ao2mo
from pyscf.fci import cistring
from pyscf.tools import fcidump

# =============================================================================
# MPI initialization
# =============================================================================
comm   = MPI.COMM_WORLD
rank   = comm.Get_rank()
nprocs = comm.Get_size()

# =============================================================================
# Load shared library
# =============================================================================
libfci = ctypes.CDLL("./lib/build/libfci.so")

# =============================================================================
# Argument parser
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="GPU-accelerated FCI solver")
    parser.add_argument('--fcidump',   type=str, required=True,   help="Path to FCIDUMP file")
    parser.add_argument('--mol',       type=str, default=None,    help="Molecule name")
    parser.add_argument('--basis',     type=str, default='sto3g', help="Basis set")
    parser.add_argument('--dist',      type=str, default=None,    help="Bond distance")
    parser.add_argument('--max_cycle', type=int, default=100,     help="Max Davidson iterations")
    parser.add_argument('--max_space', type=int, default=12,      help="Max Davidson subspace size")
    parser.add_argument('--filename',  type=str, default=None,    help="Output JSON filename")
    parser.add_argument('--incpu',     type=int, default=0,       help="Memory mode: 0=GPU, 1=hybrid")
    parser.add_argument('--debugmode', type=int, default=1,       help="Debug level: 0=none, 1=basic (energy/iter), 2=detailed (time/iter)")
    parser.add_argument('--chunksize', type=int, default=256,     help="Chunk size for sigma=Hc")
    return parser.parse_args()

# =============================================================================
# Run Hartree-Fock
# =============================================================================
def run_hf(fcidump_path):
    mf         = fcidump.to_scf(fcidump_path)
    norbitals  = mf.get_hcore().shape[0]
    nelectrons = mf.mol.nelectron

    # Build initial density matrix
    dm = np.zeros((norbitals, norbitals))
    for p in range(nelectrons // 2):
        dm[p, p] = 2.0

    mf.kernel(dm)

    mol   = mf.mol
    mo    = mf.mo_coeff
    nelec = getattr(mf, 'nelec', mol.nelec)

    # Read ecore directly from FCIDUMP
    fd    = fcidump.read(fcidump_path)
    ecore = fd['ECORE']

    # Build h1e and eri in MO basis
    hcore   = mf.get_hcore()
    eri_ao  = mf._eri
    E_HF     = mf.e_tot
    h1e     = reduce(np.dot, (mo.conj().T, hcore, mo))
    eri     = ao2mo.kernel(eri_ao, mo)
    norb    = mo.shape[1]
    eri_new = ao2mo.restore(1, eri, norb)
    return h1e, eri_new, norb, nelec, ecore, E_HF

# =============================================================================
# Build Integral in MO basis
# =============================================================================
def build_integral(mf):
    mol     = mf.mol
    mo      = mf.mo_coeff
    nelec   = getattr(mf, 'nelec', mol.nelec)
    hcore   = mf.get_hcore()
    ecore   = mf.energy_nuc()
    eri_ao  = mf._eri

    h1e     = reduce(np.dot, (mo.conj().T, hcore, mo))
    eri     = ao2mo.kernel(eri_ao, mo)
    norb    = mo.shape[1]
    eri_new = ao2mo.restore(1, eri, norb)

    return h1e, eri_new, norb, nelec, ecore

# =============================================================================
# Save result to JSON
# =============================================================================
def save_result(args, e_value, na, norb, nelec_t, nprocs, ecore, E_HF, fci_time):
    prop = cp.cuda.runtime.getDeviceProperties(0)
    result = {
        "nelec":          int(nelec_t),
        "norb":           int(norb),
        "na":             int(na),
        "nstr":           int(na * na),
        "nprocs":         int(nprocs),
        "use_cpu_memory": bool(args.incpu),
        "chunksize":      int(args.chunksize),
        "max_space":      int(args.max_space),
        "device":         str(prop['name'].decode()),
        "ecore":          float(ecore),
        "e_HF":           float(E_HF),
        "e_FCI":          float(e_value[0]),
        "fci_time":       float(fci_time),
    }
    with open(args.filename, "w") as f:
        json.dump(result, f, indent=4)
    print(f"Result saved to {args.filename}")

# =============================================================================
# Run FCI via libfci
# =============================================================================
def run_fci(h1e, eri_new, norb, nelec, ecore, E_HF, args):
    neleca, nelecb = nelec
    assert neleca == nelecb, f"neleca={neleca} != nelecb={nelecb}: not implemented"
    nelec_t = neleca + nelecb
    start_oth = time.time()
    occslst      = cistring._gen_occslst(range(norb), neleca)
    na           = cistring.num_strings(norb, neleca)
    tol          = 1e-12

    # Initialize energy array
    e_value     = np.zeros((1,))

    if rank == 0 and args.debugmode > 0:
        print("=" * 60)
        print(f"Molecule: {args.mol}")
        print(f"Basis: {args.basis}")
        print(f"MPI Ranks: {nprocs}")
        print("=" * 60)
        #print("\n Starting FCI solver... \n")
        #print("----------------------------------------------")

        print(
            f"nelec: {nelec}, norb: {norb}, na: {na}, ndet: {na*na}\n"
            f"max_space: {args.max_space}, max_cycle: {args.max_cycle}, "
            f"in_cpu: {args.incpu}, chunksize:{args.chunksize}, ecore: {ecore}"
        )
        #print("----------------------------------------------")
    start_fci = time.time()
    # Define ctypes function signature
    libfci.fci_result.argtypes = [
        ctypes.c_void_p,  # h1e
        ctypes.c_void_p,  # eri
        ctypes.c_void_p,  # e_value
        ctypes.c_void_p,  # occslst
        ctypes.c_int64,   # na
        ctypes.c_int64,   # norb
        ctypes.c_int64,   # neleca
        ctypes.c_int,     # max_space
        ctypes.c_int,     # max_cycle
        ctypes.c_int,     # in_cpu
        ctypes.c_int,     # chunk_size
        ctypes.c_int,     # debug_mode
        ctypes.c_double,  # tol
        ctypes.c_double,  # ecore
    ]

    # Call FCI solver
    libfci.fci_result(
        h1e.ctypes.data_as(ctypes.c_void_p),
        eri_new.ctypes.data_as(ctypes.c_void_p),
        e_value.ctypes.data_as(ctypes.c_void_p),
        occslst.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int64(na),
        ctypes.c_int64(norb),
        ctypes.c_int64(neleca),
        ctypes.c_int(args.max_space),
        ctypes.c_int(args.max_cycle),
        ctypes.c_int(args.incpu),
        ctypes.c_int(args.chunksize),
        ctypes.c_int(args.debugmode),
        ctypes.c_double(tol),
        ctypes.c_double(ecore),
    )
    fci_time = time.time() - start_fci
    
    # Save and print results
    if rank == 0 and args.filename != None:
        save_result(args, e_value, na, norb, nelec_t, nprocs, ecore, E_HF, fci_time)
    if rank == 0 and args.debugmode > 1:
        #print(f"\n FCI energy: {e_value[0]:.12f} Ha")
        #print(f" FCI correlation energy: {e_value[0] - E_HF:.12f} Ha")
        print(f" fci_result time: {fci_time:.3f} s, others: {start_fci - start_oth:.3f} s\n")
    
    return e_value, na, nelec_t



# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    args      = parse_args()
    start_all = time.time()

    # Run Hartree-Fock
    start_hf = time.time()
    h1e, eri_new, norb, nelec, ecore, E_HF = run_hf(args.fcidump)
    
    # Integral transformation in MO basis
    end_integral = time.time()
    if rank == 0 and args.debugmode > 0:
        print(f"pre_time: {end_integral - start_hf:.3f} s \n")
    # Run FCI
    e_value, na, nelec_t   = run_fci(h1e, eri_new, norb, nelec, ecore, E_HF, args)

    
