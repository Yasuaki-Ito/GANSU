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
# ============================================================
# ctypes interface
# ============================================================

libfci.fci_result.argtypes = [
    ctypes.c_void_p,   # h1e
    ctypes.c_void_p,   # eri
    ctypes.c_void_p,   # e_value
    ctypes.c_void_p,   # occslst

    ctypes.c_int32,    # na
    ctypes.c_int32,    # norb
    ctypes.c_int32,    # neleca

    ctypes.c_int,      # max_space
    ctypes.c_int,      # max_cycle
    ctypes.c_int,      # in_cpu
    ctypes.c_int,      # chunk_size
    ctypes.c_int,      # debug_mode

    ctypes.c_double,   # tol
    ctypes.c_double,   # ecore
]

libfci.fci_result.restype = None


libfci.fci_result_unequal_elec.argtypes = [
    ctypes.c_void_p,   # h1e
    ctypes.c_void_p,   # eri
    ctypes.c_void_p,   # e_value
    ctypes.c_void_p,   # occslsta
    ctypes.c_void_p,   # occslstb

    ctypes.c_int32,    # na
    ctypes.c_int32,    # nb
    ctypes.c_int32,    # norb
    ctypes.c_int32,    # neleca
    ctypes.c_int32,    # nelecb

    ctypes.c_int,      # max_space
    ctypes.c_int,      # max_cycle
    ctypes.c_int,      # in_cpu
    ctypes.c_int,      # chunk_size
    ctypes.c_int,      # debug_mode

    ctypes.c_double,   # tol
    ctypes.c_double,   # ecore
]

libfci.fci_result_unequal_elec.restype = None

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
    parser.add_argument('--ms2',       type=int, default=0,       help="Spin projection ms2")
    return parser.parse_args()

# =============================================================================
# Run Hartree-Fock
# =============================================================================
def run_hf(fcidump_path, ms2=None):

    # ============================================================
    # Keep the original SCF procedure unchanged
    # ============================================================
    mf = fcidump.to_scf(fcidump_path)

    norb = mf.get_hcore().shape[0]
    nelec_t = mf.mol.nelectron

    # Original initial density
    dm = np.zeros((norb, norb))

    for p in range(nelec_t // 2):
        dm[p, p] = 2.0

    if rank == 0:
        print(f"Initial density trace = {np.trace(dm):.1f}")
        print(f"Expected electrons    = {nelec_t}")

    # Original SCF
    mf.kernel(dm)

    # IMPORTANT:
    # Do not abort here yet.
    if not mf.converged:
        if rank == 0:
            print(
                "WARNING: SCF did not converge. "
                "Continuing with the current orbitals "
                "to preserve the original workflow."
            )

    # ============================================================
    # MO transformation
    # ============================================================
    mo = mf.mo_coeff

    h1e = reduce(
        np.dot,
        (mo.conj().T, mf.get_hcore(), mo)
    )

    eri = ao2mo.restore(
        1,
        ao2mo.kernel(mf._eri, mo),
        norb
    )

    # ============================================================
    # Determine Nalpha/Nbeta only for the FCI sector
    # ============================================================
    if ms2 is None:
        fd = fcidump.read(fcidump_path)
        ms2 = int(fd.get("MS2", 0))

    if (nelec_t + ms2) % 2 != 0:
        raise ValueError(
            f"Inconsistent NELEC/MS2: "
            f"NELEC={nelec_t}, MS2={ms2}"
        )

    neleca = (nelec_t + ms2) // 2
    nelecb = (nelec_t - ms2) // 2

    if rank == 0:
        print(
            f"FCIDUMP: NELEC={nelec_t}, MS2={ms2}, "
            f"(Nalpha,Nbeta)=({neleca},{nelecb})"
        )

    # Preserve old convention
    ecore = 0.0

    return (
        h1e,
        eri,
        norb,
        (neleca, nelecb),
        ecore,
        mf.e_tot,
    )

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
def save_result(
    args,
    e_value,
    na,
    nb,
    norb,
    neleca,
    nelecb,
    nprocs,
    ecore,
    E_HF,
    fci_time,
):
    prop = cp.cuda.runtime.getDeviceProperties(0)

    result = {
        "nelec":          int(neleca + nelecb),
        "neleca":         int(neleca),
        "nelecb":         int(nelecb),
        "norb":           int(norb),
        "na":             int(na),
        "nb":             int(nb),
        "nstr":           int(na * nb),
        "nprocs":         int(nprocs),
        "use_cpu_memory": bool(args.incpu),
        "chunksize":      int(args.chunksize),
        "max_space":      int(args.max_space),
        "device":         str(prop["name"].decode()),
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
    print("start run_fci")
    neleca, nelecb = nelec
    unequal = (neleca != nelecb)

    nelec_t = neleca + nelecb
    tol = 1e-12

    start_oth = time.time()

    # ---------------------------------------------------------
    # Generate determinant strings
    # ---------------------------------------------------------
    occslsta = cistring._gen_occslst(range(norb), neleca)
    na = cistring.num_strings(norb, neleca)

    if unequal:
        occslstb = cistring._gen_occslst(range(norb), nelecb)
        nb = cistring.num_strings(norb, nelecb)
    else:
        occslstb = None
        nb = na

    ndet = na * nb

    # ---------------------------------------------------------
    # Output array
    # ---------------------------------------------------------
    e_value = np.zeros(1, dtype=np.float64)

    # ---------------------------------------------------------
    # Information
    # ---------------------------------------------------------
    if rank == 0 and args.debugmode > 0:
        print("=" * 60)
        print(f"Molecule: {args.mol}")
        print(f"Basis: {args.basis}")
        print(f"MPI Ranks: {nprocs}")
        print("=" * 60)

        print(
            f"nelec: {nelec}, norb: {norb}, "
            f"na: {na}, nb: {nb}, ndet: {ndet}\n"
            f"max_space: {args.max_space}, "
            f"max_cycle: {args.max_cycle}, "
            f"in_cpu: {args.incpu}, "
            f"chunksize: {args.chunksize}, "
            f"ecore: {ecore}"
        )

    # ---------------------------------------------------------
    # Call C/CUDA FCI
    # ---------------------------------------------------------
    start_fci = time.time()

    if unequal:

        libfci.fci_result_unequal_elec(
            h1e.ctypes.data_as(ctypes.c_void_p),
            eri_new.ctypes.data_as(ctypes.c_void_p),
            e_value.ctypes.data_as(ctypes.c_void_p),
            occslsta.ctypes.data_as(ctypes.c_void_p),
            occslstb.ctypes.data_as(ctypes.c_void_p),

            ctypes.c_int32(na),
            ctypes.c_int32(nb),
            ctypes.c_int32(norb),
            ctypes.c_int32(neleca),
            ctypes.c_int32(nelecb),

            ctypes.c_int(args.max_space),
            ctypes.c_int(args.max_cycle),
            ctypes.c_int(args.incpu),
            ctypes.c_int(args.chunksize),
            ctypes.c_int(args.debugmode),

            ctypes.c_double(tol),
            ctypes.c_double(ecore),
        )

    else:

        libfci.fci_result(
            h1e.ctypes.data_as(ctypes.c_void_p),
            eri_new.ctypes.data_as(ctypes.c_void_p),
            e_value.ctypes.data_as(ctypes.c_void_p),
            occslsta.ctypes.data_as(ctypes.c_void_p),

            ctypes.c_int32(na),
            ctypes.c_int32(norb),
            ctypes.c_int32(neleca),

            ctypes.c_int(args.max_space),
            ctypes.c_int(args.max_cycle),
            ctypes.c_int(args.incpu),
            ctypes.c_int(args.chunksize),
            ctypes.c_int(args.debugmode),

            ctypes.c_double(tol),
            ctypes.c_double(ecore),
        )

    fci_time = time.time() - start_fci

    # ---------------------------------------------------------
    # Timing
    # ---------------------------------------------------------
    if rank == 0 and args.debugmode > 1:
        print(
            f"fci_result time: {fci_time:.3f} s, "
            f"others: {start_fci - start_oth:.3f} s\n"
        )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------
    if rank == 0 and args.filename is not None:
        save_result(
            args,
            e_value,
            na,
            nb,
            norb,
            neleca,
            nelecb,
            nprocs,
            ecore,
            E_HF,
            fci_time,
        )

    return e_value, na, nb, nelec_t

def read_fcidump_direct(fcidump_path, ms2=None):
    """
    Read FCIDUMP integrals directly without performing SCF.

    Returns
    -------
    h1e : ndarray, shape (norb, norb)
        One-electron integrals.
    eri : ndarray, shape (norb, norb, norb, norb)
        Two-electron integrals in 4-index form.
    norb : int
        Number of spatial orbitals.
    nelec : tuple
        (N_alpha, N_beta)
    ecore : float
        Constant/core energy from FCIDUMP.
    """

    # ----------------------------------------------------------
    # Read FCIDUMP
    # ----------------------------------------------------------
    fd = fcidump.read(fcidump_path)

    norb = int(fd["NORB"])
    nelec_t = int(fd["NELEC"])
   
    print("line412_ms2:", ms2)
    if ms2 is None:
        ms2 = int(fd.get("MS2", 0))
    else:
        ms2 = int(ms2)

    # ----------------------------------------------------------
    # Determine N_alpha and N_beta
    #
    # N_alpha + N_beta = NELEC
    # N_alpha - N_beta = MS2
    # ----------------------------------------------------------
    if (nelec_t + ms2) % 2 != 0:
        raise ValueError(
            f"Inconsistent NELEC/MS2: "
            f"NELEC={nelec_t}, MS2={ms2}"
        )

    neleca = (nelec_t + ms2) // 2
    nelecb = (nelec_t - ms2) // 2

    if neleca < 0 or nelecb < 0:
        raise ValueError(
            f"Invalid electron numbers: "
            f"Nalpha={neleca}, Nbeta={nelecb}"
        )

    # ----------------------------------------------------------
    # Integrals
    # ----------------------------------------------------------
    h1e = np.asarray(fd["H1"], dtype=np.float64)

    eri = ao2mo.restore(
        1,
        fd["H2"],
        norb
    )

    eri = np.asarray(eri, dtype=np.float64)

    # ctypes/CUDA prefers contiguous arrays
    h1e = np.ascontiguousarray(h1e)
    eri = np.ascontiguousarray(eri)

    # ----------------------------------------------------------
    # Constant energy
    # ----------------------------------------------------------
    ecore = float(fd.get("ECORE", 0.0))

    # ----------------------------------------------------------
    # Information
    # ----------------------------------------------------------
    if rank == 0:
        na = cistring.num_strings(norb, neleca)
        nb = cistring.num_strings(norb, nelecb)

        print("=" * 60)
        print("Direct FCIDUMP input (no SCF)")
        print("=" * 60)
        print(f"NORB           = {norb}")
        print(f"NELEC          = {nelec_t}")
        print(f"MS2            = {ms2}")
        print(f"M_S            = {ms2 / 2.0}")
        print(f"Nalpha         = {neleca}")
        print(f"Nbeta          = {nelecb}")
        print(f"n_alpha        = {na}")
        print(f"n_beta         = {nb}")
        print(f"Ndet           = {na * nb}")
        print(f"ECORE          = {ecore:.15f}")
        print(f"H1 shape       = {h1e.shape}")
        print(f"H2 shape       = {eri.shape}")
        print("=" * 60)

    return (
        h1e,
        eri,
        norb,
        (neleca, nelecb),
        ecore,
    )


# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    args      = parse_args()
    start_all = time.time()

    # Run Hartree-Fock
    start_hf = time.time()
    print("ms2:", args.ms2)
    h1e, eri_new, norb, nelec, ecore = read_fcidump_direct(args.fcidump, ms2=args.ms2)
    E_HF = np.nan
    # Integral transformation in MO basis
    end_integral = time.time()
    if rank == 0 and args.debugmode > 0:
        print(f"pre_time: {end_integral - start_hf:.3f} s \n")
    # Run FCI
    e_value, na, nb, nelec_t   = run_fci(h1e, eri_new, norb, nelec, ecore, E_HF, args)
    
