/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 */

/**
 * @file gansu_api.h
 * @brief C API for GANSU — stable ABI for Python/GUI/external bindings.
 *
 * All functions are extern "C" with opaque handles. CUDA dependencies are
 * hidden behind the implementation. Thread safety: NOT thread-safe (GPU
 * state is global). Call gansu_init() once before any other function.
 *
 * Usage:
 *   gansu_init(0);  // 0 = auto-detect GPU, 1 = force CPU
 *   gansu_handle_t h = gansu_create();
 *   gansu_set_xyz(h, "../xyz/H2O.xyz");
 *   gansu_set_basis(h, "../basis/cc-pvdz.gbs");
 *   gansu_set_method(h, "RHF");
 *   gansu_set_post_hf(h, "ccsd");
 *   int err = gansu_run(h);
 *   double e = gansu_get_total_energy(h);
 *   gansu_destroy(h);
 *   gansu_finalize();
 */

#ifndef GANSU_API_H
#define GANSU_API_H

#ifdef __cplusplus
extern "C" {
#endif

/** Opaque handle to a GANSU calculation context. */
typedef void* gansu_handle_t;

/* ---- Lifecycle ---- */

/** Initialize GANSU runtime. Call once before any other function.
 *  @param force_cpu  If nonzero, disable GPU and use CPU backend. */
void gansu_init(int force_cpu);

/** Finalize GANSU runtime. */
void gansu_finalize(void);

/** Create a new calculation context. */
gansu_handle_t gansu_create(void);

/** Destroy a calculation context and free all resources. */
void gansu_destroy(gansu_handle_t h);

/* ---- Configuration ---- */

/** Set a string parameter. Common keys:
 *    "xyzfilename", "gbsfilename", "method" (RHF/UHF/ROHF),
 *    "post_hf_method" (none/mp2/mp3/ccsd/ccsd_t/ccsd_density/fci/cis/adc2/eom_ccsd/...),
 *    "run_type" (energy/gradient/optimize/hessian),
 *    "eri_method" (stored/RI/direct/...),
 *    "initial_guess" (core/gwh/sad/minao),
 *    "convergence_method" (DIIS/SOSCF/...),
 *    "n_excited_states", "optimizer", etc.
 *  @return 0 on success, nonzero on error. */
int gansu_set(gansu_handle_t h, const char* key, const char* value);

/** Convenience: set xyz file path. */
int gansu_set_xyz(gansu_handle_t h, const char* path);

/** Convenience: set basis file path. */
int gansu_set_basis(gansu_handle_t h, const char* path);

/** Convenience: set HF method (RHF, UHF, ROHF). */
int gansu_set_method(gansu_handle_t h, const char* method);

/** Convenience: set post-HF method. */
int gansu_set_post_hf(gansu_handle_t h, const char* post_hf);

/* ---- Execution ---- */

/** Run the calculation (SCF + post-HF if configured).
 *  @return 0 on success, nonzero on error. */
int gansu_run(gansu_handle_t h);

/* ---- Results ---- */

/** Get HF total energy (electronic + nuclear repulsion) in Hartree. */
double gansu_get_total_energy(gansu_handle_t h);

/** Get post-HF correlation energy in Hartree (0 if no post-HF). */
double gansu_get_post_hf_energy(gansu_handle_t h);

/** Get nuclear repulsion energy in Hartree. */
double gansu_get_nuclear_repulsion_energy(gansu_handle_t h);

/** Get number of basis functions. */
int gansu_get_num_basis(gansu_handle_t h);

/** Get number of electrons. */
int gansu_get_num_electrons(gansu_handle_t h);

/** Get number of atoms. */
int gansu_get_num_atoms(gansu_handle_t h);

/** Get orbital energies. Writes to buffer (size >= num_basis).
 *  @return num_basis, or -1 on error. */
int gansu_get_orbital_energies(gansu_handle_t h, double* buf, int buf_size);

/** Get MO coefficient matrix (row-major, nao x nao).
 *  @return nao*nao elements written, or -1 on error. */
int gansu_get_mo_coefficients(gansu_handle_t h, double* buf, int buf_size);

/** Get CCSD 1-RDM in MO basis (nao x nao, row-major). Only valid after ccsd_density.
 *  @return nao*nao elements written, or -1 on error. */
int gansu_get_ccsd_1rdm_mo(gansu_handle_t h, double* buf, int buf_size);

/** Get excited state report string. Returns pointer to internal buffer (valid until destroy). */
const char* gansu_get_excited_state_report(gansu_handle_t h);

/* ---- Atom coordinates ---- */

/** Get atomic number of atom i (0-indexed). Returns 0 on error. */
int gansu_get_atomic_number(gansu_handle_t h, int i);

/** Get atom coordinates (x, y, z in Bohr) for atom i. Returns 0 on success. */
int gansu_get_atom_coords(gansu_handle_t h, int i, double* x, double* y, double* z);

/* ---- Analysis ---- */

/** Compute Mulliken charges. Writes to buf (size >= num_atoms).
 *  @return num_atoms, or -1 on error. */
int gansu_get_mulliken_charges(gansu_handle_t h, double* buf, int buf_size);

/** Compute Mayer bond order matrix (num_atoms x num_atoms, row-major).
 *  @return num_atoms*num_atoms, or -1 on error. */
int gansu_get_mayer_bond_order(gansu_handle_t h, double* buf, int buf_size);

/** Compute Wiberg bond order matrix (num_atoms x num_atoms, row-major).
 *  @return num_atoms*num_atoms, or -1 on error. */
int gansu_get_wiberg_bond_order(gansu_handle_t h, double* buf, int buf_size);

/** Get density matrix in AO basis (nao x nao, row-major).
 *  @return nao*nao, or -1 on error. */
int gansu_get_density_matrix(gansu_handle_t h, double* buf, int buf_size);

/** Get overlap matrix (nao x nao, row-major).
 *  @return nao*nao, or -1 on error. */
int gansu_get_overlap_matrix(gansu_handle_t h, double* buf, int buf_size);

/* ---- Progress callback ---- */

/**
 * @brief Progress callback type.
 *
 * Called during iterative procedures (SCF, CCSD, Davidson, optimization, etc.)
 * to report current progress.
 *
 * @param stage     Identifier: "scf", "ccsd", "ccsd_lambda", "davidson",
 *                  "optimize", or method-specific name.
 * @param iter      Current iteration number (0-indexed).
 * @param n_values  Number of doubles in the values array.
 * @param values    Progress data (meaning depends on stage):
 *                    scf:          [total_energy, delta_e]
 *                    ccsd:         [correlation_energy, delta_e]
 *                    ccsd_lambda:  [residual_norm]
 *                    davidson:     [eigenvalue_0, ..., eigenvalue_k, max_residual]
 *                    optimize:     [energy, max_gradient, rms_gradient]
 * @param user_data User pointer passed to gansu_set_progress_callback.
 */
typedef void (*gansu_progress_fn)(const char* stage, int iter, int n_values,
                                  const double* values, void* user_data);

/** Set progress callback. Pass NULL to disable.
 *  Callback is invoked from the thread that calls gansu_run(). */
void gansu_set_progress_callback(gansu_handle_t h, gansu_progress_fn fn, void* user_data);

#ifdef __cplusplus
}
#endif

#endif /* GANSU_API_H */
