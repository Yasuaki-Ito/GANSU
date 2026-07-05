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

/** Get number of frozen core orbitals (0 if no frozen core). */
int gansu_get_num_frozen_core(gansu_handle_t h);

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

/** Get excited-state data as arrays (valid after a CIS/ADC/EOM/STEOM run).
 *  Writes up to n_max excitation energies (Hartree) into energies_out and the
 *  matching oscillator strengths (dimensionless) into osc_out. Either buffer may
 *  be NULL to skip it. Energies are relative to the ground state.
 *  @return number of states written (min(n_max, available)), or -1 on error. */
int gansu_get_excited_states(gansu_handle_t h, double* energies_out,
                             double* osc_out, int n_max);

/* ---- Derivatives and molecular properties ---- */

/** Get the analytic energy gradient (nuclear forces), computed on demand from
 *  the converged wavefunction. Writes 3*num_atoms values (dE/dx, dE/dy, dE/dz
 *  per atom, Hartree/Bohr) in atom-major row-major order.
 *  @return 3*num_atoms, or -1 on error, or -2 if unavailable for this method. */
int gansu_get_energy_gradient(gansu_handle_t h, double* buf, int len);

/** Get the analytic Hessian d²E/dR_i dR_j (Hartree/Bohr²), 3N x 3N row-major,
 *  computed on demand.
 *  @return (3*num_atoms)^2, or -1 on error, or -2 if unavailable for this method. */
int gansu_get_hessian(gansu_handle_t h, double* buf, int len);

/** Get harmonic vibrational frequencies (cm⁻¹), computed on demand (builds the
 *  Hessian, mass-weights, projects out translations/rotations, diagonalizes).
 *  Imaginary modes are returned as negative values. The number of frequencies is
 *  3N minus the projected-out modes (5 for linear, 6 for non-linear molecules).
 *  @param len size of buf; must be >= 3*num_atoms to be safe.
 *  @return number of frequencies written, or -1 on error, or -2 if unavailable. */
int gansu_get_frequencies(gansu_handle_t h, double* buf, int len);

/** Get the ground-state SCF dipole moment in atomic units (e·Bohr). Writes 3
 *  doubles (mu_x, mu_y, mu_z) into xyz. Multiply by 2.5417464157 for Debye.
 *  Closed-shell RHF only.
 *  @return 0 on success, -1 on error, -3 if not RHF. */
int gansu_get_dipole(gansu_handle_t h, double* xyz);

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

/** Set initial density matrix for next run (enables density reuse for PES).
 *  Pass NULL to clear. buf is nao*nao doubles, row-major. */
int gansu_set_initial_density(gansu_handle_t h, const double* buf, int buf_size);

/* ---- SCF-free energy-functional evaluation (FMQA interface) ----
 *
 * These functions evaluate the RHF energy functional
 *   E = sum_pq P_pq h_pq + 1/2 sum_pq P_pq G_pq(P) + E_nn,   G = J - 1/2 K
 * for a caller-supplied density / MO coefficients WITHOUT running SCF.
 *
 * Prerequisites: xyz and basis set on the handle. gansu_run() is NOT
 * required — on the first call the integrals (core Hamiltonian, overlap,
 * ERI) are lazily prepared once and reused across subsequent calls, so
 * repeated evaluation (10^2..10^4 calls) only costs one Fock build each.
 * Closed-shell RHF only (method must be RHF; even electron count).
 */

/** Prepare integrals for SCF-free evaluation without running SCF.
 *  Idempotent; called implicitly by the functions below. After this,
 *  gansu_get_num_basis / gansu_get_num_electrons / gansu_get_overlap_matrix /
 *  gansu_get_hcore / gansu_get_eri are usable without gansu_run().
 *  @return 0 on success, <0 on error. */
int gansu_prepare(gansu_handle_t h);

/** Evaluate the RHF energy functional from an AO density matrix P
 *  (n x n, row-major, RHF convention Tr(PS) = n_electrons). No SCF.
 *  The density is used as given — no normalization, no checks.
 *  Note: overwrites the handle's working density/Fock matrices.
 *  @return 0 on success; -1 on dimension mismatch / error; -3 if not RHF.
 *  On success *energy_out = total energy in Hartree (E_nn included). */
int gansu_energy_from_density(gansu_handle_t h,
                              const double* density, int n,
                              double* energy_out);

/** Evaluate the RHF energy functional from occupied MO coefficients C_occ
 *  (n x nocc, row-major; column j = j-th occupied orbital). Builds
 *  P = 2 C_occ C_occ^T and evaluates as gansu_energy_from_density.
 *  Orthonormality (C^T S C = I) is the CALLER's responsibility — the
 *  coefficients are trusted as-is (no checks, for high call rates).
 *  @return 0 on success; -1 on dimension mismatch / error; -3 if not RHF. */
int gansu_energy_from_mo(gansu_handle_t h,
                         const double* c_occ, int n, int nocc,
                         double* energy_out);

/** Batched variant: batch sets of occupied MO coefficients, contiguous
 *  (set b starts at c_occ_batch + (size_t)b*n*nocc). energies_out has
 *  length batch. Integrals are prepared once; one Fock build per point.
 *  @return 0 on success (all points evaluated), <0 on first error. */
int gansu_energy_from_mo_batch(gansu_handle_t h,
                               const double* c_occ_batch,
                               int batch, int n, int nocc,
                               double* energies_out);

/** Get core Hamiltonian h = T + V_ne (n x n, row-major).
 *  Usable after gansu_prepare() or gansu_run().
 *  @return n*n elements written, or -1 on error (len < n*n, not prepared). */
int gansu_get_hcore(gansu_handle_t h, double* buf, int len);

/** Get the full AO ERI tensor (pq|rs), chemists' notation, row-major
 *  n^4 layout: buf[((p*n + q)*n + r)*n + s]. Small-scale verification use.
 *  Only available for eri_method=stored.
 *  @return n^4 elements written; -1 on error (len too small, not prepared);
 *          -2 if refused (non-stored ERI method, or n^4 exceeds INT_MAX). */
int gansu_get_eri(gansu_handle_t h, double* buf, int len);

/* ---- SCF-free UHF evaluation (FMQA broken-symmetry interface) ----
 *
 * UHF counterparts of the RHF functions above, for FMQA global search of
 * the HF energy landscape (broken-symmetry / multiple-solution enumeration).
 * The evaluated functional (chemists' J, K, Pt = Pa + Pb) is
 *   E = sum_pq Pt_pq h_pq + 1/2 sum_pq Pt_pq J[Pt]_pq
 *       - 1/2 ( sum_pq Pa_pq K[Pa]_pq + sum_pq Pb_pq K[Pb]_pq ) + E_nn.
 * The handle's method must be UHF (set "method"="UHF"); functions return -3
 * otherwise. Alpha/beta occupations (na, nb) are explicit arguments, so
 * closed-shell broken-symmetry (na == nb, Ca != Cb) and open-shell radicals
 * (na != nb) share one API independent of the handle's charge/multiplicity.
 * Consistency: passing Ca == Cb == C_RHF (occupied columns) reproduces the
 * RHF total energy to |dE| < 1e-10.
 */

/** Evaluate the UHF energy functional from occupied alpha/beta MO coefficients
 *  c_alpha_occ (n x na) and c_beta_occ (n x nb), row-major (column j = j-th
 *  occupied orbital). Builds Pa = Ca_occ Ca_occ^T, Pb = Cb_occ Cb_occ^T
 *  (single occupancy per spin orbital), one Fock build, no SCF. Orthonormality
 *  is the CALLER's responsibility (no checks, for high call rates).
 *  @return 0 on success; -1 on dimension mismatch / error; -3 if not UHF.
 *  On success *energy_out = total energy in Hartree (E_nn included). */
int gansu_energy_from_mo_uhf(gansu_handle_t h,
                             const double* c_alpha_occ,
                             const double* c_beta_occ,
                             int n, int na, int nb,
                             double* energy_out);

/** Evaluate the UHF energy functional from AO density matrices Pa, Pb
 *  (each n x n, row-major). No SCF; densities are used as given.
 *  Overwrites the handle's working density/Fock matrices.
 *  @return 0 on success; -1 on dimension mismatch / error; -3 if not UHF. */
int gansu_energy_from_density_uhf(gansu_handle_t h,
                                  const double* Pa, const double* Pb,
                                  int n, double* energy_out);

/** Batched UHF evaluation: `batch` sets of (Ca, Cb), contiguous (set b starts
 *  at ca_batch + (size_t)b*n*na and cb_batch + (size_t)b*n*nb). energies_out
 *  has length batch. Integrals are prepared once; one Fock build per point.
 *  @return 0 on success (all points evaluated), <0 on first error. */
int gansu_energy_from_mo_uhf_batch(gansu_handle_t h,
                                   const double* ca_batch,
                                   const double* cb_batch,
                                   int batch, int n, int na, int nb,
                                   double* energies_out);

/** Run a UHF-SCF to convergence starting from the given occupied MOs, i.e.
 *  the "polish" step that turns an FMQA basin into a true stationary point.
 *  Builds Pa/Pb (single occupancy) as the initial density and runs UHF-SCF.
 *  No extra symmetry breaking is applied, so a symmetric initial guess
 *  (Ca == Cb) stays at the RHF stationary point while a spin-alternating guess
 *  relaxes to the broken-symmetry solution. If pa_out/pb_out are non-NULL
 *  (each n*n, row-major) the converged alpha/beta densities are written there
 *  (for duplicate removal / spin classification).
 *  @return 0 on success; -1 on error; -3 if not UHF.
 *  On success *energy_out = converged total energy in Hartree. */
int gansu_uhf_scf_from_mo(gansu_handle_t h,
                          const double* c_alpha_occ,
                          const double* c_beta_occ,
                          int n, int na, int nb,
                          double* energy_out,
                          double* pa_out /* n*n or NULL */,
                          double* pb_out /* n*n or NULL */);

/** Spin properties of a UHF solution from AO densities Pa, Pb (each n x n,
 *  row-major). Writes <S^2> to *s_squared_out (may be NULL) and the per-atom
 *  Mulliken spin population m_A = sum_{mu in A} ((Pa - Pb) S)_mu,mu to
 *  atom_spin_out (natom doubles, may be NULL). <S^2> uses the UHF formula
 *  S(S+1) + Nb - Tr(Pa S Pb S) with Na = Tr(Pa S), Nb = Tr(Pb S),
 *  S = (Na - Nb)/2. Usable after gansu_prepare() or gansu_run().
 *  @return 0 on success; -1 on dimension mismatch / error; -3 if not UHF. */
int gansu_spin_properties(gansu_handle_t h,
                          const double* Pa, const double* Pb, int n,
                          double* s_squared_out,
                          double* atom_spin_out, int natom);

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
