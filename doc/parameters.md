# Parameter desctiption

> [!NOTE]
> Parameters excluding file paths are not case-sensitive.

## All parameters

| Parameter | Description | Type | Default |
| --- | --- | --- | --- |
| parameter_file | Parameter recipe file (command line only) | string | |
| xyzfilename | XYZ file | string | |
| gbsfilename | Gaussian basis set file | string | |
| auxiliary_gbsfilename | Path to auxiliary Gaussian basis set file for RI approximation (auto-generated if omitted) | string | |
| verbose | Verbose mode | bool | false |
| method | Method to use (RHF, UHF, ROHF) | string | RHF |
| charge | Charge of the molecule | int | 0 |
| beta_to_alpha | Number of shifted electrons from beta-spin to alpha-spin | int | 0 |
| maxiter | Maximum number of SCF iterations | int | 100 |
| convergence_energy_threshold | Energy convergence threshold | double | 1.0e-6 |
| int1e_method | Method to use for one-electron integrals | string | hybrid |
| eri_method | Method to use for two-electron repulsion integrals | string | stored |
| post_hf_method | Post-Hartree-Fock method to use (FCI, MP2, SCS_MP2, SOS_MP2, LT_MP2, LT_SOS_MP2, MP3, MP4, CC2, CCSD, CCSD_T, CCSD_DENSITY, DMET_CCSD, DMET_CCSD_T, CIS, ADC2, SOS_ADC2, LT_SOS_ADC2, ADC2X, EOM_MP2, EOM_CC2, EOM_CCSD) | string | none |
| n_excited_states | Number of excited states to compute | int | 5 |
| spin_type | Spin type for excited states (singlet, triplet) | string | singlet |
| adc2_solver | Solver for ADC(2) (auto, schur_static, schur_omega, full) | string | auto |
| eom_mp2_solver | Solver for EOM-MP2 (auto, schur_static, schur_omega, full) | string | auto |
| eom_cc2_solver | Solver for EOM-CC2 (auto, schur_static, schur_omega, full) | string | auto |
| schwarz_screening_threshold | Schwarz screening threshold | double | 1.0e-12 |
| initial_guess | Method to use for initial guess | string | core |
| convergence_method | Method to use for convergence | string | DIIS |
| damping_factor | Damping factor | double | 0.9 |
| diis_size | Number of previous Fock matrices to store | int | 8 |
| diis_include_transform | Include the transformation matrix in DIIS | bool | false |
| rohf_parameter_name | ROHF parameter set name | string | Roothaan |
| run_type | Type of calculation to perform | string | energy |
| optimizer | Optimization algorithm for geometry optimization | string | bfgs |
| mulliken | Perform Mulliken population analysis | bool | false |
| mayer | Perform Mayer bond order analysis | bool | false |
| wiberg | Perform Wiberg bond order analysis | bool | false |
| export_molden | Output Molden file (canonical MOs) | bool | false |
| export_lmo_molden | Output Molden file with Pipek-Mezey localized occupied orbitals (RHF/UHF/ROHF; see [DLPNO-CCSD / DLPNO-CCSD(T) parameters](#dlpno-ccsd--dlpno-ccsdt-parameters)) | bool | false |
| ecp_filename | Path to ECP file for effective core potentials | string | |
| num_gpus | Number of GPUs for multi-GPU RI-HF (-1 = auto-detect all available) | int | -1 |
| dmet_fragments | DMET fragment specification (e.g. `"{0,6} {1,7}"`); empty = auto-detect by X-H bonds | string | "" |
| dmet_threshold | SVD threshold for DMET bath orbital selection (σ < threshold excluded) | double | 1.0e-6 |
| dmet_n_tol | DMET bisection tolerance on \|Σ N_frag − N_elec\| (Vayesta-compat: 4.2e-3 for benzene) | double | 1.0e-5 |
| dmet_mu_refine_ccsd | DMET 2-stage μ optimization (Stage 1: HF density, Stage 2: CCSD-relaxed density) | bool | false |
| dlpno_preset | DLPNO truncation preset (loose / normal / tight / very_tight, ORCA-compatible). See [DLPNO-CCSD / DLPNO-CCSD(T) parameters](#dlpno-ccsd--dlpno-ccsdt-parameters) | string | normal |
| dlpno_localizer | DLPNO occupied localization method (pm / boys / ibo) | string | pm |
| dlpno_lmp2_max_iter | DLPNO LMP2 / CCSD T2 iter max (shared) | int | 100 |
| dlpno_lmp2_conv | DLPNO LMP2 residual convergence | double | 1e-8 |
| dlpno_sc_pno_iter | DLPNO self-consistent PNO refinement rounds | int | 1 |
| dlpno_pair_distance_cutoff | DLPNO pair distance pre-screening (Bohr) | double | 15.0 |
| dlpno_verbose | DLPNO log verbosity (0/1/2/3) | int | 1 |
| opt_max_iter | Geometry optimization max iterations | int | 200 |
| opt_grad_threshold | Convergence: max gradient component (Hartree/Bohr) | double | 3.0e-4 |
| opt_rms_grad_threshold | Convergence: RMS gradient (Hartree/Bohr) | double | 2.0e-4 |
| opt_energy_threshold | Convergence: energy change (Hartree) | double | 1.0e-6 |
| opt_disp_threshold | Convergence: max displacement (Bohr) | double | 3.0e-4 |
| opt_step_max | Trust-region radius (Bohr) | double | 0.3 |




## Parameter parameters

| Parameter | Description | Type | Default |
| --- | --- | --- | --- |
| parameter_file | Parameter recipe file (command line only) | string | |

If the parameter recipe file is given, the parameters are read from the file. 
The parameters in the recipe file are overwritten by the other parameters.
This parameter is used only in the command line.

The contents of the parameter recipe file RHF_OptimalDamping.txt are a text file in which each line contains a parameter name and its value.
For example, the contents of the parameter recipe are as follows:
```
xyzfilename = ../xyz/H2O.xyz
gbsfilename = ../basis/sto-3g.gbs
method = RHF
convergence_method = OptimalDamping
```



## Input parameters

| Parameter | Description | Type | Default |
| --- | --- | --- | --- |
| xyzfilename | Path to XYZ file | string | |
| gbsfilename | Path to Gaussian basis set file | string | |


#### xyzfilename - Path to XYZ file
If the input molecular is given by Molecule class, this parameter is ignored.

#### gbsfilename - Path to Gaussian basis set file
If the input basis set is given by BasisSet class, this parameter is ignored.
However, if ``sad'' is used as the initial guess, the basis set is required to set this parameter.



## General parameters

| Parameter | Description | Type | Default |
| --- | --- | --- | --- |
| verbose | Verbose mode | bool | false |
| method | Method to use (RHF, UHF, ROHF) | string | RHF |

#### verbose - Verbose mode
* default:  false
* true - Print additional information
* false - Do not print additional information

#### method - Method to use (RHF, UHF, ROHF)
* default:  RHF
* RHF - Restricted Hartree-Fock
* UHF - Unrestricted Hartree-Fock
* ROHF - Restricted Open-Shell Hartree-Fock

## Molecule parameters

| Parameter | Description | Type | Default |
| --- | --- | --- | --- |
| charge | Charge of the molecule | int | 0 |
| beta_to_alpha | Number of shifted electrons from beta-spin to alpha-spin | int | 0 |

#### charge - Charge of the molecule
* default:  0
* Charge of the molecule (positive for cations, negative for anions)

#### beta_to_alpha - Number of shifted electrons from beta-spin to alpha-spin
* default:  0
* Number of shifted electrons from beta-spin to alpha-spin

### How are the electrons assigned to the alpha and beta spins?
Given parameters:
* $Z$ - The total number of positive charges in the nucleus of atoms in the molecule (= number of protons)
* $c$ - The charge of the molecule
* $u$ - The number of shifted electrons from beta-spin to alpha-spin


The numbers of electrons (alpha- and beta-spin electrons) 
* $N$ - The total number of electrons in the molecule
* $N_{\alpha}$ - The number of electrons with alpha spin
* $N_{\beta}$ - The number of electrons with beta spin

are calculated as follows:
* $N = Z - c$
* $N_{\alpha} = \left\lceil \frac{N}{2} \right\rceil + u$
* $N_{\beta} = \left\lfloor \frac{N}{2} \right\rfloor - u$

When the number of electrons is odd, the number of alpha-spin electrons is greater than the number of beta-spin electrons by one.
If any of the following conditions are met, an exception is thrown:
* $N < 1$ (no electrons in the molecule)
* $N_{\beta} < 0$ (the number of beta-spin electrons is negative)

## SCF parameters

| Parameter | Description | Type | Default |
| --- | --- | --- | --- |
| maxiter | Maximum number of SCF iterations | int | 100 |
| convergence_energy_threshold | Energy convergence threshold | double | 1.0e-6 |
| int1e_method | Method to use for one-electron integrals | string | hybrid |
| eri_method | Method to use for two-electron repulsion integrals | string | stored |
| schwarz_screening_threshold | Schwarz screening threshold | double | 1.0e-12 |
| initial_guess | Method to use for initial guess | string | core |
| convergence_method | Method to use for convergence | string | DIIS |
| damping_factor | Damping factor | double | 0.9 |
| diis_size | Number of previous Fock matrices to store | int | 8 |
| diis_include_transform | Include the transformation matrix in DIIS | bool | false |
| rohf_parameter_name | ROHF parameter set name | string | Roothaan |
| post_hf_method | Post-Hartree-Fock method to use | string | none |


#### maxiter - Maximum number of SCF iterations
* default:  100
* Maximum number of SCF iterations to perform

#### convergence_energy_threshold - Energy convergence threshold
* default:  1.0e-6
* Energy convergence threshold for the SCF iterations

#### int1e_method - Method to use for one-electrion integrals (overlap integrals, kinetic energy integrals, and nuclear attraction integrals)
* default: hybrid
* md: McMurchie-Davidson algorithm
* os: Obara-Saika algorithm

#### eri_method - Method to use for two-electron repulsion integrals
* default: stored
* stored - Two-electron repulsion integrals are stored in the device memory
* RI - Resolution of the Identity (RI) approximation is used for the two-electron repulsion integrals (ERIs)
* Direct - Direct calculation of the two-electron repulsion integrals (ERIs) without any approximation (Direct-SCF)
* Direct_RI - Resolution of the Identity (RI) approximation, but three-center ERIs are directly computed without storing

#### auxiliary_gbsfilename - Auxiliary basis set file for RI approximation
* default: (empty)
* When `eri_method` is `RI` or `Direct_RI`, an auxiliary basis set is required.
* If a file path is specified (e.g., `-ag ../auxiliary_basis/cc-pvdz-rifit.gbs`), the auxiliary basis is loaded from the file.
* If omitted, the auxiliary basis is **automatically generated** from the primary basis using the product basis approach:
  1. Collect all primitive Gaussian exponents $\{\alpha_i\}$ from the primary basis for each element
  2. Generate pairwise sums $\alpha_i + \alpha_j$ ($i \le j$)
  3. Remove near-duplicate exponents (keep only if consecutive exponents differ by a factor of $\ge 2$)
  4. Create uncontracted auxiliary functions (coefficient = 1.0) for angular momenta $L = 0, 1, \ldots, 2L_{\max}$, where $L_{\max}$ is the maximum angular momentum in the primary basis
* The auto-generated auxiliary basis provides a quick approximation but is less accurate than purpose-built auxiliary basis sets (e.g., cc-pVDZ-RIFIT). Use explicit auxiliary basis files for production calculations.

#### post_hf_method - Post-Hartree-Fock method to use
* default: none

Method names are case-insensitive. Hyphen variants are accepted (`scs-mp2` ≡ `scs_mp2`, `adc2-x` ≡ `adc2x`, `dmet-ccsd(t)` ≡ `dmet_ccsd_t`, etc.).

##### Ground-state correlation methods

| Value | Description |
| --- | --- |
| none | No post-HF method applied (HF only) |
| FCI | Full Configuration Interaction. Exact within the basis. Cost grows factorially — only feasible for very small systems |
| MP2 | Møller–Plesset perturbation theory of 2nd order. Spin-orbital form: $E_{\mathrm{MP2}} = \tfrac{1}{4}\sum_{ijab} \frac{\|\langle ij\|\|ab\rangle\|^2}{\varepsilon_i+\varepsilon_j-\varepsilon_a-\varepsilon_b}$ with antisymmetrized $\langle ij\|\|ab\rangle = (ia\|jb) - (ib\|ja)$. Closed-shell (RMP2) reduces to $\sum_{ijab}\frac{(ia\|jb)\,[2(ia\|jb)-(ib\|ja)]}{\varepsilon_i+\varepsilon_j-\varepsilon_a-\varepsilon_b}$. Splits into opposite-spin (OS) and same-spin (SS) components |
| SCS_MP2 | Spin-Component-Scaled MP2 (Grimme 2003). $E = c_{\mathrm{OS}} E_{\mathrm{OS}} + c_{\mathrm{SS}} E_{\mathrm{SS}}$ with $c_{\mathrm{OS}} = 6/5$, $c_{\mathrm{SS}} = 1/3$. More accurate than MP2 at the same cost |
| SOS_MP2 | Scaled Opposite-Spin MP2 (Jung & Head-Gordon 2004). Drops the same-spin term and rescales: $E = 1.3 \cdot E_{\mathrm{OS}}$. Same-spin omission removes the exchange integral path, enabling cheaper algorithms |
| LT_MP2 | Laplace-Transform MP2. Replaces the orbital-energy denominator with $\frac{1}{x} \approx \sum_k w_k e^{-t_k x}$ (double-exponential quadrature). Decouples occupied/virtual indices, useful for RI-MP2 acceleration. Aliases: `lt-mp2`, `laplace_mp2` |
| LT_SOS_MP2 | Laplace transform applied to SOS-MP2. The combination of opposite-spin-only and Laplace decoupling gives $\mathcal{O}(N^4)$ scaling with RI. Aliases: `lt-sos-mp2`, `laplace_sos_mp2` |
| MP3 | MP3. Adds 3rd-order doubles correction over MP2. $\mathcal{O}(N^6)$ |
| MP4 | MP4 (full: SDQ + (T) contributions). $\mathcal{O}(N^7)$. Alias: `mp4` |
| CC2 | Coupled-Cluster with approximate doubles. Doubles are kept at MP1 quality but coupled iteratively to singles. $\mathcal{O}(N^5)$ |
| CCSD | Coupled-Cluster with Singles and Doubles. $\mathcal{O}(N^6)$. Iterative |
| CCSD_T | CCSD(T) — CCSD plus perturbative triples evaluated once at the converged CCSD amplitudes. The "gold standard" for ground-state correlation. $\mathcal{O}(N^7)$ for the (T) step |
| CCSD_DENSITY | CCSD + Λ-equation solve + 1-RDM construction. Used internally by DMET and for natural-orbital / property analysis. Same energy as CCSD; adds Λ + 1-RDM cost |
| DMET_CCSD | Density Matrix Embedding Theory with CCSD as the impurity solver. Auto-fragmentation by X–H bonds (or `dmet_fragments` manual spec). See [DMET-CCSD parameters](#dmet-ccsd-parameters) |
| DMET_CCSD_T | DMET-CCSD plus per-fragment perturbative triples evaluated at the converged $\mu_{\mathrm{DMET}}$. Aliases: `dmet-ccsd_t`, `dmet_ccsd(t)`, `dmet-ccsd(t)`, `dmet_ccsdt` |
| DLPNO_CCSD | Domain-based Local Pair Natural Orbital CCSD (Riplinger & Neese 2013, 2016). Closed-shell RHF only; requires RI. Pipek-Mezey occupied LMOs → PAO + per-LMO atom domains → per-pair PNO truncation → strong/weak-pair partitioning. Aliases: `dlpno-ccsd`. See [DLPNO-CCSD / DLPNO-CCSD(T) parameters](#dlpno-ccsd--dlpno-ccsdt-parameters) |
| DLPNO_CCSD_T | DLPNO-CCSD plus perturbative triples evaluated on per-triple TNO bases via batched GPU kernels. PySCF-equivalent 6-W formulation. Aliases: `dlpno-ccsd_t`, `dlpno_ccsd(t)`, `dlpno-ccsd(t)`, `dlpno_ccsdt` |

##### Excited-state methods

| Value | Description |
| --- | --- |
| CIS | Configuration Interaction Singles. Diagonalizes $\langle\Phi_i^a\|H-E_0\|\Phi_j^b\rangle$. Lowest-cost excited-state method ($\mathcal{O}(N^4)$); generally over-estimates excitation energies by ~1 eV |
| ADC2 | Algebraic Diagrammatic Construction of 2nd order, strict variant ADC(2)-s. M22 (doubles–doubles block) is purely diagonal: $D_2 = \varepsilon_a + \varepsilon_b - \varepsilon_i - \varepsilon_j$ |
| SOS_ADC2 | Scaled-Opposite-Spin ADC(2). Same-spin doubles dropped, OS scaled. Aliases: `sos-adc2`, `sos_adc(2)` |
| LT_SOS_ADC2 | SOS-ADC(2) with Laplace-transformed denominators — $\mathcal{O}(N^4)$ with RI + Laplace quadrature. Aliases: `lt-sos-adc2`, `lt_sos_adc(2)` |
| ADC2X | ADC(2) extended (`adc(2)-x`). M22 includes first-order off-diagonal terms (oooo, vvvv, voov), giving lower (more accurate) excitation energies than ADC(2)-s. Always uses the full Davidson solver. Aliases: `adc2-x`, `adc(2)-x` |
| EOM_MP2 | Equation-of-Motion MP2 (≈ ADC(2) but Stanton–Bartlett style). M22 has off-diagonal couplings via T2 |
| EOM_CC2 | Equation-of-Motion CC2. Doubles kept at CC2 quality. M22 is exactly diagonal — Schur complement is exact (no approximation in `schur_omega`) |
| EOM_CCSD | Equation-of-Motion CCSD. Most accurate single-reference excited-state method available here |
| THC_SOS_ADC2 | Tensor Hypercontraction SOS-ADC(2) with Laplace transform (`O(N^3)` sigma build). LS-THC factorisation $V_{\mu\nu\lambda\sigma}\!\approx\!\sum_{PQ}X^P_\mu X^P_\nu Z_{PQ}X^Q_\lambda X^Q_\sigma$ on a Becke–Lebedev grid. Schur-folded Davidson + ω-iter. Set `--eri_method ri` (with auxiliary basis) for the memory-light Phase 2.3 RI-Z path; `--eri_method stored` works at small scales. See [THC parameters](#thc-parameters). Aliases: `thc-sos-adc2`, `thc_sos_adc(2)` |

For excited-state methods, see the [Excited state parameters](#excited-state-parameters) section for `n_excited_states`, `spin_type`, and per-method solver (`adc2_solver`, `eom_mp2_solver`, `eom_cc2_solver`).

#### schwarz_screening_threshold - schwarz screening threshold
* default:  1.0e-12
* Schwarz screening threshold for the two-electron repulsion integrals (ERIs)

Schwarz screening is used to reduce the computational cost of the two-electron repulsion integrals (ERIs).
Schwarz inequality is applied to the two-electron repulsion integrals (ERIs) to reduce the computational cost.
Schwarz inequality is given by the following inequality:
```math
\left|(\mu\nu|\lambda\sigma)\right| \le \sqrt{(\mu\nu|\mu\nu)} \sqrt{(\lambda\sigma|\lambda\sigma)}
```
where $(\mu\nu|\lambda\sigma)$ is the two-electron repulsion integral (ERI) of the basis functions $\phi_{\mu}$, $\phi_{\nu}$, $\phi_{\lambda}$, and $\phi_{\sigma}$:
```math
(\mu\nu|\lambda\sigma)=\iint \phi_{\mu}(\mathbf{r}_1) \phi_{\nu}(\mathbf{r}_1) \frac{1}{\mathbf{r}_{12}} \phi_{\lambda}(\mathbf{r}_2) \phi_{\sigma}(\mathbf{r}_2) d\mathbf{r}_1 d\mathbf{r}_2
```
Using Schwarz inequality, the two-electron repulsion integrals (ERIs) are calculated if $\sqrt{(\mu\nu|\mu\nu)} \sqrt{(\lambda\sigma|\lambda\sigma)}$ is greater than the Schwarz screening threshold.
Otherwise, the two-electron repulsion integrals (ERIs) are set to zero.

#### initial_guess - Method to use for initial guess
* default:  core
* core - Core Hamiltonian is used as the initial guess of the Fock matrix
* gwh - Generalized Wolfsberg-Helmholz method (GWH) is used as the initial guess of the Fock matrix
* sad - Superposition of Atomic Densities (SAD) is used as the initial guess of the Fock matrix
* density - Given density matrix is used as the initial guess of the Fock matrix
* minao - Minimal ANO (ANO-RCC-MB) projection initial guess (RHF)

#### convergence_method - Method to use for convergence
* default:  DIIS
* Damping - Damping method with constant damping factor
* OptimalDamping - Damping method with optimal damping factor (RHF, ROHF)
* DIIS - Direct Inversion of the Iterative Subspace (DIIS)
* SOSCF - Second-Order SCF (DIIS→SOSCF automatic switching, RHF)
* ADIIS - Augmented DIIS (RHF)
* EDIIS - Energy DIIS (RHF)
* AEDIIS - Automatic EDIIS→ADIIS→DIIS switching (RHF)

#### damping_factor - Damping factor for DIIS
* default:  0.9
* Damping factor for damping method

#### diis_size - Number of previous Fock matrices to store
* default:  8
* Number of previous Fock matrices to store for DIIS convergence algorithm

#### diis_include_transform - Includes the transformation matrix in DIIS
* default:  false
* true - Include the transformation matrix in DIIS for calculation of the error matrix $e$:
    * $e = X(FPS-SPF)X^T$
    * where $F$ is the Fock matrix, $P$ is the density matrix, and $S$ is the overlap matrix, and $X$ is the transformation matrix

* false - Do not include the transformation matrix for calculation of the error matrix $e$
    * $e = FPS - SPF$



#### rohf_parameter_name - ROHF parameter set name
* default:  Roothaan
* Parameter set name in computing the ROHF Fock matrix

| Parameter set name |  $A^{CC}$  |  $B^{CC}$  |  $A^{OO}$  |  $B^{OO}$  |  $A^{VV}$  |  $B^{VV}$  |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Roothaan |  $-1/2$    |  $3/2$     |  $1/2$     |  $1/2$     |  $3/2$     |  $-1/2$    |
| McWeeny-Diercksen |  $1/3$     |  $2/3$     |  $1/3$     |  $1/3$     |  $2/3$     |  $1/3$     |
| Davidson |  $1/2$     |  $1/2$     |  $1$       |  $0$       |  $1$       |  $0$       |
| Guest-Saunders |  $1/2$     |  $1/2$     |  $1/2$     |  $1/2$     |  $1/2$     |  $1/2$     |
| Binkley-Pople-Dobosh |  $1/2$     |  $1/2$     |  $1$       |  $0$       |  $0$       |  $1$       | 
| Faegri-Manne |  $1/2$     |  $1/2$     |  $1$       |  $0$       |  $1/2$     |  $1/2$     |
| Goddard |  $1/2$     |  $1/2$     |  $1/2$     |  $0$       |  $1/2$     |  $1/2$     |
| Plakhutin-Gorelik-Breslavskaya |  $0$       |  $1$       |  $1$       |  $0$       |  $1$       |  $0$       | 

#### mulliken - Perform Mulliken population analysis
* default:  false
* true - Perform Mulliken population analysis after the SCF calculation
* false - Do not perform Mulliken population analysis

#### mayer - Perform Mayer bond order analysis
* default:  false
* true - Perform Mayer bond order analysis after the SCF calculation
* false - Do not perform Mayer bond order analysis

#### wiberg - Perform Wiberg bond order analysis
* default:  false
* true - Perform Wiberg bond order analysis after the SCF calculation
* false - Do not perform Wiberg bond order analysis

#### export_molden - Export Molden file
* default:  false
* true - Export Molden file after the SCF calculation (output filename: output.molden)
* false - Do not output Molden file

#### export_lmo_molden - Export Pipek-Mezey localized occupied orbitals
* default:  false
* true - Localize occupied MOs via Pipek-Mezey and write them to `output_lmo.molden`. Occupied block contains the LMOs (C_LMO = C_occ · U); virtual block keeps the canonical orbitals. For UHF, α and β are localized independently. For ROHF, doubly-occupied and singly-occupied subspaces are localized separately so closed-shell core and open-shell electrons do not mix. The orbital energies written for each LMO are the diagonal Fock-in-LMO-basis values $\varepsilon_i^{\mathrm{LMO}} = \sum_k |U_{ki}|^2 \cdot \varepsilon_k^{\mathrm{can}}$ (exact since canonical $F$ is diagonal in the canonical MO basis).
* false - Do not export localized orbitals

Compatible Molden viewers: [MOrbVis](https://yasuaki-ito.github.io/morbvis/), [Avogadro](https://avogadro.cc/), Jmol, VMD, [Pegamoid](https://github.com/Jellby/Pegamoid).


## Excited state parameters

| Parameter | Description | Type | Default |
| --- | --- | --- | --- |
| n_excited_states | Number of excited states to compute | int | 5 |
| spin_type | Spin type for excited states | string | singlet |
| adc2_solver | Solver for ADC(2) | string | auto |
| eom_mp2_solver | Solver for EOM-MP2 | string | auto |
| eom_cc2_solver | Solver for EOM-CC2 | string | auto |

These parameters are used when `post_hf_method` is set to an excited state method (CIS, ADC2, ADC2X, EOM_MP2, EOM_CC2, EOM_CCSD).

#### n_excited_states - Number of excited states to compute
* default: 5
* Number of lowest excited states to compute. Must not exceed the singles dimension (nocc × nvir).

#### spin_type - Spin type for excited states
* default: singlet
* singlet - Compute singlet excited states (default). Oscillator strengths are computed.
* triplet - Compute triplet excited states. Oscillator strengths are zero (spin-forbidden electric dipole transitions from singlet ground state).
* Supported methods: CIS, ADC2, ADC2X. For triplet states, only the M11 block (CIS + ISR correction) differs from singlet; M12, M21, D2, and M22 are identical.

#### adc2_solver - Solver for ADC(2)
* default: auto
* auto - Automatically selects `full` or `schur_omega` based on available GPU memory (80% threshold)
* full - Full Davidson in singles+doubles space. Exact but requires more GPU memory.
* schur_omega - ω-dependent Schur complement with self-consistent iteration. M_eff(ω) = M11 + M12·(ωI − D2)⁻¹·M21. Iterates each root until ω converges.
* schur_static - Single Schur complement at ω=0. M_eff = M11 − M12·D2⁻¹·M21. Fastest but least accurate.

#### eom_mp2_solver - Solver for EOM-MP2
* default: auto
* auto - Automatically selects `full` or `schur_omega` based on available GPU memory (80% threshold)
* full - Full Davidson in singles+doubles space. Exact but may encounter near-zero eigenvalues from the doubles null space.
* schur_omega - ω-dependent Schur complement with self-consistent iteration. Builds dense M_eff(ω) matrix and uses non-symmetric eigendecomposition. More accurate than schur_static.
* schur_static - Approximate Schur complement at ω=0 using Davidson. M22 off-diagonal (t2×r2 coupling) is ignored. Fast but approximate.

#### eom_cc2_solver - Solver for EOM-CC2
* default: auto
* auto - Automatically selects `full` or `schur_omega` based on available GPU memory (80% threshold)
* full - Full Davidson in singles+doubles space. Exact. M22 is diagonal so no null space issues (unlike EOM-MP2).
* schur_omega - ω-dependent Schur complement with self-consistent Davidson iteration. EXACT (M22 is purely diagonal, no approximation). Default and recommended.
* schur_static - Schur complement at ω=0 using Davidson. EXACT Schur (no M22 approximation) but ω=0 approximation remains.

Note: For EOM-CC2, M22 is exactly diagonal, so the Schur complement introduces NO approximation (unlike EOM-MP2 where M22 off-diagonal terms are ignored). The only approximation in `schur_static` is the ω=0 assumption.

#### ADC(2)-s vs ADC(2)-x

ADC(2) has two variants:

| | ADC(2)-s (`adc2`) | ADC(2)-x (`adc2x`) |
|---|---|---|
| M11 (singles) | CIS + ISR + self-energy | Same as ADC(2)-s |
| M12, M21 (coupling) | Coupling blocks | Same as ADC(2)-s |
| M22 (doubles) | Diagonal only: D2 = ε_a+ε_b−ε_i−ε_j | D2 + first-order off-diagonal terms (oooo, vvvv, voov) |

ADC(2)-x includes explicit electron correlation in the doubles space, generally giving lower (more accurate) excitation energies. ADC(2)-x always uses the full Davidson solver.

```bash
# CIS with 10 excited states
./gansu -x ../xyz/H2O.xyz -g sto-3g --post_hf_method cis --n_excited_states 10

# ADC(2) with auto solver selection (default)
./gansu -x ../xyz/H2O.xyz -g cc-pvdz --post_hf_method adc2

# ADC(2)-x (extended)
./gansu -x ../xyz/H2O.xyz -g cc-pvdz --post_hf_method adc2x

# ADC(2) with explicit full Davidson solver
./gansu -x ../xyz/H2O.xyz -g cc-pvdz --post_hf_method adc2 --adc2_solver full

# Triplet excited states
./gansu -x ../xyz/H2O.xyz -g cc-pvdz --post_hf_method cis --spin_type triplet
./gansu -x ../xyz/H2O.xyz -g cc-pvdz --post_hf_method adc2 --spin_type triplet
./gansu -x ../xyz/H2O.xyz -g cc-pvdz --post_hf_method adc2x --spin_type triplet

# EOM-MP2 with schur_omega solver
./gansu -x ../xyz/H2O.xyz -g cc-pvdz --post_hf_method eom_mp2 --eom_mp2_solver schur_omega

# EOM-CC2 with full Davidson solver
./gansu -x ../xyz/H2O.xyz -g cc-pvdz --post_hf_method eom_cc2 --eom_cc2_solver full

# EOM-CCSD
./gansu -x ../xyz/H2O.xyz -g cc-pvdz --post_hf_method eom_ccsd
```

## Geometry optimization parameters

| Parameter | Short | Description | Type | Default |
| --- | --- | --- | --- | --- |
| run_type | -r | Type of calculation | string | energy |
| optimizer | | Optimization algorithm | string | bfgs |

#### run_type - Type of calculation to perform

* default: energy
* energy - Single-point energy calculation only
* gradient - Single-point energy calculation followed by analytical energy gradient evaluation
* optimize - Geometry optimization using analytical energy gradients (BFGS by default)
* hessian - Single-point energy followed by analytical Hessian and harmonic vibrational frequencies (mass-weighted normal modes, IR intensities). Translational/rotational modes are projected out

```bash
# Single-point energy (default)
./gansu -x ../xyz/H2O.xyz -g sto-3g

# Energy gradient
./gansu -x ../xyz/H2O.xyz -g sto-3g -r gradient

# Geometry optimization
./gansu -x ../xyz/H2O.xyz -g sto-3g -r optimize

# Analytical Hessian + vibrational frequencies
./gansu -x ../xyz/H2O.xyz -g sto-3g -r hessian
```

#### optimizer - Optimization algorithm for geometry optimization

* default: bfgs

This parameter is used only when `run_type` is set to `optimize`.

##### Quasi-Newton methods

Quasi-Newton methods build and update an approximate inverse Hessian matrix $H^{-1}$ to determine the search direction $\mathbf{p} = -H^{-1} \mathbf{g}$.
All quasi-Newton methods use Armijo backtracking line search with a trust radius.

| Value | Algorithm | Hessian update formula |
| --- | --- | --- |
| bfgs | Broyden-Fletcher-Goldfarb-Shanno | $H'^{-1} = H^{-1} + \frac{(\mathbf{s}^T \mathbf{y} + \mathbf{y}^T H^{-1} \mathbf{y})}{(\mathbf{s}^T \mathbf{y})^2} \mathbf{s}\mathbf{s}^T - \frac{H^{-1}\mathbf{y}\mathbf{s}^T + \mathbf{s}\mathbf{y}^T H^{-1}}{\mathbf{s}^T \mathbf{y}}$ |
| dfp | Davidon-Fletcher-Powell | $H'^{-1} = H^{-1} + \frac{\mathbf{s}\mathbf{s}^T}{\mathbf{y}^T\mathbf{s}} - \frac{H^{-1}\mathbf{y}\mathbf{y}^T H^{-1}}{\mathbf{y}^T H^{-1}\mathbf{y}}$ |
| sr1 | Symmetric Rank-1 | $H'^{-1} = H^{-1} + \frac{(\mathbf{s}-H^{-1}\mathbf{y})(\mathbf{s}-H^{-1}\mathbf{y})^T}{(\mathbf{s}-H^{-1}\mathbf{y})^T\mathbf{y}}$ |

where $\mathbf{s} = \mathbf{x}_{k+1} - \mathbf{x}_k$ (position change) and $\mathbf{y} = \mathbf{g}_{k+1} - \mathbf{g}_k$ (gradient change).

* **BFGS** is the most robust and widely used method. Maintains positive definiteness of the Hessian.
* **DFP** is the dual of BFGS. Updates the Hessian directly rather than the inverse. Generally less robust than BFGS.
* **SR1** can capture negative curvature in the Hessian, which may be useful for transition state searches. Does not guarantee positive definiteness.

The Hessian update is skipped when the curvature condition is not met ($\mathbf{s}^T\mathbf{y} \le 10^{-10}$ for BFGS/DFP, or $|(\mathbf{s}-H^{-1}\mathbf{y})^T\mathbf{y}| < 10^{-8} \|\mathbf{y}\| \|\mathbf{s}-H^{-1}\mathbf{y}\|$ for SR1).

##### Conjugate gradient methods

Conjugate gradient methods determine the search direction as $\mathbf{d}_k = -\mathbf{g}_k + \beta_k \mathbf{d}_{k-1}$, where $\beta_k$ is the conjugate gradient coefficient.
These methods do not require storage of an $N \times N$ Hessian matrix, making them memory-efficient for large systems.
All CG methods use Armijo backtracking line search with a trust radius.

| Value | Algorithm | $\beta_k$ |
| --- | --- | --- |
| cg-fr | Fletcher-Reeves | $\beta_k = \frac{\|\mathbf{g}_{k+1}\|^2}{\|\mathbf{g}_k\|^2}$ |
| cg-pr | Polak-Ribière | $\beta_k = \frac{\mathbf{g}_{k+1}^T(\mathbf{g}_{k+1} - \mathbf{g}_k)}{\|\mathbf{g}_k\|^2}$ |
| cg-hs | Hestenes-Stiefel | $\beta_k = \frac{\mathbf{g}_{k+1}^T(\mathbf{g}_{k+1} - \mathbf{g}_k)}{\mathbf{d}_k^T(\mathbf{g}_{k+1} - \mathbf{g}_k)}$ |
| cg-dy | Dai-Yuan | $\beta_k = \frac{\|\mathbf{g}_{k+1}\|^2}{\mathbf{d}_k^T(\mathbf{g}_{k+1} - \mathbf{g}_k)}$ |

Automatic restart: When $\beta_k < 0$, the method restarts with steepest descent ($\beta_k = 0$).
Descent check: If $\mathbf{g}_k^T \mathbf{d}_k \ge 0$ (not a descent direction), the search direction is reset to $-\mathbf{g}_k$.

##### GDIIS (Geometry Direct Inversion in the Iterative Subspace)

| Value | Algorithm |
| --- | --- |
| gdiis | GDIIS |

GDIIS combines quasi-Newton steps with DIIS extrapolation. It maintains an internal BFGS inverse Hessian and a subspace of recent geometries and error vectors.
At each step, the error vector $\mathbf{e}_i = -H^{-1}\mathbf{g}_i$ is computed and the DIIS equation

```math
\min_{\mathbf{c}} \sum_{ij} c_i B_{ij} c_j \quad \text{subject to} \quad \sum_i c_i = 1
```

is solved, where $B_{ij} = \mathbf{e}_i \cdot \mathbf{e}_j$. The new geometry is obtained as $\mathbf{x}_{\mathrm{new}} = \sum_i c_i (\mathbf{x}_i + \mathbf{e}_i)$.

GDIIS does not use line search (the step is directly accepted). The maximum subspace size is 6.

##### Steepest descent

| Value | Algorithm |
| --- | --- |
| sd | Steepest Descent |

The search direction is simply $\mathbf{p} = -\mathbf{g}$. Uses Armijo backtracking line search with a trust radius. Convergence is slow due to zigzag behavior, but it is useful for debugging.

### Convergence criteria

The geometry optimization uses four convergence criteria:

| Criterion | Threshold | Description |
| --- | --- | --- |
| Max gradient | $3.0 \times 10^{-4}$ Hartree/Bohr | Maximum gradient component |
| RMS gradient | $2.0 \times 10^{-4}$ Hartree/Bohr | Root mean square of gradient |
| Energy change | $1.0 \times 10^{-6}$ Hartree | Absolute energy change between steps |
| Max displacement | $3.0 \times 10^{-4}$ Bohr | Maximum atomic displacement |

Convergence is declared when **both** gradient criteria (max and RMS) are satisfied, **or** when both energy change and max displacement criteria are satisfied.
The maximum number of optimization steps is 200 and the trust radius is 0.3 Bohr.

### Translational and rotational invariance

At each optimization step, the translational and rotational components are projected out from the gradient and the search direction using Gram-Schmidt orthogonalization against the 6 (or 5 for linear molecules) basis vectors spanning the translational and rotational degrees of freedom.

### Output control

| run_type | SCF iterations | Profiler | SAD log | Gradient |
| --- | --- | --- | --- | --- |
| energy | displayed | displayed | displayed | - |
| gradient | displayed | displayed | displayed | displayed |
| optimize | suppressed | suppressed | suppressed | used internally |

```bash
# Geometry optimization with BFGS (default)
./gansu -x ../xyz/H2O.xyz -g sto-3g -r optimize

# Geometry optimization with Polak-Ribière conjugate gradient
./gansu -x ../xyz/H2O.xyz -g sto-3g -r optimize --optimizer cg-pr

# Geometry optimization with GDIIS
./gansu -x ../xyz/H2O.xyz -g sto-3g -r optimize --optimizer gdiis

# UHF geometry optimization with SAD initial guess
./gansu -x ../xyz/O2.xyz -g sto-3g -m UHF --initial_guess sad -r optimize
```


## DMET-CCSD parameters

| Parameter | Description | Type | Default |
| --- | --- | --- | --- |
| dmet_fragments | Fragment specification (e.g. `"{0,6} {1,7}"`); empty = auto-detect by X-H bonds | string | "" |
| dmet_threshold | SVD threshold for bath orbital selection (σ < threshold excluded) | double | 1.0e-6 |
| dmet_n_tol | Bisection tolerance on \|Σ N_frag − N_elec\| | double | 1.0e-5 |
| dmet_mu_refine_ccsd | 2-stage μ optimization (HF density → CCSD-relaxed density) | bool | false |

DMET-CCSD partitions the molecule into atom-localized fragments, builds a Schmidt-decomposed embedding cluster (fragment AOs + bath orbitals) for each, and solves CCSD on each cluster independently. A global chemical potential μ is bisected to satisfy the embedded-fragment density-consistency condition Σ N_frag = N_elec. Equivalent fragments are detected via embedding-Hamiltonian eigenvalue matching and reused.

For DMET-CCSD(T), pass `--post_hf_method dmet_ccsd_t` (or `dmet-ccsd_t`, `dmet_ccsdt`, `dmet_ccsd(t)`). The (T) perturbative-triples correction is computed per fragment at the converged μ_DMET using canonical denominators (small \|f_ov\| at the converged μ makes the canonical-(T) approximation acceptable), then summed across fragments with equivalent-fragment reuse.

#### dmet_fragments - Fragment specification
Empty string (default) triggers automatic detection: each heavy atom becomes a fragment, and each hydrogen joins its nearest heavy-atom fragment within 2.6 Bohr (≈ 1.38 Å). For example, benzene (C6H6) auto-detects 6 CH fragments.

Manual specification uses brace notation, atoms 0-indexed:
```bash
# 6 CH pairs in benzene (atoms C0/H6, C1/H7, ...)
./gansu -x ../xyz/Benzene.xyz -g sto-3g --post_hf_method dmet \
    --dmet_fragments "{0,6} {1,7} {2,8} {3,9} {4,10} {5,11}"
```

#### dmet_threshold - SVD threshold
Bath orbitals are constructed from the SVD of `C_lo_occ[frag_AOs, :]`. Singular values σ < `dmet_threshold` are excluded as numerical noise; σ ≥ 1 − 1e-12 are core orbitals (excluded from the active cluster, counted in n_core).

#### dmet_n_tol - Bisection tolerance
Bisection of μ stops when |Σ_F N_frag(μ) − N_elec| < `dmet_n_tol`. Default 1e-5 enforces strict density consistency. For Vayesta-compatible loose convergence (4.2e-3 for benzene, matching Vayesta's `max_elec_err = 1e-4 × N_elec`), pass `--dmet_n_tol 4.2e-3` (yields earlier termination, slightly different μ_DMET).

#### dmet_mu_refine_ccsd - 2-stage μ optimization
* **false** (default): single-stage CCSD-density bisection. Runs CCSD/Lambda/dm1 at every μ evaluation; the relaxed dm1 trace defines `N_frag(μ)`.
* **true**: two-stage refinement. Stage 1 performs HF-density bisection (fast, μ-continuous). Stage 2 refines around `μ_HF*` using CCSD-relaxed dm1 in a tight bracket. Falls back to Stage 1 result if Stage 2 hits a discontinuity (e.g. heteroatom OH fragments). Recommended only for paper-quality match with external DMET implementations; default is sufficient for most uses.

```bash
# Auto fragment detection, default tight tol
./gansu -x ../xyz/Benzene.xyz -g sto-3g --eri_method ri \
    -ag ../auxiliary_basis/cc-pvdz-rifit.gbs \
    --post_hf_method dmet --num_gpus 4

# DMET-CCSD(T) — adds perturbative triples per fragment at μ_DMET
./gansu -x ../xyz/Benzene.xyz -g sto-3g --eri_method ri \
    -ag ../auxiliary_basis/cc-pvdz-rifit.gbs \
    --post_hf_method dmet_ccsd_t --num_gpus 4

# Vayesta-compatible loose tolerance for benchmarking
./gansu -x ../xyz/Benzene.xyz -g sto-3g --eri_method ri \
    -ag ../auxiliary_basis/cc-pvdz-rifit.gbs \
    --post_hf_method dmet --dmet_n_tol 4.2e-3

# Verbose per-fragment diagnostics (for debugging)
GANSU_DMET_VERBOSE=1 ./gansu ... --post_hf_method dmet
```


## DLPNO-CCSD / DLPNO-CCSD(T) parameters

| Parameter | Description | Type | Default |
| --- | --- | --- | --- |
| dlpno_preset | Truncation preset (`loose`, `normal`, `tight`, `very_tight`) — sets all `t_cut_*` cutoffs to ORCA-compatible values | string | normal |
| dlpno_localizer | Occupied localization method (`pm`, `boys`, `ibo`) | string | pm |
| dlpno_t_cut_pno | PNO occupation cutoff. `-1` = use preset value | double | -1 |
| dlpno_t_cut_do | PAO redundancy threshold (overlap eigenvalue). `-1` = preset | double | -1 |
| dlpno_t_cut_pairs | Strong/weak pair MP2 cutoff in Ha. `-1` = preset | double | -1 |
| dlpno_t_cut_mkn | Boughton-Pulay Mulliken cumulative threshold for domain selection. `-1` = preset | double | -1 |
| dlpno_t_cut_triples | Triple screening threshold for (T). `-1` = preset | double | -1 |
| dlpno_t_cut_tno | TNO occupation cutoff for (T). `-1` = preset | double | -1 |
| dlpno_pair_distance_cutoff | Pair distance pre-screening cutoff in Bohr. `0` = off | double | 15.0 |
| dlpno_max_iter | DLPNO-CCSD residual max iterations | int | 50 |
| dlpno_diis_size | DIIS subspace size for DLPNO-CCSD | int | 6 |
| dlpno_localizer_max_sweep | Pipek-Mezey / Boys Jacobi sweep upper bound | int | 200 |
| dlpno_localizer_conv | Localizer functional ΔL convergence threshold | double | 1e-10 |
| dlpno_lmp2_max_iter | LMP2 / CCSD T2 dressing iter max (shared cap) | int | 100 |
| dlpno_lmp2_conv | LMP2 residual convergence (max\|R\|) | double | 1e-8 |
| dlpno_sc_pno_iter | Self-consistent PNO refinement rounds (`0` = single-shot from semi-canonical guess) | int | 1 |
| dlpno_pno_os_only | PNO selection from opposite-spin amplitudes only (use with SOS-MP2 scaling) | bool | false |
| dlpno_verbose | `0`=summary, `1`=phase, `2`=per-pair profile, `3`=residual | int | 1 |
| export_lmo_molden | Write Pipek-Mezey localized occupied MOs to `output_lmo.molden` | bool | false |

DLPNO-CCSD and DLPNO-CCSD(T) (Riplinger, Neese, Pinski, Sandhoefer 2013/2016) achieve near-canonical CCSD(T) accuracy at near-linear scaling by exploiting four nested locality layers: (1) occupied LMOs (Pipek-Mezey), (2) projected atomic orbitals (PAO) + per-LMO atom domains, (3) per-pair PNO truncation, and (4) strong/weak-pair partitioning where weak pairs are reduced to MP2.

This release implements the **closed-shell RHF** variant; UHF / ROHF DLPNO is not yet available. The RI back-end (`--eri_method ri` plus an `-ag <aux>`) is required.

### dlpno_preset — ORCA-compatible preset values

| Preset | t_cut_pno | t_cut_do | t_cut_pairs | t_cut_triples | t_cut_tno |
| --- | --- | --- | --- | --- | --- |
| loose      | 1.0e-6   | 2.0e-2 | 1.0e-3 | 1.0e-6 | 1.0e-9  |
| normal     | 3.33e-7  | 1.0e-2 | 1.0e-4 | 1.0e-7 | 1.0e-9  |
| tight      | 1.0e-7   | 5.0e-3 | 1.0e-5 | 1.0e-8 | 1.0e-10 |
| very_tight | 1.0e-8   | 2.0e-3 | 1.0e-6 | 1.0e-9 | 1.0e-10 |

Individual `dlpno_t_cut_*` parameters override the preset when set to a non-negative value.

### Examples

```bash
# DLPNO-CCSD on water hexamer (normal preset)
./gansu -x ../xyz/large_molecular/water_hexamer.xyz -g cc-pvdz \
    --eri_method ri -ag ../auxiliary_basis/cc-pvdz-rifit.gbs \
    --post_hf_method dlpno_ccsd --dlpno_preset normal --num_gpus 8

# DLPNO-CCSD(T) with tight preset for benchmarking
./gansu -x ../xyz/large_molecular/water_hexamer.xyz -g cc-pvdz \
    --eri_method ri -ag ../auxiliary_basis/cc-pvdz-rifit.gbs \
    --post_hf_method dlpno_ccsd_t --dlpno_preset tight --num_gpus 8

# Detailed per-section profile (debug)
./gansu ... --post_hf_method dlpno_ccsd_t --dlpno_verbose 2

# Export Pipek-Mezey localized orbitals for visualization
./gansu -x ../xyz/Benzene.xyz -g cc-pvdz --export_lmo_molden 1
```


## THC parameters

Used by `--post_hf_method thc_sos_adc2`. The grid+collocation+LS-THC infrastructure also serves the placeholder `thc_mp2` / `thc_sos_mp2` paths.

| Parameter | Description | Type | Default |
| --- | --- | --- | --- |
| `thc_n_radial` | Treutler–Ahlrichs M3 radial points per atom | int | 50 |
| `thc_lebedev_order` | Lebedev angular order: 110, 194, or 302 | int | 194 |
| `thc_n_laplace` | Laplace quadrature points (SOS-MP2 / SOS-ADC(2)) | int | 12 |
| `thc_rel_cutoff` | LS-THC SVD relative cutoff for the rank of M | real | 1e-7 |
| `thc_sos_c_os` | Opposite-spin scaling factor for SOS-MP2 (ADC(2) uses internal 1.17) | real | 1.3 |
| `thc_density_threshold` | Drop grid points with electron density ρ ≤ threshold (Phase 2.3 (B)). 0 disables | real | 0 |
| `thc_b3a3` | Master switch for B3-exchange + A3-Coulomb Schur corrections (Phase 2.2b). Off by default — present implementation over-corrects by ~1.5 eV (LS-THC structural limit) | bool | false |
| `thc_b3` | Per-term toggle for B3 (only consulted when `thc_b3a3=true`) | bool | true |
| `thc_a3` | Per-term toggle for A3 (only consulted when `thc_b3a3=true`) | bool | true |

### Recommended usage

* **Default (Coulomb-only Schur)** — gives CIS-level excitation energies; matches RI-Coulomb-only-SOS-Schur up to grid quality. Production-ready
* **B3+A3 enabled** — implementation is complete but over-corrects (LS-THC structural). Use only for debugging / benchmarking. Per-term toggles `thc_b3` / `thc_a3` allow isolated study
* **RI-Z path** — `--eri_method ri -ag <aux_basis>` selects the memory-light Phase 2.3 (A) path; the LS-THC core $Z$ is built from the RI 3-index tensor without ever materialising the analytic 4-index ERI. Multi-GPU is supported via auto-gather of `d_B_local_` to GPU 0
* **Density pruning** — `--thc_density_threshold 1e-8` typical. Drops grid points where the SCF density falls below threshold; preserves excitation-energy precision (8-digit agreement in H2O/sto-3g testing) and reduces N_g 3–10 %

### Memory budget

Per-GPU RAM cost on benzene/cc-pVDZ scale:

| Buffer | Size | n=15/L=194 (N_g=33232) |
| --- | --- | --- |
| 3 × N_g² | Y, ZY, T (Coulomb) | 26.5 GB |
| Z replica | LS-THC core | 8.8 GB |
| 3 × N_g² extra | B3+A3 (Z_occ, W, U) | 26.5 GB |
| Other (M, F, MT, X_mo, …) | ov × N_g · O(1) | ~1 GB |

A100 80 GB fits Coulomb-only at `thc_n_radial 20–25 / lebedev 194` and Coulomb+B3+A3 at `thc_n_radial 15 / lebedev 194` for benzene-class systems.

### Example

```bash
# Production: Phase 2.3 RI-Z + density pruning, Coulomb-only
./gansu -x ../xyz/large_molecular/Benzene.xyz -g cc-pvdz -m RHF \
    --eri_method ri -ag ../auxiliary_basis/cc-pvdz-rifit.gbs \
    --post_hf_method thc_sos_adc2 \
    --n_excited_states 5 \
    --thc_n_radial 15 --thc_lebedev_order 194 --thc_n_laplace 8 \
    --thc_density_threshold 1e-8

# Diagnostic: B3+A3 with per-term toggles
./gansu -x ../xyz/H2O.xyz -g sto-3g -m RHF \
    --post_hf_method thc_sos_adc2 --thc_b3a3 true --thc_b3 true --thc_a3 false
```

## Hardware / backend parameters

| Parameter | Description | Type | Default |
| --- | --- | --- | --- |
| num_gpus | Number of GPUs for multi-GPU RI-HF | int | -1 |
| --cpu | Run entirely on CPU (no GPU required) | flag | off |
| --list-basis | List available built-in basis sets and exit | flag | — |

#### num_gpus - Number of GPUs for multi-GPU RI-HF

* default: -1 (auto-detect all visible GPUs)
* Multi-GPU parallelism is currently implemented for **RI-HF** (`eri_method = RI`) and **DMET fragment-parallel** execution. Other ERI methods (`stored`, `Direct`) run on a single GPU regardless of this setting
* Set to a positive integer to cap the number of GPUs used (e.g. `--num_gpus 2` on a 4-GPU node)
* For DMET, fragments are distributed across GPUs in Replicated mode (each GPU holds the full RI tensor) when memory permits; otherwise Distributed mode is used (RI tensor sharded by aux index, NCCL AllReduce for Fock build)

```bash
# Single-GPU RI-HF
./gansu -x ../xyz/large_molecular/fullerene.xyz -g sto-3g \
    --eri_method ri -ag ../auxiliary_basis/cc-pvdz-rifit.gbs

# 4-GPU RI-HF
./gansu -x ../xyz/large_molecular/fullerene.xyz -g sto-3g \
    --eri_method ri -ag ../auxiliary_basis/cc-pvdz-rifit.gbs --num_gpus 4

# Multi-GPU DMET-CCSD(T) on water hexamer
./gansu -x ../xyz/H2O_hexamer_prism_opt.xyz -g sto-3g \
    --eri_method ri -ag ../auxiliary_basis/def2-svp-rifit.gbs \
    --post_hf_method dmet_ccsd_t --num_gpus 4
```

#### --cpu - CPU-only execution

* default: off (GPU is used)
* When the `--cpu` flag is passed, GANSU runs entirely on CPU using OpenMP-parallelized integral and tensor kernels. No NVIDIA GPU is required
* Supported: HF (RHF/UHF/ROHF), gradients, Hessians, all post-HF methods (MP2/3/4, CCSD, CCSD(T), CIS, ADC(2), EOM-CCSD, FCI), DMET-CCSD
* Performance is substantially lower than the GPU path; use for development, debugging, or environments without CUDA

```bash
./gansu -x ../xyz/H2O.xyz -g sto-3g --cpu
./gansu -x ../xyz/H2O.xyz -g sto-3g -r hessian --cpu
./gansu -x ../xyz/H2O.xyz -g cc-pvdz --post_hf_method ccsd --cpu
```

#### --list-basis - List available basis sets

* Prints the names of all built-in basis sets shipped under `basis/` and exits without running a calculation
* Useful for discovering the exact basis name to pass to `-g`

```bash
./gansu --list-basis
```


## ECP parameters

| Parameter | Description | Type | Default |
| --- | --- | --- | --- |
| ecp_filename | Path to ECP (Effective Core Potential) file | string | (none) |

#### ecp_filename - Effective Core Potential file

* default: (empty — all-electron calculation)
* When set, the specified ECP replaces the core electrons of the listed elements with a parameterized potential. Useful for heavy elements where relativistic / large-core effects matter
* File format follows the Gaussian-style ECP block convention (matches the entries on Basis Set Exchange)

```bash
./gansu -x ../xyz/heavy_atom.xyz -g cc-pvdz --ecp_filename ../basis/cc-pvdz-pp.ecp
```

