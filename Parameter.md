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
| post_hf_method | Post-Hartree-Fock method to use (FCI, MP2, CCSD, CCSD(T), CIS, ADC2, EOM_MP2, EOM_CC2, EOM_CCSD) | string | none |
| n_excited_states | Number of excited states to compute | int | 5 |
| adc2_solver | Solver for ADC(2) (auto, schur_static, schur_omega, full) | string | auto |
| eom_mp2_solver | Solver for EOM-MP2 (auto, schur_static, schur_omega, full) | string | auto |
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
| export_molden | Output Molden file | bool | false |
| n_excited_states | Number of excited states to compute | int | 5 |
| adc2_solver | Solver for ADC(2) | string | auto |
| eom_mp2_solver | Solver for EOM-MP2 | string | auto |




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
* none - No post-Hartree-Fock method is applied
* FCI - Full Configuration Interaction method (FCI)
* MP2 - Møller-Plesset perturbation theory of second order (MP2)
* MP3 - Møller-Plesset perturbation theory of third order (MP3)
* CCSD - Coupled Cluster with Single and Double excitations (CCSD)
* CCSD_T - Coupled Cluster with Single, Double, and perturbative Triple excitations (CCSD(T))
* CIS - Configuration Interaction Singles (excited states)
* ADC2 - Algebraic Diagrammatic Construction of second order (excited states)
* EOM_MP2 - Equation-of-Motion MP2 (excited states)
* EOM_CC2 - Equation-of-Motion CC2 (excited states)
* EOM_CCSD - Equation-of-Motion CCSD (excited states)

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

#### convergence_method - Method to use for convergence
* default:  DIIS
* Damping - Damping method with constant damping factor
* OptimalDamping - Damping method with optimal damping factor (RHF, ROHF)
* DIIS - Direct Inversion of the Iterative Subspace (DIIS)

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


## Excited state parameters

| Parameter | Description | Type | Default |
| --- | --- | --- | --- |
| n_excited_states | Number of excited states to compute | int | 5 |
| adc2_solver | Solver for ADC(2) | string | auto |
| eom_mp2_solver | Solver for EOM-MP2 | string | auto |

These parameters are used when `post_hf_method` is set to an excited state method (CIS, ADC2, EOM_MP2, EOM_CC2, EOM_CCSD).

#### n_excited_states - Number of excited states to compute
* default: 5
* Number of lowest excited states to compute. Must not exceed the singles dimension (nocc × nvir).

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

```bash
# CIS with 10 excited states
./HF_main -x ../xyz/H2O.xyz -g ../basis/sto-3g.gbs --post_hf_method cis --n_excited_states 10

# ADC(2) with auto solver selection (default)
./HF_main -x ../xyz/H2O.xyz -g ../basis/cc-pvdz.gbs --post_hf_method adc2

# ADC(2) with explicit full Davidson solver
./HF_main -x ../xyz/H2O.xyz -g ../basis/cc-pvdz.gbs --post_hf_method adc2 --adc2_solver full

# EOM-MP2 with schur_omega solver
./HF_main -x ../xyz/H2O.xyz -g ../basis/cc-pvdz.gbs --post_hf_method eom_mp2 --eom_mp2_solver schur_omega

# EOM-CCSD
./HF_main -x ../xyz/H2O.xyz -g ../basis/cc-pvdz.gbs --post_hf_method eom_ccsd
```

## Geometry optimization parameters

| Parameter | Short | Description | Type | Default |
| --- | --- | --- | --- | --- |
| run_type | -r | Type of calculation | string | energy |
| optimizer | | Optimization algorithm | string | bfgs |

#### run_type - Type of calculation to perform

* default: energy
* energy - Single-point energy calculation only
* gradient - Single-point energy calculation followed by energy gradient evaluation
* optimize - Geometry optimization using analytical energy gradients

```bash
# Single-point energy (default)
./HF_main -x ../xyz/H2O.xyz -g ../basis/sto-3g.gbs

# Energy gradient
./HF_main -x ../xyz/H2O.xyz -g ../basis/sto-3g.gbs -r gradient

# Geometry optimization
./HF_main -x ../xyz/H2O.xyz -g ../basis/sto-3g.gbs -r optimize
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
./HF_main -x ../xyz/H2O.xyz -g ../basis/sto-3g.gbs -r optimize

# Geometry optimization with Polak-Ribière conjugate gradient
./HF_main -x ../xyz/H2O.xyz -g ../basis/sto-3g.gbs -r optimize --optimizer cg-pr

# Geometry optimization with GDIIS
./HF_main -x ../xyz/H2O.xyz -g ../basis/sto-3g.gbs -r optimize --optimizer gdiis

# UHF geometry optimization with SAD initial guess
./HF_main -x ../xyz/O2.xyz -g ../basis/sto-3g.gbs -m UHF --initial_guess sad -r optimize
```
