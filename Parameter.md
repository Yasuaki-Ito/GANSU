# Parameter desctiption

> [!NOTE]
> Parameters excluding file paths are not case-sensitive.

## All parameters

| Parameter | Description | Type | Default |
| --- | --- | --- | --- |
| parameter_file | Parameter recipe file (command line only) | string | |
| xyzfilename | XYZ file | string | |
| gbsfilename | Gaussian basis set file | string | |
| auxiliary_gbsfilename | Path to auxiliary Gaussian basis set file for RI approximation| string | |
| verbose | Verbose mode | bool | false |
| method | Method to use (RHF, UHF, ROHF) | string | RHF |
| charge | Charge of the molecule | int | 0 |
| beta_to_alpha | Number of shifted electrons from beta-spin to alpha-spin | int | 0 |
| maxiter | Maximum number of SCF iterations | int | 100 |
| convergence_energy_threshold | Energy convergence threshold | double | 1.0e-6 |
| eri_method | Method to use for two-electron repulsion integrals | string | stored |
| post_hf_method | Post-Hartree-Fock method to use (MP2, CCSD, CCSD(T)) | string | none |
| schwarz_screening_threshold | Schwarz screening threshold | double | 1.0e-12 |
| initial_guess | Method to use for initial guess | string | core |
| convergence_method | Method to use for convergence | string | DIIS |
| damping_factor | Damping factor | double | 0.9 |
| diis_size | Number of previous Fock matrices to store | int | 8 |
| diis_include_transform | Include the transformation matrix in DIIS | bool | false |
| rohf_parameter_name | ROHF parameter set name | string | Roothaan |
| mulliken | Perform Mulliken population analysis | bool | false |
| mayer | Perform Mayer bond order analysis | bool | false |
| wiberg | Perform Wiberg bond order analysis | bool | false |
| export_molden | Output Molden file | bool | false |




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
| eri_method | Method to use for two-electron repulsion integrals | string | stored |
| schwarz_screening_threshold | Schwarz screening threshold | double | 1.0e-12 |
| initial_guess | Method to use for initial guess | string | core |
| convergence_method | Method to use for convergence | string | DIIS |
| damping_factor | Damping factor | double | 0.9 |
| diis_size | Number of previous Fock matrices to store | int | 8 |
| diis_include_transform | Include the transformation matrix in DIIS | bool | false |
| rohf_parameter_name | ROHF parameter set name | string | Roothaan |


#### maxiter - Maximum number of SCF iterations
* default:  100
* Maximum number of SCF iterations to perform

#### convergence_energy_threshold - Energy convergence threshold
* default:  1.0e-6
* Energy convergence threshold for the SCF iterations

#### eri_method - Method to use for two-electron repulsion integrals
* default: stored
* stored - Two-electron repulsion integrals are stored in the device memory
* RI - Resolution of the Identity (RI) approximation is used for the two-electron repulsion integrals (ERIs)
* Direct - Direct calculation of the two-electron repulsion integrals (ERIs) without any approximation (Direct-SCF)

#### post_hf_method - Post-Hartree-Fock method to use (MP2, CCSD, CCSD(T))
* default: none
* none - No post-Hartree-Fock method is applied
* MP2 - Møller-Plesset perturbation theory of second order (MP2)
* MP3 - Møller-Plesset perturbation theory of third order (MP3)
* CCSD - Coupled Cluster with Single and Double excitations (CCSD)

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
