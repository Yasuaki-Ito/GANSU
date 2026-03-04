/** Types shared between UI and API */

export interface CalculationParams {
  xyz_text: string;
  xyz_file: string;
  xyz_dir: string;
  basis: string;
  method: string;
  charge: number;
  beta_to_alpha: number;
  convergence_method: string;
  diis_size: number;
  diis_include_transform: boolean;
  damping_factor: number;
  rohf_parameter_name: string;
  maxiter: number;
  convergence_energy_threshold: number;
  schwarz_screening_threshold: number;
  initial_guess: string;
  post_hf_method: string;
  eri_method: string;
  auxiliary_basis: string;
  auxiliary_basis_dir: string;
  mulliken: boolean;
  mayer: boolean;
  wiberg: boolean;
  export_molden: boolean;
  verbose: boolean;
  timeout: number;
}

export const DEFAULT_PARAMS: CalculationParams = {
  xyz_text: '',
  xyz_file: '',
  xyz_dir: '.',
  basis: 'sto-3g',
  method: 'RHF',
  charge: 0,
  beta_to_alpha: 0,
  convergence_method: 'diis',
  diis_size: 8,
  diis_include_transform: false,
  damping_factor: 0.9,
  rohf_parameter_name: 'Roothaan',
  maxiter: 100,
  convergence_energy_threshold: 1e-6,
  schwarz_screening_threshold: 1e-12,
  initial_guess: 'core',
  post_hf_method: 'none',
  eri_method: 'stored',
  auxiliary_basis: '',
  auxiliary_basis_dir: 'auxiliary_basis',
  mulliken: false,
  mayer: false,
  wiberg: false,
  export_molden: false,
  verbose: false,
  timeout: 600,
};

export interface SCFIteration {
  iteration: number;
  energy?: number;
  total_energy?: number;
  delta_e?: number;
}

export interface AtomInfo {
  index: number;
  element: string;
  coords: number[];
}

export interface OrbitalEnergy {
  index: number;
  occupation: string;
  energy: number;
}

export interface MullikenCharge {
  index: number;
  element: string;
  charge: number;
}

export interface CalculationResult {
  ok: boolean;
  error?: string;
  raw_output: string;
  molecule: {
    num_atoms?: number;
    num_electrons?: number;
    alpha_electrons?: number;
    beta_electrons?: number;
    atoms?: AtomInfo[];
  };
  basis_set: {
    num_basis?: number;
    num_primitives?: number;
    num_auxiliary?: number;
  };
  scf_iterations: SCFIteration[];
  summary: {
    method?: string;
    total_energy?: number;
    electronic_energy?: number;
    iterations?: number;
    computing_time_ms?: number;
    convergence_algorithm?: string;
    initial_guess?: string;
    schwarz_threshold?: number;
    convergence_criterion?: number;
    energy_difference?: number;
  };
  post_hf?: {
    method?: string;
    correction?: number;
    total_energy?: number;
  };
  orbital_energies: OrbitalEnergy[];
  orbital_energies_beta: OrbitalEnergy[];
  mulliken: MullikenCharge[];
  mayer_bond_order: number[][];
  wiberg_bond_order: number[][];
  timing: {
    entries?: { name: string; time: number; unit: string; calls: number }[];
  };
  molden_content?: string;
}

export interface SampleMolecule {
  filename: string;
  name: string;
}

/** SSE event types */
export type StreamEvent =
  | { type: 'line'; text: string }
  | { type: 'result'; data: CalculationResult }
  | { type: 'error'; error: string; raw_output: string }
  | { type: 'done' };
