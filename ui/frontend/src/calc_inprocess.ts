/**
 * GANSU-UI — In-Process Calculation page.
 * Same UI as the main Calculation page, but uses Python API directly (no subprocess).
 * Streams progress via SSE for real-time ProgressTracker popup.
 */

import { fetchSampleDirs, fetchSamples, fetchBasisSets, fetchAuxiliaryBasisSets } from './api';
import { DEFAULT_PARAMS } from './types';
import type { CalculationParams, CalculationResult } from './types';
import { createMoleculePanel } from './ui/moleculePanel';
import { createSettingsPanel } from './ui/settingsPanel';
import { createResultsPanel } from './ui/resultsPanel';
import { toggleTheme } from './ui/theme';
import { ProgressTracker, buildSteps } from './ui/progressTracker';

async function initApp(root: HTMLElement) {
  let sampleDirs = ['.'];
  let samples = [{ filename: 'H2.xyz', name: 'H2' }, { filename: 'H2O.xyz', name: 'H2O' }];
  let basisSets = ['sto-3g', '3-21g', '6-31g', 'cc-pvdz', 'cc-pvtz'];
  let auxBasisSets: { name: string; dir: string }[] = [];

  try {
    const [sd, s, b, ab] = await Promise.all([fetchSampleDirs(), fetchSamples(), fetchBasisSets(), fetchAuxiliaryBasisSets()]);
    if (sd.length > 0) sampleDirs = sd;
    if (s.length > 0) samples = s;
    if (b.length > 0) basisSets = b;
    if (ab.length > 0) auxBasisSets = ab;
  } catch (e) {
    console.warn('API not available, using defaults:', e);
  }

  root.innerHTML = `
    <header>
      <div class="header-top">
        <h1>GANSU</h1>
        <span class="subtitle">GPU Accelerated Numerical Simulation Utility</span>
        <button id="theme-btn" class="icon-btn" title="Toggle theme">
          <span id="theme-icon">&#9790;</span>
        </button>
      </div>
      <nav class="demo-nav" id="demo-nav">
        <a href="./" class="demo-tab">Calculation (subprocess)</a>
        <a href="./pes.html" class="demo-tab">PES</a>
        <a href="./geomopt.html" class="demo-tab">Geometry Opt</a>
        <a class="demo-tab active">In-Process</a>
      </nav>
    </header>
    <div class="main-grid">
      <div id="molecule-col"></div>
      <div id="settings-col"></div>
    </div>
    <div class="action-bar">
      <button id="run-btn" class="primary-btn">Run Calculation</button>
      <button id="cancel-btn" class="secondary-btn hidden">Cancel</button>
    </div>
    <div id="results-col"></div>
  `;

  // Theme toggle
  const themeBtn = root.querySelector<HTMLButtonElement>('#theme-btn')!;
  const themeIcon = root.querySelector<HTMLElement>('#theme-icon')!;
  themeBtn.addEventListener('click', () => {
    const next = toggleTheme();
    themeIcon.textContent = next === 'dark' ? '\u2600' : '\u263E';
  });

  // Init panels
  const molPanel = createMoleculePanel(
    root.querySelector('#molecule-col')!,
    sampleDirs,
    samples,
    () => {},
  );
  const settingsPanel = createSettingsPanel(
    root.querySelector('#settings-col')!,
    basisSets,
    auxBasisSets,
  );
  const resultsPanel = createResultsPanel(root.querySelector('#results-col')!);

  const runBtn = root.querySelector<HTMLButtonElement>('#run-btn')!;
  const cancelBtn = root.querySelector<HTMLButtonElement>('#cancel-btn')!;

  let abortController: AbortController | null = null;

  runBtn.addEventListener('click', async () => {
    const xyz = molPanel.getXyz();
    const xyzFile = molPanel.getXyzFile();

    if (!xyz && !xyzFile) {
      alert('Please enter a molecule (XYZ text or select a sample).');
      return;
    }

    const settingsParams = settingsPanel.getParams();
    const params: CalculationParams = {
      ...DEFAULT_PARAMS,
      ...settingsParams,
      xyz_text: xyz,
      xyz_file: xyzFile,
      xyz_dir: molPanel.getXyzDir(),
    };

    runBtn.disabled = true;
    cancelBtn.classList.remove('hidden');
    resultsPanel.hide();

    // Create progress tracker
    const steps = buildSteps(params.post_hf_method, params.eri_method, 'energy');
    const tracker = new ProgressTracker(steps, () => {
      if (abortController) {
        abortController.abort();
        abortController = null;
      }
      tracker.close();
      runBtn.disabled = false;
      cancelBtn.classList.add('hidden');
    });

    abortController = new AbortController();

    try {
      const res = await fetch('/api/run/inprocess/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          xyz_text: params.xyz_text,
          xyz_file: params.xyz_file,
          xyz_dir: params.xyz_dir,
          basis: params.basis,
          method: params.method,
          charge: params.charge,
          beta_to_alpha: params.beta_to_alpha,
          convergence_method: params.convergence_method,
          diis_size: params.diis_size,
          damping_factor: params.damping_factor,
          maxiter: params.maxiter,
          convergence_energy_threshold: params.convergence_energy_threshold,
          schwarz_screening_threshold: params.schwarz_screening_threshold,
          initial_guess: params.initial_guess,
          post_hf_method: params.post_hf_method,
          n_excited_states: params.n_excited_states,
          spin_type: params.spin_type,
          eri_method: params.eri_method,
          auxiliary_basis: params.auxiliary_basis,
          auxiliary_basis_dir: params.auxiliary_basis_dir,
          mulliken: params.mulliken,
          mayer: params.mayer,
          wiberg: params.wiberg,
        }),
        signal: abortController.signal,
      });

      const reader = res.body?.getReader();
      if (!reader) return;
      const decoder = new TextDecoder();
      let buffer = '';

      const nextFrame = () => new Promise<void>(r => requestAnimationFrame(() => r()));
      let lastSeenStage = '';
      const logLines: string[] = [];
      const t0 = performance.now();

      function logProgress(event: any) {
        const elapsed = ((performance.now() - t0) / 1000).toFixed(3);
        const stage = event.stage;
        const iter = event.iteration;
        const vals = event.values || [];
        if (stage === 'setup') {
          logLines.push(`[${elapsed}s] ${iter === 0 ? 'Setup: Initializing...' : 'Setup: Core Hamiltonian computed'}`);
        } else if (stage === 'integrals') {
          logLines.push(`[${elapsed}s] ${iter === 0 ? 'Integrals: Computing ERIs...' : 'Integrals: Done'}`);
        } else if (stage === 'integrals_ri') {
          const labels: Record<number, string> = { 0: '2-center ERIs', 1: 'Cholesky', 2: '3-center ERIs', 3: 'B matrix', 4: 'RI done' };
          logLines.push(`[${elapsed}s] RI: ${labels[iter] || `step ${iter}`}`);
        } else if (stage === 'scf') {
          const E = vals[2] !== undefined ? Number(vals[2]).toFixed(10) : '';
          const dE = vals[1] !== undefined ? Number(vals[1]).toExponential(2) : '';
          logLines.push(`[${elapsed}s] SCF iter ${iter}  E=${E}  ΔE=${dE}`);
        } else if (stage === 'ccsd') {
          const dE = vals[1] !== undefined ? Number(vals[1]).toExponential(2) : '';
          logLines.push(`[${elapsed}s] CCSD iter ${iter}  ΔE=${dE}`);
        } else if (stage === 'ccsd_lambda') {
          logLines.push(`[${elapsed}s] Lambda iter ${iter}  residual=${vals[0] !== undefined ? Number(vals[0]).toExponential(2) : ''}`);
        } else if (stage === 'davidson') {
          logLines.push(`[${elapsed}s] Davidson iter ${iter}  max|r|=${vals[0] !== undefined ? Number(vals[0]).toExponential(2) : ''}`);
        } else {
          logLines.push(`[${elapsed}s] ${stage} iter ${iter}`);
        }
      }

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const parts = buffer.split('\n\n');
        buffer = parts.pop() || '';
        for (const part of parts) {
          const line = part.trim();
          if (!line.startsWith('data: ')) continue;
          try {
            const event = JSON.parse(line.slice(6));

            if (event.type === 'progress') {
              if (lastSeenStage && lastSeenStage !== event.stage) {
                await nextFrame();
              }
              lastSeenStage = event.stage;
              logProgress(event);
              tracker.handleProgress({
                type: 'progress',
                stage: event.stage,
                iteration: event.iteration,
                total_energy: event.values?.[2],
                delta_e: event.values?.[1],
                correlation_energy: event.values?.[0],
                residual: event.values?.[0],
                max_residual: event.values?.[0],
              });
            } else if (event.type === 'result') {
              const elapsed = ((performance.now() - t0) / 1000).toFixed(3);
              const s = event.data.summary || {};
              logLines.push(`[${elapsed}s] Done. Total energy: ${(s.total_energy ?? 0).toFixed(10)} Ha`);
              if (event.data.post_hf) {
                logLines.push(`  Post-HF (${event.data.post_hf.method}): correction=${event.data.post_hf.correction.toFixed(10)}, total=${event.data.post_hf.total_energy.toFixed(10)}`);
              }
              tracker.complete();
              const d = event.data;
              resultsPanel.show({
                ok: true,
                raw_output: logLines.join('\n'),
                molecule: d.molecule || {},
                basis_set: d.basis_set || {},
                scf_iterations: d.scf_iterations || [],
                summary: d.summary || {},
                post_hf: d.post_hf || undefined,
                orbital_energies: d.orbital_energies || [],
                orbital_energies_beta: d.orbital_energies_beta || [],
                mulliken: d.mulliken || [],
                mayer_bond_order: d.mayer_bond_order || [],
                wiberg_bond_order: d.wiberg_bond_order || [],
                timing: d.timing || {},
                excited_states: d.excited_states,
                excited_states_method: d.excited_states_method,
                excited_states_spin: d.excited_states_spin,
              } as CalculationResult);
            } else if (event.type === 'error') {
              logLines.push(`ERROR: ${event.error}`);
              tracker.fail(event.error);
              resultsPanel.showError(event.error, logLines.join('\n'));
            }
          } catch { /* skip malformed */ }
        }
      }
    } catch (e: unknown) {
      if (e instanceof Error && e.name !== 'AbortError') {
        tracker.fail(String(e));
      }
    }

    runBtn.disabled = false;
    cancelBtn.classList.add('hidden');
    abortController = null;
  });

  cancelBtn.addEventListener('click', () => {
    if (abortController) {
      abortController.abort();
      abortController = null;
    }
    runBtn.disabled = false;
    cancelBtn.classList.add('hidden');
  });
}

// Bootstrap
import { initTheme } from './ui/theme';
import './ui/styles.css';

document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initApp(document.getElementById('app')!);
});
