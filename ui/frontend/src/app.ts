/** Main application — orchestrates molecule input, settings, execution, and results */

import { fetchSampleDirs, fetchSamples, fetchBasisSets, fetchAuxiliaryBasisSets, streamCalculation } from './api';
import { DEFAULT_PARAMS } from './types';
import type { CalculationParams, CalculationResult, StreamEvent } from './types';
import { createMoleculePanel } from './ui/moleculePanel';
import { createSettingsPanel } from './ui/settingsPanel';
import { createProgressPanel } from './ui/progressPanel';
import { createResultsPanel } from './ui/resultsPanel';
import { toggleTheme } from './ui/theme';

export async function initApp(root: HTMLElement) {
  // Fetch data from API
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

  // Build layout
  root.innerHTML = `
    <header>
      <h1>GANSU</h1>
      <span class="subtitle">GPU Accelerated Numerical Simulation Utility</span>
      <button id="theme-btn" class="icon-btn" title="Toggle theme">
        <span id="theme-icon">&#9790;</span>
      </button>
    </header>
    <div class="main-grid">
      <div id="molecule-col"></div>
      <div id="settings-col"></div>
    </div>
    <div class="action-bar">
      <button id="run-btn" class="primary-btn">Run Calculation</button>
      <button id="cancel-btn" class="secondary-btn hidden">Cancel</button>
    </div>
    <div id="progress-col"></div>
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
  const progressPanel = createProgressPanel(root.querySelector('#progress-col')!);
  const resultsPanel = createResultsPanel(root.querySelector('#results-col')!);

  const runBtn = root.querySelector<HTMLButtonElement>('#run-btn')!;
  const cancelBtn = root.querySelector<HTMLButtonElement>('#cancel-btn')!;

  let currentController: AbortController | null = null;

  runBtn.addEventListener('click', () => {
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

    // UI state: running
    runBtn.disabled = true;
    cancelBtn.classList.remove('hidden');
    resultsPanel.hide();
    progressPanel.show();
    progressPanel.setStatus('Starting calculation...');

    let iterCount = 0;

    currentController = streamCalculation(params, (event: StreamEvent) => {
      switch (event.type) {
        case 'line':
          progressPanel.addLine(event.text);
          // Detect iteration lines for status update
          if (event.text.includes('Iteration:')) {
            iterCount++;
            progressPanel.setStatus(`SCF Iteration ${iterCount}...`);
          }
          break;
        case 'result':
          progressPanel.setStatus('Calculation complete');
          progressPanel.hide();
          resultsPanel.show(event.data as CalculationResult);
          runBtn.disabled = false;
          cancelBtn.classList.add('hidden');
          currentController = null;
          break;
        case 'error':
          progressPanel.setStatus('Error');
          progressPanel.hide();
          resultsPanel.showError(event.error, event.raw_output);
          runBtn.disabled = false;
          cancelBtn.classList.add('hidden');
          currentController = null;
          break;
        case 'done':
          runBtn.disabled = false;
          cancelBtn.classList.add('hidden');
          currentController = null;
          break;
      }
    });
  });

  cancelBtn.addEventListener('click', () => {
    if (currentController) {
      currentController.abort();
      currentController = null;
    }
    progressPanel.setStatus('Cancelled');
    progressPanel.hide();
    runBtn.disabled = false;
    cancelBtn.classList.add('hidden');
  });
}
