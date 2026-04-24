/** Results panel — display parsed calculation results */

import type { CalculationResult, OrbitalEnergy } from '../types';
import { renderConvergenceGraph } from '../viz/convergenceGraph';
import { renderOrbitalDiagram, renderUHFOrbitalDiagram } from '../viz/orbitalDiagram';
import { renderSpectrumChart } from '../viz/spectrumChart';

export function createResultsPanel(container: HTMLElement): {
  show: (result: CalculationResult) => void;
  showError: (error: string, rawOutput: string) => void;
  hide: () => void;
} {
  container.innerHTML = '<div id="results-container"></div>';
  const el = container.querySelector<HTMLElement>('#results-container')!;

  function show(result: CalculationResult) {
    let html = '<div class="results">';

    // Molecule & Basis Summary (side by side)
    const mol = result.molecule;
    const bas = result.basis_set;
    if (mol.num_atoms !== undefined || bas.num_basis !== undefined) {
      html += '<div class="summary-row">';
      if (mol.num_atoms !== undefined) {
        html += `
          <div class="panel result-card">
            <h3>Molecule</h3>
            <table class="result-table">
              ${row('Atoms', String(mol.num_atoms))}
              ${mol.num_electrons !== undefined ? row('Electrons', String(mol.num_electrons)) : ''}
              ${mol.alpha_electrons !== undefined ? subrow('Alpha', String(mol.alpha_electrons)) : ''}
              ${mol.beta_electrons !== undefined ? subrow('Beta', String(mol.beta_electrons)) : ''}
            </table>
          </div>`;
      }
      if (bas.num_basis !== undefined) {
        html += `
          <div class="panel result-card">
            <h3>Basis Set</h3>
            <table class="result-table">
              ${row('Basis Functions', String(bas.num_basis))}
              ${bas.num_primitives !== undefined ? subrow('Primitives', String(bas.num_primitives)) : ''}
              ${bas.num_auxiliary !== undefined ? row('Auxiliary', String(bas.num_auxiliary)) : ''}
            </table>
          </div>`;
      }
      html += '</div>';
    }

    // Energy Summary
    if (result.summary.total_energy !== undefined) {
      html += `
        <div class="panel result-card">
          <h3>Energy Summary</h3>
          <table class="result-table">
            ${row('Method', result.summary.method || '-')}
            ${row('Total Energy', formatEnergy(result.summary.total_energy) + ' Hartree')}
            ${result.summary.electronic_energy !== undefined ? row('Electronic Energy', formatEnergy(result.summary.electronic_energy) + ' Hartree') : ''}
            ${result.summary.iterations !== undefined ? row('SCF Iterations', String(result.summary.iterations)) : ''}
            ${result.summary.convergence_algorithm ? row('Convergence', result.summary.convergence_algorithm) : ''}
            ${result.summary.initial_guess ? row('Initial Guess', result.summary.initial_guess) : ''}
            ${result.summary.energy_difference !== undefined ? row('Final |deltaE|', result.summary.energy_difference.toExponential(2)) : ''}
            ${result.summary.computing_time_ms !== undefined ? row('Computing Time', result.summary.computing_time_ms.toFixed(1) + ' ms') : ''}
          </table>
        </div>`;
    }

    // Post-HF (hide when only method is set, e.g. excited state methods)
    if (result.post_hf && (result.post_hf.correction !== undefined || result.post_hf.total_energy !== undefined)) {
      const mol = result.molecule as any;
      const nfrozen = mol?.num_frozen || 0;
      const nocc = mol?.num_occ;
      const nvir = mol?.num_vir;
      const activeOcc = nfrozen > 0 ? nocc - nfrozen : nocc;
      html += `
        <div class="panel result-card">
          <h3>Post-HF: ${result.post_hf.method || ''}${nfrozen > 0 ? ' (frozen core)' : ''}</h3>
          <table class="result-table">
            ${nocc !== undefined ? row('Occupied / Virtual', `${nocc} / ${nvir}`) : ''}
            ${nfrozen > 0 ? row('Frozen Core', `${nfrozen} orbitals frozen, ${activeOcc} active occupied`) : ''}
            ${result.post_hf.correction !== undefined ? row('Correlation Energy', formatEnergy(result.post_hf.correction) + ' Hartree') : ''}
            ${result.post_hf.total_energy !== undefined ? row('Total Energy', formatEnergy(result.post_hf.total_energy) + ' Hartree') : ''}
          </table>
        </div>`;
    }

    // Excited States Spectrum (full width)
    if (result.excited_states && result.excited_states.length > 0) {
      html += '<div class="panel result-card full-width" id="spectrum-chart-container"></div>';
    }

    // Orbital Energies — diagram + collapsible table
    if (result.orbital_energies.length > 0) {
      const hasBeta = result.orbital_energies_beta.length > 0;
      html += '<div class="panel result-card orbital-card" id="orbital-diagram-container"><h3>Orbital Energies</h3></div>';

      // Collapsible table
      if (hasBeta) {
        html += `
          <details class="panel result-card">
            <summary>Orbital Energies (Alpha) — Table</summary>
            ${renderOrbitalTable(result.orbital_energies)}
          </details>
          <details class="panel result-card">
            <summary>Orbital Energies (Beta) — Table</summary>
            ${renderOrbitalTable(result.orbital_energies_beta)}
          </details>`;
      } else {
        html += `
          <details class="panel result-card">
            <summary>Orbital Energies — Table</summary>
            ${renderOrbitalTable(result.orbital_energies)}
          </details>`;
      }
    }

    // Atom labels for tables
    const atoms = result.molecule.atoms || [];
    const atomLabel = (i: number) => {
      const a = atoms[i];
      return a ? `${a.element}${i + 1}` : String(i);
    };

    // Mulliken Population
    if (result.mulliken.length > 0) {
      html += `
        <div class="panel result-card">
          <h3>Mulliken Population</h3>
          <table class="result-table">
            <tr><th>Atom</th><th>Charge</th></tr>
            ${result.mulliken.map((m, i) =>
              `<tr><td>${atomLabel(i)}</td><td>${m.charge.toFixed(6)}</td></tr>`
            ).join('')}
          </table>
        </div>`;
    }

    // Mayer Bond Order
    if (result.mayer_bond_order.length > 0) {
      html += `
        <div class="panel result-card">
          <h3>Mayer Bond Order</h3>
          ${renderBondMatrix(result.mayer_bond_order, atomLabel)}
        </div>`;
    }

    // Wiberg Bond Order
    if (result.wiberg_bond_order.length > 0) {
      html += `
        <div class="panel result-card">
          <h3>Wiberg Bond Order</h3>
          ${renderBondMatrix(result.wiberg_bond_order, atomLabel)}
        </div>`;
    }

    // Convergence Graph
    html += '<div class="panel result-card" id="conv-graph-container"></div>';

    // Molden download
    if (result.molden_content) {
      html += `
        <div class="panel result-card">
          <h3>Molden</h3>
          <button class="secondary-btn" id="download-molden">Download output.molden</button>
        </div>`;
    }

    // Raw Output
    html += `
      <details class="panel result-card">
        <summary>Raw Output</summary>
        <pre class="raw-output">${escapeHtml(result.raw_output)}</pre>
      </details>`;

    html += '</div>';
    el.innerHTML = html;

    // Molden download handler — auto-download + manual button
    if (result.molden_content) {
      const downloadMolden = () => {
        const blob = new Blob([result.molden_content!], { type: 'chemical/x-molden' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'output.molden';
        a.click();
        URL.revokeObjectURL(url);
      };
      // Auto-download
      downloadMolden();
      // Also attach to button for re-download
      const dlBtn = el.querySelector<HTMLButtonElement>('#download-molden');
      dlBtn?.addEventListener('click', downloadMolden);
    }

    // Render orbital diagram
    const orbContainer = el.querySelector<HTMLElement>('#orbital-diagram-container');
    if (orbContainer && result.orbital_energies.length > 0) {
      const hasBeta = result.orbital_energies_beta.length > 0;
      if (hasBeta) {
        renderUHFOrbitalDiagram(orbContainer, result.orbital_energies, result.orbital_energies_beta);
      } else {
        renderOrbitalDiagram(orbContainer, result.orbital_energies);
      }
    }

    // Render convergence graph
    const graphContainer = el.querySelector<HTMLElement>('#conv-graph-container');
    if (graphContainer && result.scf_iterations.length > 0) {
      const iters = result.scf_iterations
        .filter(it => it.delta_e !== undefined)
        .map(it => ({ iter: it.iteration, deltaE: it.delta_e! }));
      const threshold = result.summary.convergence_criterion || 1e-6;
      renderConvergenceGraph(graphContainer, iters, threshold);
    }

    // Render spectrum chart
    const spectrumContainer = el.querySelector<HTMLElement>('#spectrum-chart-container');
    if (spectrumContainer && result.excited_states && result.excited_states.length > 0) {
      renderSpectrumChart(
        spectrumContainer,
        result.excited_states,
        result.excited_states_method || '',
        result.excited_states_spin || 'singlet',
      );
    }
  }

  function showError(error: string, rawOutput: string) {
    let html = '<div class="results">';
    html += `
      <div class="panel result-card error-card">
        <h3>Error</h3>
        <pre class="error-output">${escapeHtml(error)}</pre>
      </div>`;
    if (rawOutput) {
      html += `
        <details class="panel result-card" open>
          <summary>Output</summary>
          <pre class="raw-output">${escapeHtml(rawOutput)}</pre>
        </details>`;
    }
    html += '</div>';
    el.innerHTML = html;
  }

  return {
    show,
    showError,
    hide: () => { el.innerHTML = ''; },
  };
}

function row(label: string, value: string): string {
  return `<tr><td class="label">${label}</td><td class="value">${value}</td></tr>`;
}

function subrow(label: string, value: string): string {
  return `<tr><td class="label sub-label">${label}</td><td class="value">${value}</td></tr>`;
}

function formatEnergy(e: number): string {
  return e.toFixed(10);
}

function renderBondMatrix(matrix: number[][], atomLabel: (i: number) => string): string {
  if (matrix.length === 0) return '';
  const n = matrix.length;
  let html = '<table class="result-table bond-table"><tr><th></th>';
  for (let i = 0; i < n; i++) html += `<th>${atomLabel(i)}</th>`;
  html += '</tr>';
  for (let i = 0; i < n; i++) {
    html += `<tr><th>${atomLabel(i)}</th>`;
    for (let j = 0; j < matrix[i].length; j++) {
      const v = matrix[i][j];
      const cls = v > 0.5 ? 'bond-strong' : '';
      html += `<td class="${cls}">${v.toFixed(3)}</td>`;
    }
    html += '</tr>';
  }
  html += '</table>';
  return html;
}

function renderOrbitalTable(orbitals: OrbitalEnergy[]): string {
  const occLabel: Record<string, string> = {
    occ: 'Occupied', vir: 'Virtual', closed: 'Closed', open: 'Open', '?': '?',
  };
  let html = '<table class="result-table"><tr><th>#</th><th>Occupation</th><th>Energy (Hartree)</th><th>Energy (eV)</th></tr>';
  for (const o of orbitals) {
    html += `<tr><td>${o.index}</td><td>${occLabel[o.occupation] || o.occupation}</td><td>${o.energy.toFixed(6)}</td><td>${(o.energy * 27.2114).toFixed(4)}</td></tr>`;
  }
  html += '</table>';
  return html;
}

function stripAnsi(text: string): string {
  return text.replace(/\x1b\[[0-9;]*m/g, '');
}

function escapeHtml(text: string): string {
  return stripAnsi(text).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
