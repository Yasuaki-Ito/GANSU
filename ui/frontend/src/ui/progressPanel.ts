/** Progress panel — real-time structured progress display with convergence graph */

import type { ProgressEvent } from '../types';
import { renderConvergenceGraph } from '../viz/convergenceGraph';

export function createProgressPanel(container: HTMLElement): {
  show: () => void;
  hide: () => void;
  addLine: (text: string) => void;
  addProgress: (event: ProgressEvent) => void;
  clear: () => void;
  setStatus: (status: string) => void;
} {
  container.innerHTML = `
    <div class="panel progress-panel hidden" id="progress-panel">
      <h2>Running...</h2>
      <div class="progress-status" id="progress-status"></div>
      <div class="progress-stages" id="progress-stages">
        <div class="progress-stage" id="stage-scf" style="display:none">
          <h3>SCF</h3>
          <table class="progress-table" id="scf-table">
            <thead><tr><th>Iter</th><th>Energy (Ha)</th><th>ΔE</th></tr></thead>
            <tbody id="scf-tbody"></tbody>
          </table>
          <div id="scf-graph"></div>
        </div>
        <div class="progress-stage" id="stage-ccsd" style="display:none">
          <h3>CCSD</h3>
          <table class="progress-table" id="ccsd-table">
            <thead><tr><th>Iter</th><th>E_corr (Ha)</th><th>ΔE</th></tr></thead>
            <tbody id="ccsd-tbody"></tbody>
          </table>
        </div>
        <div class="progress-stage" id="stage-davidson" style="display:none">
          <h3>Davidson</h3>
          <table class="progress-table" id="davidson-table">
            <thead><tr><th>Iter</th><th>Eigenvalues</th><th>max|r|</th></tr></thead>
            <tbody id="davidson-tbody"></tbody>
          </table>
        </div>
        <div class="progress-stage" id="stage-lambda" style="display:none">
          <h3>CCSD Lambda</h3>
          <table class="progress-table" id="lambda-table">
            <thead><tr><th>Iter</th><th>||Δλ||</th></tr></thead>
            <tbody id="lambda-tbody"></tbody>
          </table>
        </div>
      </div>
      <div class="progress-output" id="progress-output" style="display:none"></div>
    </div>
  `;

  const panel = container.querySelector<HTMLElement>('#progress-panel')!;
  const status = container.querySelector<HTMLElement>('#progress-status')!;
  const output = container.querySelector<HTMLElement>('#progress-output')!;
  const scfStage = container.querySelector<HTMLElement>('#stage-scf')!;
  const ccsdStage = container.querySelector<HTMLElement>('#stage-ccsd')!;
  const davidsonStage = container.querySelector<HTMLElement>('#stage-davidson')!;
  const lambdaStage = container.querySelector<HTMLElement>('#stage-lambda')!;
  const scfTbody = container.querySelector<HTMLElement>('#scf-tbody')!;
  const ccsdTbody = container.querySelector<HTMLElement>('#ccsd-tbody')!;
  const davidsonTbody = container.querySelector<HTMLElement>('#davidson-tbody')!;
  const lambdaTbody = container.querySelector<HTMLElement>('#lambda-tbody')!;
  const scfGraph = container.querySelector<HTMLElement>('#scf-graph')!;

  const scfData: Array<{ iter: number; deltaE: number }> = [];

  function addProgressRow(tbody: HTMLElement, cells: string[]) {
    const row = document.createElement('tr');
    row.innerHTML = cells.map(c => `<td>${c}</td>`).join('');
    tbody.appendChild(row);
    // Keep last 20 visible, scroll
    const parent = tbody.closest('.progress-stage');
    if (parent) parent.scrollTop = parent.scrollHeight;
  }

  return {
    show: () => {
      panel.classList.remove('hidden');
      status.textContent = '';
      output.innerHTML = '';
      scfTbody.innerHTML = '';
      ccsdTbody.innerHTML = '';
      davidsonTbody.innerHTML = '';
      lambdaTbody.innerHTML = '';
      scfGraph.innerHTML = '';
      scfData.length = 0;
      scfStage.style.display = 'none';
      ccsdStage.style.display = 'none';
      davidsonStage.style.display = 'none';
      lambdaStage.style.display = 'none';
      output.style.display = 'none';
    },
    hide: () => panel.classList.add('hidden'),
    addLine: (text: string) => {
      output.style.display = 'block';
      const line = document.createElement('div');
      line.className = 'output-line';
      line.textContent = text;
      output.appendChild(line);
      output.scrollTop = output.scrollHeight;
    },
    addProgress: (event: ProgressEvent) => {
      if (event.stage === 'scf') {
        scfStage.style.display = 'block';
        const e = event.total_energy ?? 0;
        const de = event.delta_e ?? 0;
        status.textContent = `SCF Iteration ${event.iteration}`;
        if (event.iteration > 0) {
          addProgressRow(scfTbody, [
            `${event.iteration}`,
            e.toFixed(10),
            de.toExponential(2),
          ]);
          scfData.push({ iter: event.iteration, deltaE: de });
          if (scfData.length >= 2) {
            renderConvergenceGraph(scfGraph, scfData, 1e-6);
          }
        }
      } else if (event.stage === 'ccsd') {
        ccsdStage.style.display = 'block';
        status.textContent = `CCSD Iteration ${event.iteration}`;
        addProgressRow(ccsdTbody, [
          `${event.iteration}`,
          (event.correlation_energy ?? 0).toFixed(12),
          (event.delta_e ?? 0).toExponential(4),
        ]);
      } else if (event.stage === 'davidson') {
        davidsonStage.style.display = 'block';
        const evals = event.eigenvalues ?? [];
        status.textContent = `Davidson Iteration ${event.iteration}`;
        addProgressRow(davidsonTbody, [
          `${event.iteration}`,
          evals.slice(0, 5).map(v => v.toFixed(4)).join(', '),
          (event.max_residual ?? 0).toExponential(2),
        ]);
      } else if (event.stage === 'ccsd_lambda') {
        lambdaStage.style.display = 'block';
        status.textContent = `Lambda Iteration ${event.iteration}`;
        addProgressRow(lambdaTbody, [
          `${event.iteration}`,
          (event.residual ?? 0).toExponential(3),
        ]);
      }
    },
    clear: () => { output.innerHTML = ''; },
    setStatus: (s: string) => { status.textContent = s; },
  };
}
