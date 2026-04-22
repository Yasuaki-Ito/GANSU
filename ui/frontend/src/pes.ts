/**
 * GANSU-UI — Potential Energy Scan page.
 * Scans a structural parameter (bond distance or angle) and plots energy vs parameter.
 * Adapted from GANSU-web optimize.ts — computation via GPU backend API.
 */

import './ui/styles.css';
import { initTheme, toggleTheme, getThemeColors } from './ui/theme';
import { renderMoleculePreview, lockMoleculeScale, unlockMoleculeScale, type MolAnnotation } from './viz/moleculeViewer3D';

// ── Scenario definitions (from GANSU-web) ────────────────────────────

interface Scenario {
  id: string;
  label: string;
  description: string;
  paramType: 'bond' | 'angle' | 'height';
  paramLabel: string;
  paramUnit: string;
  defaultMin: number;
  defaultMax: number;
  defaultSteps: number;
  defaultCharge: number;
  defaultBasis: string;
  generateXYZ: (param: number) => string;
  annotations?: MolAnnotation[];  // which measurements to display in 3D
}

const SCENARIOS: Scenario[] = [
  {
    id: 'h2', label: 'H₂', description: 'Hydrogen dissociation',
    paramType: 'bond', paramLabel: 'H-H distance', paramUnit: 'Å',
    defaultMin: 0.4, defaultMax: 5.0, defaultSteps: 24,
    defaultCharge: 0, defaultBasis: 'cc-pvdz',
    generateXYZ: (r) => `2\nH2 R=${r.toFixed(3)}\nH  0.0  0.0  0.0\nH  0.0  0.0  ${r.toFixed(6)}`,
  },
  {
    id: 'hf', label: 'HF', description: 'Hydrogen fluoride dissociation',
    paramType: 'bond', paramLabel: 'H-F distance', paramUnit: 'Å',
    defaultMin: 0.5, defaultMax: 3.0, defaultSteps: 20,
    defaultCharge: 0, defaultBasis: 'sto-3g',
    generateXYZ: (r) => `2\nHF R=${r.toFixed(3)}\nH  0.0  0.0  0.0\nF  0.0  0.0  ${r.toFixed(6)}`,
  },
  {
    id: 'lih', label: 'LiH', description: 'Lithium hydride dissociation',
    paramType: 'bond', paramLabel: 'Li-H distance', paramUnit: 'Å',
    defaultMin: 0.8, defaultMax: 4.0, defaultSteps: 20,
    defaultCharge: 0, defaultBasis: 'sto-3g',
    generateXYZ: (r) => `2\nLiH R=${r.toFixed(3)}\nLi  0.0  0.0  0.0\nH   0.0  0.0  ${r.toFixed(6)}`,
  },
  {
    id: 'n2', label: 'N₂', description: 'Nitrogen dissociation (triple bond)',
    paramType: 'bond', paramLabel: 'N-N distance', paramUnit: 'Å',
    defaultMin: 0.8, defaultMax: 3.0, defaultSteps: 20,
    defaultCharge: 0, defaultBasis: 'sto-3g',
    generateXYZ: (r) => `2\nN2 R=${r.toFixed(3)}\nN  0.0  0.0  0.0\nN  0.0  0.0  ${r.toFixed(6)}`,
  },
  {
    id: 'h2o_angle', label: 'H₂O (angle)', description: 'Water bond angle scan',
    paramType: 'angle', paramLabel: 'H-O-H angle', paramUnit: '°',
    defaultMin: 80, defaultMax: 180, defaultSteps: 20,
    defaultCharge: 0, defaultBasis: 'sto-3g',
    annotations: [{ type: 'angle', atoms: [0, 1, 2] }],  // O=vertex, H-O-H
    generateXYZ: (angleDeg) => {
      const R = 0.96, rad = angleDeg * Math.PI / 180, half = rad / 2;
      const hx = R * Math.sin(half), hz = R * Math.cos(half);
      return `3\nH2O angle=${angleDeg.toFixed(1)}\nO  0.0  0.0  0.0\nH  ${hx.toFixed(6)}  0.0  ${hz.toFixed(6)}\nH  ${(-hx).toFixed(6)}  0.0  ${hz.toFixed(6)}`;
    },
  },
  {
    id: 'h2o_bond', label: 'H₂O (OH stretch)', description: 'Water O-H bond stretch',
    paramType: 'bond', paramLabel: 'O-H distance', paramUnit: 'Å',
    defaultMin: 0.5, defaultMax: 3.0, defaultSteps: 20,
    defaultCharge: 0, defaultBasis: 'sto-3g',
    annotations: [{ type: 'distance', atoms: [0, 2] }],  // O-H₂ being stretched
    generateXYZ: (r) => {
      const s = Math.sin(52 * Math.PI / 180), c = Math.cos(52 * Math.PI / 180);
      return `3\nH2O R=${r.toFixed(3)}\nO  0.0  0.0  0.0\nH  ${(0.96*s).toFixed(6)}  0.0  ${(0.96*c).toFixed(6)}\nH  ${(-r*s).toFixed(6)}  0.0  ${(r*c).toFixed(6)}`;
    },
  },
  {
    id: 'nh3_inversion', label: 'NH₃ (inversion)', description: 'Ammonia umbrella inversion',
    paramType: 'height', paramLabel: 'N height above H₃ plane', paramUnit: 'Å',
    defaultMin: -0.5, defaultMax: 0.5, defaultSteps: 20,
    defaultCharge: 0, defaultBasis: 'sto-3g',
    annotations: [{ type: 'height', atom: 0, planeAtoms: [1, 2, 3] }],  // N above H₃ plane
    generateXYZ: (h) => {
      const RNH = 1.012, r = Math.sqrt(Math.max(0, RNH*RNH - h*h)), s3 = Math.sqrt(3)/2;
      return `4\nNH3 h=${h.toFixed(3)}\nN  0.0  0.0  ${h.toFixed(6)}\nH  ${r.toFixed(6)}  0.0  0.0\nH  ${(-r/2).toFixed(6)}  ${(r*s3).toFixed(6)}  0.0\nH  ${(-r/2).toFixed(6)}  ${(-r*s3).toFixed(6)}  0.0`;
    },
  },
];

// ── API ──────────────────────────────────────────────────────────────

const API_BASE = '';

async function resetPESDensity() {
  await fetch(`${API_BASE}/api/pes/reset`, { method: 'POST' });
}

async function runPESPoint(xyzText: string, basis: string, method: string, charge: number, postHf: string, usePrevDensity: boolean): Promise<{ energy: number; postHfEnergy: number; converged: boolean } | null> {
  try {
    const res = await fetch(`${API_BASE}/api/pes/point`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        xyz_text: xyzText, basis, method, charge,
        post_hf_method: postHf,
        use_prev_density: usePrevDensity,
        timeout: 120,
      }),
    });
    const data = await res.json();
    if (!data.ok) return null;
    return { energy: data.energy, postHfEnergy: data.correction, converged: data.converged ?? true };
  } catch { return null; }
}

// ── SVG Plot ─────────────────────────────────────────────────────────

function renderPESPlot(container: HTMLElement, params: number[], series: { label: string; color: string; data: (number | null)[] }[], paramLabel: string, paramUnit: string, onPointClick?: (index: number) => void, highlightIndex?: number) {
  const tc = getThemeColors();
  const width = 600, height = 360;
  const ml = 72, mr = 20, mt = 30, mb = 40;
  const pw = width - ml - mr, ph = height - mt - mb;

  // Collect valid data points
  const allY: number[] = [];
  for (const s of series) for (const v of s.data) if (v !== null) allY.push(v);
  if (allY.length === 0) { container.innerHTML = '<p>No data</p>'; return; }

  const xMin = params[0], xMax = params[params.length - 1];
  const yMin = Math.min(...allY) - 0.02 * (Math.max(...allY) - Math.min(...allY) || 1);
  const yMax = Math.max(...allY) + 0.02 * (Math.max(...allY) - Math.min(...allY) || 1);

  const toX = (x: number) => ml + ((x - xMin) / (xMax - xMin || 1)) * pw;
  const toY = (y: number) => mt + ph - ((y - yMin) / (yMax - yMin || 1)) * ph;

  let svg = `<svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg" style="display:block;width:100%;max-width:${width}px;height:auto" viewBox="0 0 ${width} ${height}">`;

  // Grid
  const nGridY = 5;
  for (let i = 0; i <= nGridY; i++) {
    const y = yMin + (yMax - yMin) * i / nGridY;
    const yy = toY(y);
    svg += `<line x1="${ml}" y1="${yy}" x2="${ml + pw}" y2="${yy}" stroke="${tc.grid}" stroke-width="0.5"/>`;
    svg += `<text x="${ml - 4}" y="${yy + 3}" text-anchor="end" font-size="9" fill="${tc.dim}">${y.toFixed(4)}</text>`;
  }
  // Axes
  svg += `<line x1="${ml}" y1="${mt}" x2="${ml}" y2="${mt + ph}" stroke="${tc.axis}" stroke-width="1"/>`;
  svg += `<line x1="${ml}" y1="${mt + ph}" x2="${ml + pw}" y2="${mt + ph}" stroke="${tc.axis}" stroke-width="1"/>`;
  // X labels
  const nGridX = Math.min(params.length - 1, 8);
  const xStep = Math.max(1, Math.floor(params.length / nGridX));
  for (let i = 0; i < params.length; i += xStep) {
    const xx = toX(params[i]);
    svg += `<text x="${xx}" y="${mt + ph + 16}" text-anchor="middle" font-size="9" fill="${tc.dim}">${params[i].toFixed(2)}</text>`;
  }

  // Series
  for (const s of series) {
    let path = '';
    for (let i = 0; i < params.length; i++) {
      if (s.data[i] === null) continue;
      const px = toX(params[i]), py = toY(s.data[i]!);
      path += path ? ` L${px},${py}` : `M${px},${py}`;
    }
    svg += `<path d="${path}" fill="none" stroke="${s.color}" stroke-width="2"/>`;
    for (let i = 0; i < params.length; i++) {
      if (s.data[i] === null) continue;
      const cx = toX(params[i]), cy = toY(s.data[i]!);
      svg += `<circle cx="${cx}" cy="${cy}" r="3" fill="${s.color}"/>`;
      if (onPointClick) {
        svg += `<circle cx="${cx}" cy="${cy}" r="8" fill="transparent" style="cursor:pointer" data-point-idx="${i}"/>`;
      }
    }
  }

  // Highlight current slider position
  if (highlightIndex !== undefined && highlightIndex >= 0 && highlightIndex < params.length) {
    const hx = toX(params[highlightIndex]);
    // Vertical guide line
    svg += `<line x1="${hx}" y1="${mt}" x2="${hx}" y2="${mt + ph}" stroke="#ff6600" stroke-width="1" stroke-dasharray="4,3" opacity="0.6"/>`;
    // Highlight circle on each series
    for (const s of series) {
      const val = s.data[highlightIndex];
      if (val !== null) {
        const hy = toY(val);
        svg += `<circle cx="${hx}" cy="${hy}" r="6" fill="none" stroke="#ff6600" stroke-width="2"/>`;
        svg += `<circle cx="${hx}" cy="${hy}" r="3" fill="#ff6600"/>`;
      }
    }
  }

  // Minimum energy marker
  for (const s of series) {
    let minIdx = -1, minVal = Infinity;
    for (let i = 0; i < s.data.length; i++) {
      if (s.data[i] !== null && s.data[i]! < minVal) { minVal = s.data[i]!; minIdx = i; }
    }
    if (minIdx >= 0) {
      const mx = toX(params[minIdx]), my = toY(minVal);
      svg += `<circle cx="${mx}" cy="${my}" r="6" fill="none" stroke="#00cc44" stroke-width="2"/>`;
      svg += `<circle cx="${mx}" cy="${my}" r="3" fill="#00cc44"/>`;
      svg += `<text x="${mx + 8}" y="${my - 6}" font-size="9" fill="#00cc44">min</text>`;
    }
  }

  // Legend
  let lx = ml + 8;
  for (const s of series) {
    svg += `<line x1="${lx}" y1="${mt + 8}" x2="${lx + 16}" y2="${mt + 8}" stroke="${s.color}" stroke-width="2"/>`;
    svg += `<text x="${lx + 20}" y="${mt + 12}" font-size="10" fill="${tc.dim}">${s.label}</text>`;
    lx += 20 + s.label.length * 6 + 16;
  }

  // Axis labels
  svg += `<text x="${ml + pw / 2}" y="${height - 4}" text-anchor="middle" font-size="11" fill="${tc.dim}">${paramLabel} (${paramUnit})</text>`;
  svg += `<text x="12" y="${mt + ph / 2}" text-anchor="middle" font-size="11" fill="${tc.dim}" transform="rotate(-90,12,${mt + ph / 2})">Energy (Hartree)</text>`;

  svg += '</svg>';
  container.innerHTML = svg;

  // Wire up click events on data points
  if (onPointClick) {
    container.querySelectorAll<SVGCircleElement>('circle[data-point-idx]').forEach(el => {
      el.addEventListener('click', () => {
        const idx = parseInt(el.getAttribute('data-point-idx')!, 10);
        onPointClick(idx);
      });
    });
  }
}

// ── Page UI ──────────────────────────────────────────────────────────

function initPES() {
  initTheme();
  const root = document.getElementById('app')!;

  root.innerHTML = `
    <header class="header-top">
      <h1>GANSU</h1>
      <span class="subtitle">Potential Energy Scan</span>
      <nav class="demo-nav">
        <a href="./" class="demo-tab">Calculation</a>
        <a class="demo-tab active">PES</a>
        <a href="./geomopt.html" class="demo-tab">Geometry Opt</a>
      </nav>
      <button id="theme-btn" class="icon-btn" title="Toggle theme"><span id="theme-icon">&#9790;</span></button>
    </header>

    <div class="pes-layout">
      <div class="pes-left">
        <div class="panel pes-scenarios" id="scenarios">
          <h2>Scenarios</h2>
          <div class="scenario-grid" id="scenario-grid"></div>
        </div>

        <div class="panel pes-settings" id="settings-panel">
          <h2>Settings</h2>
          <div class="pes-form">
            <label>Method</label>
            <select id="pes-method">
              <option value="RHF" selected>RHF</option>
              <option value="UHF">UHF</option>
            </select>
            <label>Post-HF</label>
            <select id="pes-posthf">
              <option value="none" selected>None</option>
              <option value="mp2">MP2</option>
              <option value="ccsd">CCSD</option>
              <option value="fci">FCI</option>
            </select>
            <label>Basis</label>
            <select id="pes-basis">
              <option value="sto-3g">STO-3G</option>
              <option value="3-21g">3-21G</option>
              <option value="6-31g">6-31G</option>
              <option value="cc-pvdz" selected>cc-pVDZ</option>
            </select>
            <label id="param-label">Range</label>
            <div class="pes-range">
              <input type="number" id="pes-min" step="0.1" />
              <span>to</span>
              <input type="number" id="pes-max" step="0.1" />
            </div>
            <label>Steps</label>
            <input type="number" id="pes-steps" value="20" min="3" max="100" />
            <div class="pes-actions">
              <button id="pes-run" class="primary-btn">Run Scan</button>
              <button id="pes-cancel" class="secondary-btn hidden">Cancel</button>
            </div>
            <div class="pes-progress" id="pes-progress"></div>
          </div>
        </div>
      </div>

      <div class="pes-right">
        <div class="panel pes-results" id="pes-results">
          <div id="pes-mol-preview" style="width:100%;max-width:360px;aspect-ratio:1;margin:0 auto;"></div>
          <div class="pes-slider-row">
            <input type="range" id="pes-slider" min="0" max="100" value="0" />
            <span id="pes-slider-label" class="pes-slider-label"></span>
          </div>
          <div id="pes-chart"></div>
          <details id="pes-table-details" class="hidden">
            <summary>Data Table</summary>
            <table class="result-table" id="pes-data-table"></table>
          </details>
        </div>
      </div>
    </div>
  `;

  // Theme
  const themeBtn = root.querySelector<HTMLButtonElement>('#theme-btn')!;
  const themeIcon = root.querySelector<HTMLElement>('#theme-icon')!;
  themeBtn.addEventListener('click', () => {
    const next = toggleTheme();
    themeIcon.textContent = next === 'dark' ? '\u2600' : '\u263E';
  });

  // Scenario cards
  const grid = root.querySelector('#scenario-grid')!;
  let activeScenario = SCENARIOS[0];

  function renderScenarioCards() {
    grid.innerHTML = SCENARIOS.map(s => `
      <button class="scenario-card ${s.id === activeScenario.id ? 'active' : ''}" data-id="${s.id}">
        <strong>${s.label}</strong>
        <span>${s.description}</span>
      </button>
    `).join('');
  }
  renderScenarioCards();

  const minInput = root.querySelector<HTMLInputElement>('#pes-min')!;
  const maxInput = root.querySelector<HTMLInputElement>('#pes-max')!;
  const stepsInput = root.querySelector<HTMLInputElement>('#pes-steps')!;
  const basisSelect = root.querySelector<HTMLSelectElement>('#pes-basis')!;
  const methodSelect = root.querySelector<HTMLSelectElement>('#pes-method')!;
  const postHfSelect = root.querySelector<HTMLSelectElement>('#pes-posthf')!;
  const paramLabel = root.querySelector<HTMLElement>('#param-label')!;
  const runBtn = root.querySelector<HTMLButtonElement>('#pes-run')!;
  const cancelBtn = root.querySelector<HTMLButtonElement>('#pes-cancel')!;
  const progressEl = root.querySelector<HTMLElement>('#pes-progress')!;
  const chartEl = root.querySelector<HTMLElement>('#pes-chart')!;
  const tableDetails = root.querySelector<HTMLElement>('#pes-table-details')!;
  const dataTable = root.querySelector<HTMLTableElement>('#pes-data-table')!;

  const molPreview = root.querySelector<HTMLElement>('#pes-mol-preview')!;
  const slider = root.querySelector<HTMLInputElement>('#pes-slider')!;
  const sliderLabel = root.querySelector<HTMLElement>('#pes-slider-label')!;

  function updateMolPreview(paramValue: number) {
    const xyz = activeScenario.generateXYZ(paramValue);
    renderMoleculePreview(molPreview, xyz, activeScenario.annotations);
    sliderLabel.textContent = `${activeScenario.paramLabel} = ${paramValue.toFixed(3)} ${activeScenario.paramUnit}`;
  }

  slider.addEventListener('input', () => {
    const pMin = parseFloat(minInput.value);
    const pMax = parseFloat(maxInput.value);
    const frac = parseInt(slider.value) / 100;
    const paramValue = pMin + (pMax - pMin) * frac;
    updateMolPreview(paramValue);
    // Find nearest scan point for highlight
    if (scanParams.length > 0 && scanDone) {
      let bestIdx = 0, bestDist = Infinity;
      for (let i = 0; i < scanParams.length; i++) {
        const d = Math.abs(scanParams[i] - paramValue);
        if (d < bestDist) { bestDist = d; bestIdx = i; }
      }
      rerenderPlotWithHighlight(bestIdx);
    }
  });

  function selectScenario(s: Scenario) {
    activeScenario = s;
    minInput.value = String(s.defaultMin);
    maxInput.value = String(s.defaultMax);
    stepsInput.value = String(s.defaultSteps);
    basisSelect.value = s.defaultBasis;
    paramLabel.textContent = `${s.paramLabel} (${s.paramUnit})`;
    renderScenarioCards();
    // Initial 3D preview at default min
    updateMolPreview(s.defaultMin);
    slider.value = '0';
  }
  selectScenario(SCENARIOS[0]);

  grid.addEventListener('click', (e) => {
    const card = (e.target as HTMLElement).closest('.scenario-card') as HTMLElement;
    if (!card) return;
    const s = SCENARIOS.find(sc => sc.id === card.dataset.id);
    if (s) selectScenario(s);
  });

  // Run scan
  let cancelled = false;
  let scanParams: number[] = [];  // saved for post-scan interaction
  let scanEnergies: (number | null)[] = [];
  let scanSeriesLabel = '';
  let scanSeriesColor = '#3b82f6';
  let scanDone = false;
  function rerenderPlotWithHighlight(idx: number) {
    if (scanParams.length === 0) return;
    renderPESPlot(chartEl, scanParams,
      [{ label: scanSeriesLabel, color: scanSeriesColor, data: scanEnergies }],
      activeScenario.paramLabel, activeScenario.paramUnit,
      scanDone ? onChartPointClick : undefined, idx);
  }

  function onChartPointClick(idx: number) {
    if (idx < 0 || idx >= scanParams.length) return;
    const paramValue = scanParams[idx];
    const pMin = parseFloat(minInput.value);
    const pMax = parseFloat(maxInput.value);
    const frac = (paramValue - pMin) / (pMax - pMin) * 100;
    slider.value = String(Math.round(frac));
    updateMolPreview(paramValue);
    rerenderPlotWithHighlight(idx);
  }

  runBtn.addEventListener('click', async () => {
    const pMin = parseFloat(minInput.value);
    const pMax = parseFloat(maxInput.value);
    const steps = parseInt(stepsInput.value);
    const basis = basisSelect.value;
    const method = methodSelect.value;
    const postHf = postHfSelect.value;

    const params: number[] = [];
    for (let i = 0; i < steps; i++) params.push(pMin + (pMax - pMin) * i / (steps - 1));
    scanParams = params;
    scanEnergies = new Array(steps).fill(null);
    scanSeriesLabel = postHf !== 'none' ? `${method}/${postHf.toUpperCase()}` : method;
    scanSeriesColor = '#3b82f6';
    scanDone = false;
    cancelled = false;
    runBtn.disabled = true;
    cancelBtn.classList.remove('hidden');
    chartEl.innerHTML = '';
    tableDetails.classList.add('hidden');

    // Unlock 3D scale so it auto-adjusts during scan
    unlockMoleculeScale(molPreview);

    // Reset density cache for new scan
    await resetPESDensity();

    for (let i = 0; i < steps; i++) {
      if (cancelled) break;
      progressEl.textContent = `Point ${i + 1}/${steps}: ${activeScenario.paramLabel} = ${params[i].toFixed(3)} ${activeScenario.paramUnit}${i > 0 ? ' (density reuse)' : ''}`;

      const xyz = activeScenario.generateXYZ(params[i]);
      // Sync slider and 3D preview with current scan point
      const sliderFrac = (params[i] - pMin) / (pMax - pMin) * 100;
      slider.value = String(Math.round(sliderFrac));
      updateMolPreview(params[i]);

      let result = await runPESPoint(xyz, basis, method, activeScenario.defaultCharge, postHf, i > 0);

      if (result) {
        if (!result.converged) {
          progressEl.textContent += ' (not converged, skipped)';
          scanEnergies[i] = null;
        } else {
          if (i > 0 && scanEnergies[i - 1] !== null) {
            const jump = Math.abs((result.energy + result.postHfEnergy) - scanEnergies[i - 1]!);
            if (jump > 0.5) {
              progressEl.textContent += ' (jump, retrying)';
              await resetPESDensity();
              result = await runPESPoint(xyz, basis, method, activeScenario.defaultCharge, postHf, false);
              if (result && result.converged) {
                scanEnergies[i] = result.energy + result.postHfEnergy;
              }
            } else {
              scanEnergies[i] = result.energy + result.postHfEnergy;
            }
          } else {
            scanEnergies[i] = result.energy + result.postHfEnergy;
          }
        }
      }

      // Live plot update with current point highlighted
      renderPESPlot(chartEl, params,
        [{ label: scanSeriesLabel, color: scanSeriesColor, data: scanEnergies }],
        activeScenario.paramLabel, activeScenario.paramUnit, undefined, i);
    }

    progressEl.textContent = cancelled ? 'Cancelled' : 'Done';
    runBtn.disabled = false;
    cancelBtn.classList.add('hidden');
    scanDone = true;

    // Lock 3D scale so slider interaction keeps fixed zoom
    lockMoleculeScale(molPreview);

    // Final render with clickable points, highlight last point
    rerenderPlotWithHighlight(steps - 1);

    // Data table
    tableDetails.classList.remove('hidden');
    let tableHtml = `<tr><th>${activeScenario.paramLabel} (${activeScenario.paramUnit})</th><th>Energy (Hartree)</th></tr>`;
    for (let i = 0; i < steps; i++) {
      tableHtml += `<tr><td>${params[i].toFixed(4)}</td><td>${scanEnergies[i] !== null ? scanEnergies[i]!.toFixed(10) : 'FAILED'}</td></tr>`;
    }
    dataTable.innerHTML = tableHtml;
  });

  cancelBtn.addEventListener('click', () => { cancelled = true; });
}

initPES();
