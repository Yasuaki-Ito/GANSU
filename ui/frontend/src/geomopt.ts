/**
 * GANSU-UI — Geometry Optimization page.
 * Streams optimization steps in real-time, plotting energy and gradient convergence.
 * Force vectors shown on 3D viewer; slider to scrub through steps.
 */

import './ui/styles.css';
import { initTheme, toggleTheme, getThemeColors } from './ui/theme';
import { renderMoleculePreview, type MolAnnotation } from './viz/moleculeViewer3D';
import { fetchSamples, fetchSampleContent, fetchBasisSets } from './api';

// ── Types ───────────────────────────────────────────────────────────

interface OptStep {
  step: number;
  energy: number;
  maxGrad: number;
  rmsGrad: number;
}

interface OptAtom {
  element: string;
  x: number; y: number; z: number;
}

interface StepGeom {
  atoms: OptAtom[];
  forces: { x: number; y: number; z: number }[];
}

// ── Distort XYZ ─────────────────────────────────────────────────────

function distortXYZ(xyzText: string, magnitude: number): string {
  const lines = xyzText.split(/\r?\n/);
  if (lines.length < 3) return xyzText;
  const n = parseInt(lines[0].trim(), 10);
  if (isNaN(n) || n < 1) return xyzText;
  const out = [lines[0], lines[1]];
  for (let i = 2; i < 2 + n && i < lines.length; i++) {
    const parts = lines[i].trim().split(/\s+/);
    if (parts.length < 4) { out.push(lines[i]); continue; }
    const x = parseFloat(parts[1]) + (Math.random() - 0.5) * 2 * magnitude;
    const y = parseFloat(parts[2]) + (Math.random() - 0.5) * 2 * magnitude;
    const z = parseFloat(parts[3]) + (Math.random() - 0.5) * 2 * magnitude;
    out.push(`${parts[0]}  ${x.toFixed(6)}  ${y.toFixed(6)}  ${z.toFixed(6)}`);
  }
  return out.join('\n');
}

// ── Parse optimization output ───────────────────────────────────────

const bohrToAng = 0.529177249;

function parseOptSteps(text: string): OptStep[] {
  const steps: OptStep[] = [];
  const re = /--- Geometry Optimization Step (\d+) ---[\s\S]*?Energy:\s*([-+]?\d+\.\d+)\s+Hartree[\s\S]*?Max gradient:\s*([-+]?\d+\.\d+[eE][-+]?\d+)[\s\S]*?RMS gradient:\s*([-+]?\d+\.\d+[eE][-+]?\d+)/g;
  let m;
  while ((m = re.exec(text)) !== null) {
    steps.push({
      step: parseInt(m[1]),
      energy: parseFloat(m[2]),
      maxGrad: parseFloat(m[3]),
      rmsGrad: parseFloat(m[4]),
    });
  }
  return steps;
}

function parseStepGeometries(text: string, expectedAtoms: number): Map<number, StepGeom> {
  const map = new Map<number, StepGeom>();
  const blockRe = /\[Geometry Step (\d+)\]\s*\n([\s\S]*?)(?=\n\s*\n|\n---|\n\[|\nWARNING|\n=)/g;
  let m;
  while ((m = blockRe.exec(text)) !== null) {
    const step = parseInt(m[1]);
    const lines = m[2].trim().split('\n');
    const atoms: OptAtom[] = [];
    const forces: { x: number; y: number; z: number }[] = [];
    for (const line of lines) {
      const parts = line.trim().split(/\s+/);
      if (parts.length >= 7) {
        atoms.push({
          element: parts[0],
          x: parseFloat(parts[1]) * bohrToAng,
          y: parseFloat(parts[2]) * bohrToAng,
          z: parseFloat(parts[3]) * bohrToAng,
        });
        forces.push({
          x: parseFloat(parts[4]) * bohrToAng,
          y: parseFloat(parts[5]) * bohrToAng,
          z: parseFloat(parts[6]) * bohrToAng,
        });
      }
    }
    // Only accept complete blocks (all atoms parsed)
    if (expectedAtoms > 0 && atoms.length !== expectedAtoms) continue;
    if (atoms.length > 0) map.set(step, { atoms, forces });
  }
  return map;
}

function parseOptimizedGeometry(text: string): OptAtom[] {
  const atoms: OptAtom[] = [];
  const block = text.match(/Optimized Geometry \(Bohr\):\s*\n([\s\S]*?)(?:\n\s*\n|\n\[|$)/);
  if (!block) return atoms;
  const lines = block[1].trim().split('\n');
  for (const line of lines) {
    const parts = line.trim().split(/\s+/);
    if (parts.length >= 4) {
      atoms.push({
        element: parts[0],
        x: parseFloat(parts[1]) * bohrToAng,
        y: parseFloat(parts[2]) * bohrToAng,
        z: parseFloat(parts[3]) * bohrToAng,
      });
    }
  }
  return atoms;
}

function atomsToXYZ(atoms: OptAtom[]): string {
  const lines = [`${atoms.length}`, 'geometry'];
  for (const a of atoms) {
    lines.push(`${a.element}  ${a.x.toFixed(6)}  ${a.y.toFixed(6)}  ${a.z.toFixed(6)}`);
  }
  return lines.join('\n');
}

// ── SVG Plot with highlight ─────────────────────────────────────────

function renderLinePlot(
  container: HTMLElement,
  xData: number[], yData: number[],
  xLabel: string, yLabel: string,
  color: string, logScale: boolean = false,
  highlightIdx?: number,
) {
  const tc = getThemeColors();
  const width = 500, height = 260;
  const ml = 72, mr = 16, mt = 20, mb = 36;
  const pw = width - ml - mr, ph = height - mt - mb;

  if (xData.length === 0) { container.innerHTML = ''; return; }

  const xMin = Math.min(...xData), xMax = Math.max(...xData);
  let yVals = yData;
  if (logScale) yVals = yData.map(v => v > 0 ? Math.log10(v) : -12);
  const yMin = Math.min(...yVals) - 0.05 * (Math.max(...yVals) - Math.min(...yVals) || 1);
  const yMax = Math.max(...yVals) + 0.05 * (Math.max(...yVals) - Math.min(...yVals) || 1);

  const toX = (x: number) => ml + ((x - xMin) / (xMax - xMin || 1)) * pw;
  const toY = (y: number) => mt + ph - ((y - yMin) / (yMax - yMin || 1)) * ph;

  let svg = `<svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg" style="display:block;width:100%;max-width:${width}px;height:auto" viewBox="0 0 ${width} ${height}">`;

  for (let i = 0; i <= 4; i++) {
    const y = yMin + (yMax - yMin) * i / 4;
    const yy = toY(y);
    svg += `<line x1="${ml}" y1="${yy}" x2="${ml + pw}" y2="${yy}" stroke="${tc.grid}" stroke-width="0.5"/>`;
    const label = logScale ? `1e${y.toFixed(0)}` : y.toFixed(6);
    svg += `<text x="${ml - 4}" y="${yy + 3}" text-anchor="end" font-size="8" fill="${tc.dim}">${label}</text>`;
  }

  svg += `<line x1="${ml}" y1="${mt}" x2="${ml}" y2="${mt + ph}" stroke="${tc.axis}" stroke-width="1"/>`;
  svg += `<line x1="${ml}" y1="${mt + ph}" x2="${ml + pw}" y2="${mt + ph}" stroke="${tc.axis}" stroke-width="1"/>`;

  const nLabels = Math.min(xData.length, 10);
  const step = Math.max(1, Math.floor(xData.length / nLabels));
  for (let i = 0; i < xData.length; i += step) {
    svg += `<text x="${toX(xData[i])}" y="${mt + ph + 14}" text-anchor="middle" font-size="8" fill="${tc.dim}">${xData[i]}</text>`;
  }

  let path = '';
  for (let i = 0; i < xData.length; i++) {
    const px = toX(xData[i]), py = toY(yVals[i]);
    path += path ? ` L${px},${py}` : `M${px},${py}`;
  }
  svg += `<path d="${path}" fill="none" stroke="${color}" stroke-width="2"/>`;
  for (let i = 0; i < xData.length; i++) {
    svg += `<circle cx="${toX(xData[i])}" cy="${toY(yVals[i])}" r="3" fill="${color}"/>`;
  }

  // Highlight
  if (highlightIdx !== undefined && highlightIdx >= 0 && highlightIdx < xData.length) {
    const hx = toX(xData[highlightIdx]), hy = toY(yVals[highlightIdx]);
    svg += `<line x1="${hx}" y1="${mt}" x2="${hx}" y2="${mt + ph}" stroke="#ff6600" stroke-width="1" stroke-dasharray="4,3" opacity="0.6"/>`;
    svg += `<circle cx="${hx}" cy="${hy}" r="6" fill="none" stroke="#ff6600" stroke-width="2"/>`;
    svg += `<circle cx="${hx}" cy="${hy}" r="3" fill="#ff6600"/>`;
  }

  svg += `<text x="${ml + pw / 2}" y="${height - 4}" text-anchor="middle" font-size="10" fill="${tc.dim}">${xLabel}</text>`;
  svg += `<text x="10" y="${mt + ph / 2}" text-anchor="middle" font-size="10" fill="${tc.dim}" transform="rotate(-90,10,${mt + ph / 2})">${yLabel}</text>`;

  svg += '</svg>';
  container.innerHTML = svg;
}

function stripAnsi(s: string): string {
  return s.replace(/\x1b\[[0-9;]*m/g, '');
}

// ── Page UI ─────────────────────────────────────────────────────────

async function initGeomOpt() {
  initTheme();
  const root = document.getElementById('app')!;

  root.innerHTML = `
    <header class="header-top">
      <h1>GANSU</h1>
      <span class="subtitle">Geometry Optimization</span>
      <nav class="demo-nav">
        <a href="./" class="demo-tab">Calculation</a>
        <a href="./pes.html" class="demo-tab">PES</a>
        <a class="demo-tab active">Geometry Opt</a>
      </nav>
      <button id="theme-btn" class="icon-btn" title="Toggle theme"><span id="theme-icon">&#9790;</span></button>
    </header>

    <div class="geomopt-layout">
      <div class="geomopt-left">
        <div class="panel">
          <h2>Molecule</h2>
          <div class="pes-form">
            <label>Sample molecules</label>
            <select id="go-sample"><option value="">-- Select --</option></select>
            <label>XYZ</label>
            <textarea id="go-xyz" rows="8" style="font-family:monospace;font-size:0.75rem;width:100%;resize:vertical;background:var(--color-input);color:var(--color-text);border:1px solid var(--color-border-input);border-radius:5px;padding:6px"></textarea>
            <label>Distortion: <span id="go-distort-label">0.15 &#197;</span></label>
            <input type="range" id="go-distort-mag" min="1" max="50" value="15" style="width:100%" />
            <button id="go-distort" class="secondary-btn" style="margin-top:4px;width:100%">Distort</button>
          </div>
        </div>

        <div class="panel">
          <h2>Settings</h2>
          <div class="pes-form">
            <label>Method</label>
            <select id="go-method">
              <option value="RHF" selected>RHF</option>
              <option value="UHF">UHF</option>
            </select>
            <label>Basis</label>
            <select id="go-basis"></select>
            <label>Optimizer</label>
            <select id="go-optimizer">
              <option value="bfgs" selected>BFGS</option>
              <option value="dfp">DFP</option>
              <option value="sr1">SR1</option>
              <option value="gdiis">GDIIS</option>
              <option value="cg-pr">CG (Polak-Ribi&egrave;re)</option>
              <option value="cg-fr">CG (Fletcher-Reeves)</option>
              <option value="sd">Steepest Descent</option>
              <option value="newton">Newton</option>
            </select>
            <label>Initial guess</label>
            <select id="go-guess">
              <option value="sad" selected>SAD</option>
              <option value="core">Core</option>
              <option value="gwh">GWH</option>
            </select>
            <div class="pes-actions">
              <button id="go-run" class="primary-btn">Optimize</button>
              <button id="go-cancel" class="secondary-btn hidden">Cancel</button>
            </div>
            <div class="pes-progress" id="go-progress"></div>
          </div>
        </div>
      </div>

      <div class="geomopt-right">
        <div class="panel">
          <div id="go-mol-preview" style="width:100%;max-width:360px;aspect-ratio:1;margin:0 auto;"></div>
          <div class="pes-slider-row" id="go-slider-row" style="display:none">
            <input type="range" id="go-slider" min="0" max="0" value="0" />
            <span id="go-slider-label" class="pes-slider-label"></span>
          </div>
        </div>
        <div class="panel geomopt-plots">
          <div id="go-energy-plot"></div>
          <div id="go-gradient-plot"></div>
        </div>
        <div class="panel" id="go-result-panel" style="display:none">
          <h2>Result</h2>
          <pre id="go-result-text" style="font-size:0.75rem;overflow-x:auto;max-height:200px;background:var(--color-surface-alt);padding:8px;border-radius:6px"></pre>
        </div>
        <details class="panel" id="go-log-details">
          <summary>Raw Output</summary>
          <pre id="go-log" style="font-size:0.7rem;overflow-x:auto;max-height:300px;background:var(--color-surface-alt);padding:8px;border-radius:6px;white-space:pre-wrap"></pre>
        </details>
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

  // Elements
  const sampleSelect = root.querySelector<HTMLSelectElement>('#go-sample')!;
  const xyzArea = root.querySelector<HTMLTextAreaElement>('#go-xyz')!;
  const distortBtn = root.querySelector<HTMLButtonElement>('#go-distort')!;
  const distortMag = root.querySelector<HTMLInputElement>('#go-distort-mag')!;
  const distortLabel = root.querySelector<HTMLElement>('#go-distort-label')!;
  const methodSelect = root.querySelector<HTMLSelectElement>('#go-method')!;
  const basisSelect = root.querySelector<HTMLSelectElement>('#go-basis')!;
  const optimizerSelect = root.querySelector<HTMLSelectElement>('#go-optimizer')!;
  const guessSelect = root.querySelector<HTMLSelectElement>('#go-guess')!;
  const runBtn = root.querySelector<HTMLButtonElement>('#go-run')!;
  const cancelBtn = root.querySelector<HTMLButtonElement>('#go-cancel')!;
  const progressEl = root.querySelector<HTMLElement>('#go-progress')!;
  const molPreview = root.querySelector<HTMLElement>('#go-mol-preview')!;
  const sliderRow = root.querySelector<HTMLElement>('#go-slider-row')!;
  const slider = root.querySelector<HTMLInputElement>('#go-slider')!;
  const sliderLabel = root.querySelector<HTMLElement>('#go-slider-label')!;
  const energyPlot = root.querySelector<HTMLElement>('#go-energy-plot')!;
  const gradientPlot = root.querySelector<HTMLElement>('#go-gradient-plot')!;
  const resultPanel = root.querySelector<HTMLElement>('#go-result-panel')!;
  const resultText = root.querySelector<HTMLElement>('#go-result-text')!;
  const logEl = root.querySelector<HTMLElement>('#go-log')!;

  // Distort magnitude label
  distortMag.addEventListener('input', () => {
    distortLabel.innerHTML = `${(parseInt(distortMag.value) / 100).toFixed(2)} &#197;`;
  });

  // Distort button: randomly displace atoms
  let originalXYZ = '';  // the base (undistorted) XYZ
  distortBtn.addEventListener('click', () => {
    const base = originalXYZ || xyzArea.value.trim();
    if (!originalXYZ) originalXYZ = base;
    const mag = parseInt(distortMag.value) / 100;
    const distorted = distortXYZ(base, mag);
    xyzArea.value = distorted;
    renderMoleculePreview(molPreview, distorted, []);
  });

  // Load samples
  const samples = await fetchSamples('.');
  for (const s of samples) {
    const opt = document.createElement('option');
    opt.value = s.filename;
    opt.textContent = s.name;
    sampleSelect.appendChild(opt);
  }

  sampleSelect.addEventListener('change', async () => {
    if (!sampleSelect.value) return;
    const content = await fetchSampleContent(sampleSelect.value);
    xyzArea.value = content;
    originalXYZ = content;  // save as base for distortion
    renderMoleculePreview(molPreview, content, []);
  });

  xyzArea.addEventListener('input', () => {
    if (xyzArea.value.trim()) renderMoleculePreview(molPreview, xyzArea.value, []);
  });

  // Load basis sets
  const basisSets = await fetchBasisSets();
  for (const b of basisSets) {
    const opt = document.createElement('option');
    opt.value = b;
    opt.textContent = b;
    if (b === 'sto-3g') opt.selected = true;
    basisSelect.appendChild(opt);
  }

  // Default molecule
  const defaultXYZ = `3\nH2O\nO  0.000000  0.000000  0.117300\nH  0.000000  0.756950 -0.469200\nH  0.000000 -0.756950 -0.469200`;
  xyzArea.value = defaultXYZ;
  originalXYZ = defaultXYZ;
  renderMoleculePreview(molPreview, defaultXYZ, []);

  // ── Scan state (for slider interaction after completion) ──
  let allSteps: OptStep[] = [];
  let allGeoms: Map<number, StepGeom> = new Map();
  let scanDone = false;

  function showStepInViewer(idx: number) {
    if (idx < 0 || idx >= allSteps.length) return;
    const stepNum = allSteps[idx].step;
    const geom = allGeoms.get(stepNum);
    if (geom) {
      const xyz = atomsToXYZ(geom.atoms);
      const ann: MolAnnotation[] = [{ type: 'forces', forces: geom.forces }];
      renderMoleculePreview(molPreview, xyz, ann);
    }
    sliderLabel.textContent = `Step ${stepNum}: E = ${allSteps[idx].energy.toFixed(8)} Ha`;

    // Update plots with highlight
    const stepNums = allSteps.map(s => s.step);
    const energies = allSteps.map(s => s.energy);
    const maxGrads = allSteps.map(s => s.maxGrad);
    renderLinePlot(energyPlot, stepNums, energies, 'Step', 'Energy (Hartree)', '#3b82f6', false, idx);
    renderLinePlot(gradientPlot, stepNums, maxGrads, 'Step', 'Max Gradient', '#ef4444', true, idx);
  }

  slider.addEventListener('input', () => {
    if (!scanDone) return;
    const idx = parseInt(slider.value);
    showStepInViewer(idx);
  });

  // Run optimization
  let abortController: AbortController | null = null;

  runBtn.addEventListener('click', async () => {
    const xyzText = xyzArea.value.trim();
    if (!xyzText) { progressEl.textContent = 'No molecule specified'; return; }

    runBtn.disabled = true;
    cancelBtn.classList.remove('hidden');
    progressEl.textContent = 'Starting optimization...';
    energyPlot.innerHTML = '';
    gradientPlot.innerHTML = '';
    resultPanel.style.display = 'none';
    logEl.textContent = '';
    sliderRow.style.display = 'none';
    allSteps = [];
    allGeoms = new Map();
    scanDone = false;

    // Expected atom count from input XYZ (first line)
    const numAtoms = parseInt(xyzText.split(/\r?\n/)[0].trim(), 10) || 0;

    let fullOutput = '';
    abortController = new AbortController();

    try {
      const res = await fetch('/api/run/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          xyz_text: xyzText,
          basis: basisSelect.value,
          method: methodSelect.value,
          run_type: 'optimize',
          optimizer: optimizerSelect.value,
          initial_guess: guessSelect.value,
          timeout: 1200,
        }),
        signal: abortController.signal,
      });

      const reader = res.body?.getReader();
      if (!reader) return;
      const decoder = new TextDecoder();
      let buffer = '';

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
            if (event.type === 'line') {
              const cleanLine = stripAnsi(event.text);
              fullOutput += cleanLine + '\n';
              logEl.textContent = fullOutput;
              logEl.scrollTop = logEl.scrollHeight;

              // Update steps and plots
              const newSteps = parseOptSteps(fullOutput);
              if (newSteps.length > allSteps.length) {
                allSteps = newSteps;
                const last = allSteps[allSteps.length - 1];
                progressEl.textContent = `Step ${last.step}: E = ${last.energy.toFixed(8)} Ha, max grad = ${last.maxGrad.toExponential(2)}`;

                const stepNums = allSteps.map(s => s.step);
                const energies = allSteps.map(s => s.energy);
                const maxGrads = allSteps.map(s => s.maxGrad);
                renderLinePlot(energyPlot, stepNums, energies, 'Step', 'Energy (Hartree)', '#3b82f6', false, allSteps.length - 1);
                renderLinePlot(gradientPlot, stepNums, maxGrads, 'Step', 'Max Gradient', '#ef4444', true, allSteps.length - 1);
              }

              // Update 3D viewer whenever new geometry data arrives
              const newGeoms = parseStepGeometries(fullOutput, numAtoms);
              if (newGeoms.size > allGeoms.size) {
                allGeoms = newGeoms;
                // Show the latest available geometry with force vectors
                const latestStep = Math.max(...allGeoms.keys());
                const geom = allGeoms.get(latestStep);
                if (geom) {
                  const xyz = atomsToXYZ(geom.atoms);
                  const ann: MolAnnotation[] = [{ type: 'forces', forces: geom.forces }];
                  renderMoleculePreview(molPreview, xyz, ann);
                }
              }

              if (cleanLine.includes('Geometry Optimization Converged')) {
                progressEl.textContent = 'Converged!';
              }
            } else if (event.type === 'error') {
              progressEl.textContent = `Error: ${event.error}`;
              fullOutput += '\n--- ERROR ---\n' + (event.raw_output || event.error);
              logEl.textContent = fullOutput;
            } else if (event.type === 'result') {
              // Re-parse everything from full output to ensure final step is captured
              allSteps = parseOptSteps(fullOutput);
              allGeoms = parseStepGeometries(fullOutput, numAtoms);

              const optAtoms = parseOptimizedGeometry(fullOutput);
              if (optAtoms.length > 0) {
                const xyz = atomsToXYZ(optAtoms);
                // Show final geometry with last step's force vectors
                const lastGeom = allSteps.length > 0 ? allGeoms.get(allSteps[allSteps.length - 1].step) : undefined;
                const finalAnn: MolAnnotation[] = lastGeom ? [{ type: 'forces', forces: lastGeom.forces }] : [];
                renderMoleculePreview(molPreview, xyz, finalAnn);
                resultPanel.style.display = '';
                let summary = `Optimized Geometry (Angstrom):\n`;
                for (const a of optAtoms) {
                  summary += `  ${a.element.padEnd(4)} ${a.x.toFixed(8)}  ${a.y.toFixed(8)}  ${a.z.toFixed(8)}\n`;
                }
                if (allSteps.length > 0) {
                  summary += `\nFinal energy: ${allSteps[allSteps.length - 1].energy.toFixed(12)} Hartree\n`;
                  summary += `Steps: ${allSteps[allSteps.length - 1].step}\n`;
                }
                resultText.textContent = summary;
              }
            }
          } catch { /* skip */ }
        }
      }
    } catch (e: unknown) {
      if (e instanceof Error && e.name === 'AbortError') {
        progressEl.textContent = 'Cancelled';
      } else {
        progressEl.textContent = `Error: ${e}`;
      }
    }

    // Scan done — enable slider
    scanDone = true;
    runBtn.disabled = false;
    cancelBtn.classList.add('hidden');
    abortController = null;

    if (allSteps.length > 1) {
      sliderRow.style.display = '';
      slider.min = '0';
      slider.max = String(allSteps.length - 1);
      slider.value = String(allSteps.length - 1);
      sliderLabel.textContent = `Step ${allSteps[allSteps.length - 1].step}`;
    }
  });

  cancelBtn.addEventListener('click', () => {
    abortController?.abort();
  });
}

initGeomOpt();
