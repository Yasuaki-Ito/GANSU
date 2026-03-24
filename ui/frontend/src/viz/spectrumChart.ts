/** SVG UV-Vis absorption spectrum: Gaussian-broadened oscillator strengths vs energy */

import { getThemeColors } from '../ui/theme';
import type { ExcitedState } from '../types';

/** Render UV-Vis absorption spectrum as SVG with excited states table. */
export function renderSpectrumChart(
  container: HTMLElement,
  states: ExcitedState[],
  method: string,
  spin: string,
): void {
  if (!states || states.length === 0) {
    container.innerHTML = '<p style="color:var(--color-text-dim);font-size:0.85rem;">No excited state data</p>';
    return;
  }

  const tc = getThemeColors();
  const isTriplet = spin === 'triplet';

  const width = 700;
  const height = 280;
  const ml = 48, mr = 16, mt = 28, mb = 38;
  const pw = width - ml - mr;
  const ph = height - mt - mb;

  // Energy range (eV)
  const energies = states.map(s => s.energy_ev);
  const eMin = Math.max(0, Math.min(...energies) - 2);
  const eMax = Math.max(...energies) + 2;

  // Gaussian broadening
  const sigma = 0.4;
  const nPts = 200;
  const de = (eMax - eMin) / nPts;

  const spectrum = new Float64Array(nPts);
  for (let k = 0; k < nPts; k++) {
    const E = eMin + k * de;
    let val = 0.0;
    for (const s of states) {
      const f = s.osc_strength;
      if (f > 0) {
        const diff = E - s.energy_ev;
        val += f * Math.exp(-diff * diff / (2 * sigma * sigma));
      }
    }
    spectrum[k] = val;
  }

  const yMax = Math.max(...spectrum) * 1.15 || 1.0;

  const toX = (e: number) => ml + ((e - eMin) / (eMax - eMin || 1)) * pw;
  const toY = (y: number) => mt + ph - (y / yMax) * ph;

  let svg = `<svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg" style="display:block;width:100%;height:auto;" viewBox="0 0 ${width} ${height}">`;

  // Grid lines
  const eStep = Math.ceil((eMax - eMin) / 6);
  for (let ge = Math.ceil(eMin); ge <= eMax; ge += Math.max(1, eStep)) {
    const xx = toX(ge);
    if (xx < ml || xx > ml + pw) continue;
    svg += `<line x1="${xx}" y1="${mt}" x2="${xx}" y2="${mt + ph}" stroke="${tc.grid}" stroke-width="0.5"/>`;
    svg += `<text x="${xx}" y="${mt + ph + 14}" text-anchor="middle" font-size="9" fill="${tc.dim}">${ge}</text>`;
  }

  // Axes
  svg += `<line x1="${ml}" y1="${mt}" x2="${ml}" y2="${mt + ph}" stroke="${tc.axis}" stroke-width="1"/>`;
  svg += `<line x1="${ml}" y1="${mt + ph}" x2="${ml + pw}" y2="${mt + ph}" stroke="${tc.axis}" stroke-width="1"/>`;

  // Spectrum curve (filled area) — only for singlet (has f > 0)
  if (!isTriplet) {
    let pathArea = `M${ml},${mt + ph}`;
    for (let k = 0; k < nPts; k++) {
      pathArea += ` L${toX(eMin + k * de).toFixed(1)},${toY(spectrum[k]).toFixed(1)}`;
    }
    pathArea += ` L${toX(eMax).toFixed(1)},${mt + ph} Z`;
    svg += `<path d="${pathArea}" fill="${tc.accent}" fill-opacity="0.15" stroke="none"/>`;

    let pathLine = '';
    for (let k = 0; k < nPts; k++) {
      const px = toX(eMin + k * de);
      const py = toY(spectrum[k]);
      pathLine += k === 0 ? `M${px.toFixed(1)},${py.toFixed(1)}` : ` L${px.toFixed(1)},${py.toFixed(1)}`;
    }
    svg += `<path d="${pathLine}" fill="none" stroke="${tc.accent}" stroke-width="1.5"/>`;
  }

  // Stick spectrum
  for (const s of states) {
    const xx = toX(s.energy_ev);
    if (xx < ml || xx > ml + pw) continue;
    if (!isTriplet && s.osc_strength <= 0) continue;  // skip dark states for singlet
    const stickHeight = isTriplet ? ph * 0.5 : (s.osc_strength / yMax) * ph;
    const yy = mt + ph - stickHeight;
    const color = isTriplet ? tc.error : tc.accent;
    svg += `<line x1="${xx.toFixed(1)}" y1="${(mt + ph).toFixed(1)}" x2="${xx.toFixed(1)}" y2="${yy.toFixed(1)}" stroke="${color}" stroke-width="1.5" stroke-opacity="0.7"/>`;
    svg += `<circle cx="${xx.toFixed(1)}" cy="${yy.toFixed(1)}" r="2" fill="${color}"/>`;
  }

  // Title
  const spinLabel = isTriplet ? 'Triplet' : 'Singlet';
  const title = `${method} ${spinLabel} Absorption Spectrum`;
  svg += `<text x="${width / 2}" y="16" text-anchor="middle" font-size="11" fill="${tc.titleSvg}">${title}</text>`;

  // Axis labels
  svg += `<text x="${ml + pw / 2}" y="${height - 4}" text-anchor="middle" font-size="9" fill="${tc.dim}">Energy (eV)</text>`;
  if (!isTriplet) {
    svg += `<text x="12" y="${mt + ph / 2}" text-anchor="middle" font-size="9" fill="${tc.dim}" transform="rotate(-90,12,${mt + ph / 2})">Osc. Strength (arb.)</text>`;
  }

  svg += '</svg>';

  // Excited states table
  let table = `<table class="result-table"><tr><th>State</th><th>Energy (Ha)</th><th>Energy (eV)</th><th>f</th><th>Transitions</th></tr>`;
  for (const s of states) {
    table += `<tr>
      <td>${s.state}</td>
      <td>${s.energy_ha.toFixed(6)}</td>
      <td>${s.energy_ev.toFixed(3)}</td>
      <td>${s.osc_strength.toFixed(4)}</td>
      <td style="font-size:0.8rem">${escapeHtml(s.transitions)}</td>
    </tr>`;
  }
  table += '</table>';

  container.innerHTML = `<h3>Excited States (${method} ${spinLabel})</h3>${svg}${table}`;
}

function escapeHtml(text: string): string {
  return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
