/** SVG orbital energy diagram — zoomable, with label de-overlap and degeneracy grouping.
 *  Adapted from GANSU-web for GANSU-UI's OrbitalEnergy[] format. */

import type { OrbitalEnergy } from '../types';
import { getThemeColors } from '../ui/theme';

const MIN_LABEL_GAP = 12;
const DEGEN_THRESHOLD = 1e-4; // Hartree
const ORB_GAP = 6;

interface OrbGroup {
  indices: number[];
  energy: number;
  occupations: string[];
}

function groupByDegeneracy(orbitals: OrbitalEnergy[]): OrbGroup[] {
  const groups: OrbGroup[] = [];
  let i = 0;
  while (i < orbitals.length) {
    const indices = [i];
    const occs = [orbitals[i].occupation];
    const e0 = orbitals[i].energy;
    while (i + 1 < orbitals.length && Math.abs(orbitals[i + 1].energy - e0) < DEGEN_THRESHOLD) {
      i++;
      indices.push(i);
      occs.push(orbitals[i].occupation);
    }
    const avg = indices.reduce((s, idx) => s + orbitals[idx].energy, 0) / indices.length;
    groups.push({ indices, energy: avg, occupations: occs });
    i++;
  }
  return groups;
}

function groupRightEdge(xBase: number, k: number, lineLen: number): number {
  return xBase + k * lineLen + (k - 1) * ORB_GAP;
}

function deOverlap(positions: number[], minGap: number): number[] {
  const out = positions.slice();
  const idx = out.map((_, i) => i);
  idx.sort((a, b) => out[a] - out[b]);
  for (let k = 1; k < idx.length; k++) {
    const prev = idx[k - 1];
    const curr = idx[k];
    if (out[curr] - out[prev] < minGap) {
      out[curr] = out[prev] + minGap;
    }
  }
  return out;
}

function makeZoomableContainer(
  container: HTMLElement,
  buildSvg: (zoom: number) => string,
  baseHeight: number,
): void {
  let zoom = 2;
  const maxZoom = 12;
  const minZoom = 0.5;
  const scrollH = Math.min(baseHeight, 520);

  const outer = document.createElement('div');
  outer.style.position = 'relative';

  const toolbar = document.createElement('div');
  toolbar.style.cssText = 'display:flex;align-items:center;justify-content:center;gap:6px;padding:2px 0 4px;font-size:11px;color:var(--color-text-dim,#888)';
  const btnCss = 'width:22px;height:22px;border:1px solid var(--color-border,#ccc);border-radius:4px;background:var(--color-surface,#fff);color:var(--color-text,#333);cursor:pointer;font-size:14px;line-height:1;display:flex;align-items:center;justify-content:center';
  const zoomOutBtn = document.createElement('button');
  zoomOutBtn.style.cssText = btnCss;
  zoomOutBtn.textContent = '\u2212';
  zoomOutBtn.title = 'Zoom out';
  const zoomInBtn = document.createElement('button');
  zoomInBtn.style.cssText = btnCss;
  zoomInBtn.textContent = '+';
  zoomInBtn.title = 'Zoom in';
  const zoomLabel = document.createElement('span');
  zoomLabel.style.cssText = 'min-width:40px;text-align:center;font-size:10px';

  toolbar.appendChild(zoomOutBtn);
  toolbar.appendChild(zoomLabel);
  toolbar.appendChild(zoomInBtn);
  outer.appendChild(toolbar);

  const wrapper = document.createElement('div');
  wrapper.style.overflowY = 'auto';
  wrapper.style.maxHeight = `${scrollH}px`;
  wrapper.style.position = 'relative';
  outer.appendChild(wrapper);

  function applyZoom(newZoom: number) {
    newZoom = Math.max(minZoom, Math.min(maxZoom, newZoom));
    if (newZoom === zoom) return;
    const scrollRatio = wrapper.scrollHeight > scrollH
      ? wrapper.scrollTop / (wrapper.scrollHeight - scrollH)
      : 0;
    zoom = newZoom;
    render();
    const newScrollMax = wrapper.scrollHeight - scrollH;
    if (newScrollMax > 0) {
      wrapper.scrollTop = scrollRatio * newScrollMax;
    }
  }

  function render() {
    wrapper.innerHTML = buildSvg(zoom);
    zoomLabel.textContent = `${Math.round(zoom * 100)}%`;
    zoomOutBtn.disabled = zoom <= minZoom;
    zoomInBtn.disabled = zoom >= maxZoom;
  }

  wrapper.addEventListener('wheel', (e: WheelEvent) => {
    if (!e.ctrlKey && !e.metaKey) return;
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.2 : 1 / 1.2;
    applyZoom(zoom * factor);
  }, { passive: false });

  zoomInBtn.addEventListener('click', () => applyZoom(zoom * 1.3));
  zoomOutBtn.addEventListener('click', () => applyZoom(zoom / 1.3));

  render();
  container.appendChild(outer);
}

function drawDoubleArrows(cx: number, y: number, color: string): string {
  const hw = 3;
  const off = hw + 0.5;
  let svg = `<polygon points="${cx - off - hw},${y - 2} ${cx - off},${y - 10} ${cx - off + hw},${y - 2}" fill="${color}"/>`;
  svg += `<polygon points="${cx + off - hw},${y - 10} ${cx + off},${y - 2} ${cx + off + hw},${y - 10}" fill="${color}"/>`;
  return svg;
}

function drawUpArrow(cx: number, y: number, color: string): string {
  return `<polygon points="${cx - 3},${y - 2} ${cx},${y - 10} ${cx + 3},${y - 2}" fill="${color}"/>`;
}

function drawDownArrow(cx: number, y: number, color: string): string {
  return `<polygon points="${cx - 3},${y - 10} ${cx},${y - 2} ${cx + 3},${y - 10}" fill="${color}"/>`;
}

function drawGapAnnotation(
  homoE: number, lumoE: number, toY: (e: number) => number,
  gapX: number, tc: ReturnType<typeof getThemeColors>,
): string {
  const gapEv = (lumoE - homoE) * 27.2114;
  if (Math.abs(gapEv) < 0.01) return '';
  const yHOMO = toY(homoE);
  const yLUMO = toY(lumoE);
  let svg = `<line x1="${gapX}" y1="${yHOMO}" x2="${gapX}" y2="${yLUMO}" stroke="${tc.gap}" stroke-width="1.5" stroke-dasharray="3,2"/>`;
  svg += `<polygon points="${gapX - 3},${yHOMO - 4} ${gapX},${yHOMO} ${gapX + 3},${yHOMO - 4}" fill="${tc.gap}"/>`;
  svg += `<polygon points="${gapX - 3},${yLUMO + 4} ${gapX},${yLUMO} ${gapX + 3},${yLUMO + 4}" fill="${tc.gap}"/>`;
  const gapMidY = (yHOMO + yLUMO) / 2;
  svg += `<text x="${gapX + 6}" y="${gapMidY + 4}" font-size="10" fill="${tc.gap}">${gapEv.toFixed(2)} eV</text>`;
  return svg;
}

/** Determine if orbital is occupied (occ, closed, open) */
function isOccupied(occ: string): boolean {
  return occ === 'occ' || occ === 'closed' || occ === 'open';
}

/** Find HOMO index (last occupied) in flat orbital array */
function findHOMO(orbitals: OrbitalEnergy[]): number {
  for (let i = orbitals.length - 1; i >= 0; i--) {
    if (isOccupied(orbitals[i].occupation)) return i;
  }
  return -1;
}

/** RHF orbital diagram — occ = doubly occupied (up+down arrows), vir = virtual */
export function renderOrbitalDiagram(
  container: HTMLElement,
  orbitals: OrbitalEnergy[],
): void {
  const n = orbitals.length;
  if (n === 0) return;

  const groups = groupByDegeneracy(orbitals);
  const homoIdx = findHOMO(orbitals);
  const lumoIdx = homoIdx + 1 < n ? homoIdx + 1 : -1;

  const marginTop = 28;
  const marginBottom = 24;
  const marginLeft = 70;
  const lineLen = 40;
  const maxK = Math.max(...groups.map(g => g.indices.length));
  const maxRight = groupRightEdge(marginLeft, maxK, lineLen);
  const width = Math.max(300, maxRight + 100);

  const eMin = orbitals[0].energy;
  const eMax = orbitals[n - 1].energy;
  const range = eMax - eMin || 1;
  const padded = range * 0.1;
  const yMin = eMin - padded;
  const yMax = eMax + padded;
  const basePlotH = Math.max(180, Math.min(480, groups.length * 40));

  function buildSvg(zoom: number): string {
    const tc = getThemeColors();
    const plotH = basePlotH * zoom;
    const height = plotH + marginTop + marginBottom;
    const toY = (e: number) => marginTop + plotH - ((e - yMin) / (yMax - yMin)) * plotH;

    const rawGroupY = groups.map(g => toY(g.energy));
    const groupLabelY = deOverlap(rawGroupY, MIN_LABEL_GAP);

    let svg = `<svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg" style="display:block;margin:0 auto;">`;

    svg += `<line x1="${marginLeft - 8}" y1="${marginTop}" x2="${marginLeft - 8}" y2="${marginTop + plotH}" stroke="${tc.axis}" stroke-width="1"/>`;

    for (let gi = 0; gi < groups.length; gi++) {
      const group = groups[gi];
      const y = toY(group.energy);
      const ly = groupLabelY[gi];
      const k = group.indices.length;
      const gRight = groupRightEdge(marginLeft, k, lineLen);

      const groupHasHOMO = homoIdx >= 0 && group.indices.includes(homoIdx);
      const groupHasLUMO = lumoIdx >= 0 && group.indices.includes(lumoIdx);

      for (let g = 0; g < k; g++) {
        const idx = group.indices[g];
        const occ = orbitals[idx].occupation;
        const isOcc = isOccupied(occ);
        const isSpecial = idx === homoIdx || idx === lumoIdx;
        const color = isOcc ? tc.occupied : tc.virtual;
        const strokeW = isSpecial ? 2.5 : 1.5;
        const x0 = marginLeft + g * (lineLen + ORB_GAP);

        svg += `<line x1="${x0}" y1="${y}" x2="${x0 + lineLen}" y2="${y}" stroke="${color}" stroke-width="${strokeW}"/>`;

        if (occ === 'occ' || occ === 'closed') {
          svg += drawDoubleArrows(x0 + lineLen / 2, y, color);
        } else if (occ === 'open') {
          svg += drawUpArrow(x0 + lineLen / 2, y, color);
        }
      }

      if (Math.abs(ly - y) > 3) {
        svg += `<line x1="${marginLeft - 10}" y1="${y}" x2="${marginLeft - 14}" y2="${ly}" stroke="${tc.leader}" stroke-width="0.5"/>`;
      }

      const degLabel = k > 1 ? ` (\u00d7${k})` : '';
      svg += `<text x="${marginLeft - 16}" y="${ly + 4}" text-anchor="end" font-size="10" fill="${tc.label}">${group.energy.toFixed(3)}${degLabel}</text>`;

      // SOMO label for open-shell orbitals
      const hasOpen = group.occupations.some(o => o === 'open');
      if (groupHasHOMO) {
        const lbl = hasOpen ? 'SOMO' : 'HOMO';
        svg += `<text x="${gRight + 6}" y="${y + 4}" font-size="11" fill="${tc.occupied}" font-weight="bold">${lbl}</text>`;
      } else if (hasOpen) {
        svg += `<text x="${gRight + 6}" y="${y + 4}" font-size="10" fill="${tc.alpha}" font-weight="bold">SOMO</text>`;
      }
      if (groupHasLUMO && !groupHasHOMO) {
        svg += `<text x="${gRight + 6}" y="${y + 4}" font-size="11" fill="${tc.dim}" font-weight="bold">LUMO</text>`;
      }
    }

    // HOMO-LUMO gap
    if (homoIdx >= 0 && lumoIdx >= 0) {
      svg += drawGapAnnotation(orbitals[homoIdx].energy, orbitals[lumoIdx].energy, toY, maxRight + 50, tc);
    }

    svg += `<text x="${width / 2}" y="14" text-anchor="middle" font-size="10" fill="${tc.hint}">Ctrl+Scroll or +/\u2212 to zoom</text>`;
    svg += '</svg>';
    return svg;
  }

  container.innerHTML = '';
  const baseH = basePlotH + marginTop + marginBottom;
  makeZoomableContainer(container, buildSvg, baseH);
}

/** UHF orbital diagram — alpha (left) and beta (right) side by side */
export function renderUHFOrbitalDiagram(
  container: HTMLElement,
  alphaOrbitals: OrbitalEnergy[],
  betaOrbitals: OrbitalEnergy[],
): void {
  const nA = alphaOrbitals.length;
  const nB = betaOrbitals.length;
  if (nA === 0 && nB === 0) return;

  const groupsA = groupByDegeneracy(alphaOrbitals);
  const groupsB = groupByDegeneracy(betaOrbitals);
  const homoA = findHOMO(alphaOrbitals);
  const lumoA = homoA + 1 < nA ? homoA + 1 : -1;
  const homoB = findHOMO(betaOrbitals);
  const lumoB = homoB + 1 < nB ? homoB + 1 : -1;

  const marginTop = 28;
  const marginBottom = 24;
  const marginLeft = 60;
  const lineLen = 36;
  const colSep = 50;

  const maxKa = Math.max(1, ...groupsA.map(g => g.indices.length));
  const maxKb = Math.max(1, ...groupsB.map(g => g.indices.length));
  const alphaMaxRight = groupRightEdge(marginLeft, maxKa, lineLen);
  const betaX = alphaMaxRight + colSep;
  const betaMaxRight = groupRightEdge(betaX, maxKb, lineLen);
  const width = Math.max(420, betaMaxRight + 80);

  // Unified energy scale
  const allE: number[] = [];
  for (const o of alphaOrbitals) allE.push(o.energy);
  for (const o of betaOrbitals) allE.push(o.energy);
  allE.sort((a, b) => a - b);
  const eMin = allE[0];
  const eMax = allE[allE.length - 1];
  const range = eMax - eMin || 1;
  const padded = range * 0.1;
  const yMin = eMin - padded;
  const yMax = eMax + padded;
  const nMaxGroups = Math.max(groupsA.length, groupsB.length);
  const basePlotH = Math.max(180, Math.min(480, nMaxGroups * 40));

  function buildSvg(zoom: number): string {
    const tc = getThemeColors();
    const plotH = basePlotH * zoom;
    const height = plotH + marginTop + marginBottom;
    const toY = (e: number) => marginTop + plotH - ((e - yMin) / (yMax - yMin)) * plotH;

    let svg = `<svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg" style="display:block;margin:0 auto;">`;
    svg += `<text x="${width / 2}" y="14" text-anchor="middle" font-size="10" fill="${tc.hint}">Ctrl+Scroll or +/\u2212 to zoom</text>`;

    function drawColumn(
      groups: OrbGroup[], orbitals: OrbitalEnergy[], homo: number, lumo: number,
      xBase: number, maxGroupRight: number,
      color: string, arrowUp: boolean, label: string,
    ) {
      const headerX = (xBase + maxGroupRight) / 2;
      svg += `<text x="${headerX}" y="${marginTop - 4}" text-anchor="middle" font-size="10" fill="${color}" font-weight="bold">${label}</text>`;
      svg += `<line x1="${xBase - 8}" y1="${marginTop}" x2="${xBase - 8}" y2="${marginTop + plotH}" stroke="${tc.axis}" stroke-width="1"/>`;

      const rawGroupY = groups.map(g => toY(g.energy));
      const groupLabelY = deOverlap(rawGroupY, MIN_LABEL_GAP);

      for (let gi = 0; gi < groups.length; gi++) {
        const group = groups[gi];
        const y = toY(group.energy);
        const ly = groupLabelY[gi];
        const k = group.indices.length;
        const gRight = groupRightEdge(xBase, k, lineLen);

        const groupHasHOMO = homo >= 0 && group.indices.includes(homo);
        const groupHasLUMO = lumo >= 0 && group.indices.includes(lumo);

        for (let g = 0; g < k; g++) {
          const idx = group.indices[g];
          const isOcc = isOccupied(orbitals[idx].occupation);
          const isSpecial = idx === homo || idx === lumo;
          const lineColor = isOcc ? color : tc.virtual;
          const strokeW = isSpecial ? 2.5 : 1.5;
          const x0 = xBase + g * (lineLen + ORB_GAP);

          svg += `<line x1="${x0}" y1="${y}" x2="${x0 + lineLen}" y2="${y}" stroke="${lineColor}" stroke-width="${strokeW}"/>`;

          if (isOcc) {
            const cx = x0 + lineLen / 2;
            svg += arrowUp ? drawUpArrow(cx, y, lineColor) : drawDownArrow(cx, y, lineColor);
          }
        }

        if (arrowUp) {
          if (Math.abs(ly - y) > 3) {
            svg += `<line x1="${xBase - 10}" y1="${y}" x2="${xBase - 14}" y2="${ly}" stroke="${tc.leader}" stroke-width="0.5"/>`;
          }
          const degLabel = k > 1 ? ` (\u00d7${k})` : '';
          svg += `<text x="${xBase - 16}" y="${ly + 4}" text-anchor="end" font-size="9" fill="${tc.label}">${group.energy.toFixed(3)}${degLabel}</text>`;
        } else {
          if (Math.abs(ly - y) > 3) {
            svg += `<line x1="${gRight + 2}" y1="${y}" x2="${gRight + 6}" y2="${ly}" stroke="${tc.leader}" stroke-width="0.5"/>`;
          }
          const degLabel = k > 1 ? ` (\u00d7${k})` : '';
          svg += `<text x="${gRight + 8}" y="${ly + 4}" text-anchor="start" font-size="9" fill="${tc.label}">${group.energy.toFixed(3)}${degLabel}</text>`;
        }

        if (groupHasHOMO) {
          const lx = arrowUp ? gRight + 4 : xBase - 12;
          const anchor = arrowUp ? 'start' : 'end';
          svg += `<text x="${lx}" y="${y + 4}" text-anchor="${anchor}" font-size="9" fill="${color}" font-weight="bold">HOMO</text>`;
        }
        if (groupHasLUMO && !groupHasHOMO) {
          const lx = arrowUp ? gRight + 4 : xBase - 12;
          const anchor = arrowUp ? 'start' : 'end';
          svg += `<text x="${lx}" y="${y + 4}" text-anchor="${anchor}" font-size="9" fill="${tc.dim}" font-weight="bold">LUMO</text>`;
        }
      }
    }

    drawColumn(groupsA, alphaOrbitals, homoA, lumoA, marginLeft, alphaMaxRight, tc.alpha, true, '\u03B1');
    drawColumn(groupsB, betaOrbitals, homoB, lumoB, betaX, betaMaxRight, tc.beta, false, '\u03B2');

    svg += '</svg>';
    return svg;
  }

  container.innerHTML = '';
  const baseH = basePlotH + marginTop + marginBottom;
  makeZoomableContainer(container, buildSvg, baseH);
}
