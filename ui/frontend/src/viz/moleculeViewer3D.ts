/** Interactive 3D molecule viewer — Three.js WebGL */

import * as THREE from 'three';
import { ELEMENT_NAME_TO_ATOMIC_NUMBER } from './constants';
import { isDark } from '../ui/theme';

// CPK element colors
const CPK_COLORS: Record<number, number> = {
  1: 0xFFFFFF, 2: 0xD9FFFF, 3: 0xCC80FF, 4: 0xC2FF00, 5: 0xFFB5B5,
  6: 0x909090, 7: 0x3050F8, 8: 0xFF0D0D, 9: 0x90E050, 10: 0xB3E3F5,
  11: 0xAB5CF2, 12: 0x8AFF00, 13: 0xBFA6A6, 14: 0xF0C8A0, 15: 0xFF8000,
  16: 0xFFFF30, 17: 0x1FF01F, 18: 0x80D1E3, 19: 0x8F40D4, 20: 0x3DFF00,
  26: 0xE06633, 29: 0xC88033, 30: 0x7D80B0, 35: 0xA62929, 53: 0x940094,
};

const COVALENT_RADII: Record<number, number> = {
  1: 0.31, 2: 0.28, 3: 1.28, 4: 0.96, 5: 0.84, 6: 0.76, 7: 0.71, 8: 0.66,
  9: 0.57, 10: 0.58, 11: 1.66, 12: 1.41, 13: 1.21, 14: 1.11, 15: 1.07, 16: 1.05,
  17: 1.02, 18: 1.06, 19: 2.03, 20: 1.76, 26: 1.32, 29: 1.32, 30: 1.22,
  35: 1.20, 53: 1.39,
};

const ATOM_RADII: Record<number, number> = {
  1: 0.25, 6: 0.40, 7: 0.38, 8: 0.36, 9: 0.33, 15: 0.45, 16: 0.45, 17: 0.43,
};

interface Vec3 { x: number; y: number; z: number; }
interface Bond { i: number; j: number; }

function getColor(z: number): number { return CPK_COLORS[z] ?? 0xFF69B4; }
function getCovalentRadius(z: number): number { return COVALENT_RADII[z] ?? 1.5; }
function getAtomRadius(z: number): number { return ATOM_RADII[z] ?? 0.40; }

function detectBonds(positions: Vec3[], atomicNumbers: number[]): Bond[] {
  const bonds: Bond[] = [];
  const tolerance = 0.4;
  for (let i = 0; i < positions.length; i++) {
    for (let j = i + 1; j < positions.length; j++) {
      const dx = positions[i].x - positions[j].x;
      const dy = positions[i].y - positions[j].y;
      const dz = positions[i].z - positions[j].z;
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      const maxDist = getCovalentRadius(atomicNumbers[i]) + getCovalentRadius(atomicNumbers[j]) + tolerance;
      if (dist < maxDist && dist > 0.1) bonds.push({ i, j });
    }
  }
  return bonds;
}

const sphereGeo = new THREE.SphereGeometry(1, 24, 16);
const cylinderGeo = new THREE.CylinderGeometry(1, 1, 1, 12);

function createViewer(container: HTMLElement, positions: Vec3[], atomicNumbers: number[]) {
  if (positions.length === 0) {
    container.innerHTML = '<p style="color:var(--color-text-dim)">No atoms</p>';
    return;
  }

  let cx = 0, cy = 0, cz = 0;
  for (const p of positions) { cx += p.x; cy += p.y; cz += p.z; }
  cx /= positions.length; cy /= positions.length; cz /= positions.length;
  const centered = positions.map(p => ({ x: p.x - cx, y: p.y - cy, z: p.z - cz }));
  const bonds = detectBonds(centered, atomicNumbers);

  let maxExtent = 0;
  for (const p of centered) {
    const r = Math.sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
    if (r > maxExtent) maxExtent = r;
  }
  const camDist = Math.max(maxExtent * 2.5, 3);

  let atomScale = 1.0;
  let bondScale = 1.0;

  container.innerHTML = '';

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1000);
  camera.position.set(0, 0, camDist);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(400, 400);
  renderer.setPixelRatio(window.devicePixelRatio);
  const canvas = renderer.domElement;
  canvas.style.width = '100%';
  canvas.style.height = '100%';
  canvas.style.cursor = 'grab';
  canvas.style.touchAction = 'none';
  canvas.style.display = 'block';

  const wrapper = document.createElement('div');
  wrapper.style.cssText = 'position:relative;width:100%;aspect-ratio:1;border-radius:8px;overflow:hidden';
  wrapper.appendChild(canvas);

  const controls = document.createElement('div');
  controls.style.cssText = 'position:absolute;bottom:0;left:0;right:0;display:flex;align-items:center;justify-content:center;gap:6px;padding:6px 10px;background:rgba(0,0,0,0.35);backdrop-filter:blur(4px);font-size:11px;color:#eee;pointer-events:auto';
  const sliderStyle = 'width:60px;accent-color:#7ec8e3;height:3px';
  const btnStyle = 'padding:2px 8px;font-size:11px;cursor:pointer;border:1px solid rgba(255,255,255,0.3);border-radius:4px;background:rgba(255,255,255,0.1);color:#eee;transition:background 0.15s,border-color 0.15s';
  const btnActiveStyle = 'background:rgba(126,200,227,0.45);border-color:rgba(126,200,227,0.7)';
  controls.innerHTML =
    `<span style="opacity:0.7">Atom</span><input type="range" min="30" max="200" value="100" style="${sliderStyle}" id="mol-atom-sl">` +
    `<span style="opacity:0.7">Bond</span><input type="range" min="30" max="200" value="100" style="${sliderStyle}" id="mol-bond-sl">` +
    `<button style="${btnStyle}" id="mol-rot-l" title="Rotate left">\u25C0</button>` +
    `<button style="${btnStyle}" id="mol-rot-r" title="Rotate right">\u25B6</button>` +
    `<button style="${btnStyle}" id="mol-rot-u" title="Rotate up">\u25B2</button>` +
    `<button style="${btnStyle}" id="mol-rot-d" title="Rotate down">\u25BC</button>`;
  wrapper.appendChild(controls);
  container.appendChild(wrapper);

  const atomSlider = controls.querySelector<HTMLInputElement>('#mol-atom-sl')!;
  const bondSlider = controls.querySelector<HTMLInputElement>('#mol-bond-sl')!;
  const rotLBtn = controls.querySelector<HTMLButtonElement>('#mol-rot-l')!;
  const rotRBtn = controls.querySelector<HTMLButtonElement>('#mol-rot-r')!;
  const rotUBtn = controls.querySelector<HTMLButtonElement>('#mol-rot-u')!;
  const rotDBtn = controls.querySelector<HTMLButtonElement>('#mol-rot-d')!;
  const rotBtns = [rotLBtn, rotRBtn, rotUBtn, rotDBtn];

  const ro = new ResizeObserver(() => {
    const w = wrapper.clientWidth;
    if (w > 0) { renderer.setSize(w, w); renderer.render(scene, camera); }
  });
  ro.observe(wrapper);

  scene.add(new THREE.AmbientLight(0xffffff, 1.2));
  const dirLight = new THREE.DirectionalLight(0xffffff, 1.0);
  dirLight.position.set(2, 3, 4);
  scene.add(dirLight);
  const dirLight2 = new THREE.DirectionalLight(0xffffff, 0.6);
  dirLight2.position.set(-2, -1, -2);
  scene.add(dirLight2);

  const molGroup = new THREE.Group();
  scene.add(molGroup);

  const atomMeshes: THREE.Mesh[] = [];
  for (let i = 0; i < centered.length; i++) {
    const p = centered[i];
    const z = atomicNumbers[i];
    const r = getAtomRadius(z);
    const mat = new THREE.MeshPhongMaterial({ color: getColor(z), shininess: 60 });
    const mesh = new THREE.Mesh(sphereGeo, mat);
    mesh.position.set(p.x, p.y, p.z);
    mesh.scale.setScalar(r);
    molGroup.add(mesh);
    atomMeshes.push(mesh);
  }

  const bondRadius = 0.08;
  const bondMeshes: THREE.Mesh[] = [];
  for (const bond of bonds) {
    const pi = centered[bond.i], pj = centered[bond.j];
    const a = new THREE.Vector3(pi.x, pi.y, pi.z);
    const b = new THREE.Vector3(pj.x, pj.y, pj.z);
    const mid = a.clone().add(b).multiplyScalar(0.5);
    addCylinder(a, mid, getColor(atomicNumbers[bond.i]));
    addCylinder(mid, b, getColor(atomicNumbers[bond.j]));
  }

  function addCylinder(start: THREE.Vector3, end: THREE.Vector3, color: number) {
    const dir = end.clone().sub(start);
    const len = dir.length();
    if (len < 0.001) return;
    const mat = new THREE.MeshPhongMaterial({ color, shininess: 40 });
    const mesh = new THREE.Mesh(cylinderGeo, mat);
    mesh.position.copy(start.clone().add(end).multiplyScalar(0.5));
    mesh.scale.set(bondRadius, len, bondRadius);
    const axis = new THREE.Vector3(0, 1, 0);
    const quat = new THREE.Quaternion().setFromUnitVectors(axis, dir.normalize());
    mesh.quaternion.copy(quat);
    molGroup.add(mesh);
    bondMeshes.push(mesh);
  }

  const initQ = new THREE.Quaternion();
  initQ.setFromEuler(new THREE.Euler(-0.35, 0.52, 0, 'YXZ'));
  molGroup.quaternion.copy(initQ);

  function updateBackground() {
    scene.background = new THREE.Color(isDark() ? 0x1e1e2e : 0xeef1f5);
  }
  updateBackground();

  let autoRotY = 0; // -1 = left, +1 = right
  let autoRotX = 0; // -1 = up, +1 = down
  let animId = 0;
  const _worldY = new THREE.Vector3(0, 1, 0);
  const _worldX = new THREE.Vector3(1, 0, 0);
  const _tmpQ = new THREE.Quaternion();

  function isRotating() { return autoRotY !== 0 || autoRotX !== 0; }

  function render() {
    if (autoRotY !== 0) {
      _tmpQ.setFromAxisAngle(_worldY, 0.012 * autoRotY);
      molGroup.quaternion.premultiply(_tmpQ);
    }
    if (autoRotX !== 0) {
      _tmpQ.setFromAxisAngle(_worldX, 0.012 * autoRotX);
      molGroup.quaternion.premultiply(_tmpQ);
    }
    renderer.render(scene, camera);
    if (isRotating()) animId = requestAnimationFrame(render);
  }

  function requestRender() {
    if (!isRotating()) renderer.render(scene, camera);
  }

  let dragging = false;
  let lastX = 0, lastY = 0;

  canvas.addEventListener('pointerdown', (e) => {
    dragging = true; lastX = e.clientX; lastY = e.clientY;
    canvas.setPointerCapture(e.pointerId);
    canvas.style.cursor = 'grabbing';
  });
  canvas.addEventListener('pointermove', (e) => {
    if (!dragging) return;
    const dx = e.clientX - lastX, dy = e.clientY - lastY;
    lastX = e.clientX; lastY = e.clientY;
    _tmpQ.setFromAxisAngle(_worldY, dx * 0.01);
    molGroup.quaternion.premultiply(_tmpQ);
    _tmpQ.setFromAxisAngle(_worldX, dy * 0.01);
    molGroup.quaternion.premultiply(_tmpQ);
    requestRender();
  });
  canvas.addEventListener('pointerup', () => { dragging = false; canvas.style.cursor = 'grab'; });
  canvas.addEventListener('pointercancel', () => { dragging = false; canvas.style.cursor = 'grab'; });

  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    const factor = e.deltaY > 0 ? 1.1 : 0.9;
    camera.position.z = Math.max(1, Math.min(camDist * 4, camera.position.z * factor));
    requestRender();
  }, { passive: false });

  function rebuildScales() {
    for (let i = 0; i < atomMeshes.length; i++) {
      atomMeshes[i].scale.setScalar(getAtomRadius(atomicNumbers[i]) * atomScale);
    }
    for (const m of bondMeshes) {
      const sy = m.scale.y;
      m.scale.set(bondRadius * bondScale, sy, bondRadius * bondScale);
    }
    requestRender();
  }

  atomSlider.addEventListener('input', () => { atomScale = parseInt(atomSlider.value, 10) / 100; rebuildScales(); });
  bondSlider.addEventListener('input', () => { bondScale = parseInt(bondSlider.value, 10) / 100; rebuildScales(); });

  function updateRotBtnStyles() {
    for (const btn of rotBtns) btn.style.cssText = btnStyle;
    if (autoRotY === -1) rotLBtn.style.cssText = btnStyle + ';' + btnActiveStyle;
    if (autoRotY === 1) rotRBtn.style.cssText = btnStyle + ';' + btnActiveStyle;
    if (autoRotX === -1) rotUBtn.style.cssText = btnStyle + ';' + btnActiveStyle;
    if (autoRotX === 1) rotDBtn.style.cssText = btnStyle + ';' + btnActiveStyle;
  }

  function toggleRotY(dir: number) {
    autoRotY = autoRotY === dir ? 0 : dir;
    cancelAnimationFrame(animId);
    updateRotBtnStyles();
    if (isRotating()) animId = requestAnimationFrame(render);
    else requestRender();
  }

  function toggleRotX(dir: number) {
    autoRotX = autoRotX === dir ? 0 : dir;
    cancelAnimationFrame(animId);
    updateRotBtnStyles();
    if (isRotating()) animId = requestAnimationFrame(render);
    else requestRender();
  }

  rotLBtn.addEventListener('click', () => toggleRotY(-1));
  rotRBtn.addEventListener('click', () => toggleRotY(1));
  rotUBtn.addEventListener('click', () => toggleRotX(-1));
  rotDBtn.addEventListener('click', () => toggleRotX(1));
  toggleRotY(1);

  const observer = new MutationObserver(() => { updateBackground(); requestRender(); });
  observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
}

/** Render from parsed atom data (from backend result) */
export function renderMoleculeFromAtoms(
  container: HTMLElement,
  atoms: { element: string; coords: number[] }[],
): void {
  const positions: Vec3[] = [];
  const atomicNumbers: number[] = [];
  for (const atom of atoms) {
    const z = ELEMENT_NAME_TO_ATOMIC_NUMBER[atom.element];
    if (z === undefined) continue;
    positions.push({ x: atom.coords[0], y: atom.coords[1], z: atom.coords[2] });
    atomicNumbers.push(z);
  }
  createViewer(container, positions, atomicNumbers);
}

/** Render preview from raw XYZ text (coordinates in Angstrom) */
export function renderMoleculePreview(container: HTMLElement, xyzText: string): void {
  const lines = xyzText.split(/\r?\n/);
  if (lines.length < 3) { container.innerHTML = ''; return; }
  const atomCount = parseInt(lines[0].trim(), 10);
  if (isNaN(atomCount) || atomCount < 1) { container.innerHTML = ''; return; }

  const positions: Vec3[] = [];
  const atomicNumbers: number[] = [];
  for (let i = 2; i < 2 + atomCount && i < lines.length; i++) {
    const parts = lines[i]?.trim().split(/\s+/);
    if (!parts || parts.length < 4) continue;
    const z = ELEMENT_NAME_TO_ATOMIC_NUMBER[parts[0]];
    if (z === undefined) continue;
    const x = parseFloat(parts[1]);
    const y = parseFloat(parts[2]);
    const zz = parseFloat(parts[3]);
    if (isNaN(x) || isNaN(y) || isNaN(zz)) continue;
    positions.push({ x, y, z: zz });
    atomicNumbers.push(z);
  }

  createViewer(container, positions, atomicNumbers);
}
