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

// Annotation types for 3D measurements
export type MolAnnotation =
  | { type: 'distance'; atoms: [number, number] }
  | { type: 'angle'; atoms: [number, number, number] }   // atoms[0] is vertex
  | { type: 'height'; atom: number; planeAtoms: number[] }  // perpendicular distance from atom to plane
  | { type: 'forces'; forces: { x: number; y: number; z: number }[] };  // force vectors per atom (log-scaled)

// Cache viewer state per container to avoid WebGL context leak
interface ViewerState {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  molGroup: THREE.Group;
  atomMeshes: THREE.Mesh[];
  bondMeshes: THREE.Mesh[];
  labelObjects: THREE.Object3D[];  // distance labels + arrows
  atomicNumbers: number[];
  atomScale: number;
  bondScale: number;
  autoRotY: number;
  autoRotX: number;
  animId: number;
  camDist: number;
  lockScale: boolean;
  requestRender: () => void;
}

const viewerCache = new WeakMap<HTMLElement, ViewerState>();

const bondRadius = 0.08;

// Create a text sprite for distance labels
function createTextSprite(text: string): THREE.Sprite {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d')!;
  canvas.width = 128; canvas.height = 48;
  ctx.clearRect(0, 0, 128, 48);
  ctx.fillStyle = 'rgba(0,0,0,0.55)';
  ctx.roundRect(2, 2, 124, 44, 6);
  ctx.fill();
  ctx.font = 'bold 22px monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillStyle = '#ffffff';
  ctx.fillText(text, 64, 24);
  const tex = new THREE.CanvasTexture(canvas);
  tex.minFilter = THREE.LinearFilter;
  const mat = new THREE.SpriteMaterial({ map: tex, depthTest: false });
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(0.8, 0.3, 1);
  return sprite;
}

// Create arrow (cone) mesh
const arrowGeo = new THREE.ConeGeometry(0.06, 0.18, 8);

function addDistanceAnnotation(molGroup: THREE.Group, labels: THREE.Object3D[],
  a: THREE.Vector3, b: THREE.Vector3, dist: number, offset: THREE.Vector3) {
  // Line from atom center to atom center
  const lineGeo = new THREE.BufferGeometry().setFromPoints([a.clone(), b.clone()]);
  const lineMat = new THREE.LineBasicMaterial({ color: 0xffcc00, linewidth: 1, depthTest: false });
  const line = new THREE.Line(lineGeo, lineMat);
  line.renderOrder = 1;
  molGroup.add(line);
  labels.push(line);

  // Arrowheads at atom centers, pointing outward (← →)
  const dir = b.clone().sub(a).normalize();
  // Arrow at a, pointing away from b (toward left)
  const arrowA = new THREE.Mesh(arrowGeo, new THREE.MeshBasicMaterial({ color: 0xffcc00, depthTest: false }));
  arrowA.position.copy(a);
  arrowA.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.clone().negate());
  arrowA.renderOrder = 1;
  molGroup.add(arrowA);
  labels.push(arrowA);

  // Arrow at b, pointing away from a (toward right)
  const arrowB = new THREE.Mesh(arrowGeo, new THREE.MeshBasicMaterial({ color: 0xffcc00, depthTest: false }));
  arrowB.position.copy(b);
  arrowB.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
  arrowB.renderOrder = 1;
  molGroup.add(arrowB);
  labels.push(arrowB);

  // Text label at midpoint, offset perpendicular to avoid overlapping bond
  const mid = a.clone().add(b).multiplyScalar(0.5);
  const sprite = createTextSprite(`${dist.toFixed(3)} \u00C5`);
  sprite.position.copy(mid.clone().add(offset));
  sprite.renderOrder = 2;
  molGroup.add(sprite);
  labels.push(sprite);
}

function addAngleAnnotation(molGroup: THREE.Group, labels: THREE.Object3D[],
  vertex: THREE.Vector3, a: THREE.Vector3, b: THREE.Vector3) {
  // Compute angle
  const va = a.clone().sub(vertex).normalize();
  const vb = b.clone().sub(vertex).normalize();
  const angleDeg = Math.acos(Math.max(-1, Math.min(1, va.dot(vb)))) * 180 / Math.PI;

  // Draw two lines from vertex to each atom
  const lineColor = 0xffcc00;
  for (const target of [a, b]) {
    const geo = new THREE.BufferGeometry().setFromPoints([vertex.clone(), target.clone()]);
    const mat = new THREE.LineBasicMaterial({ color: lineColor, depthTest: false });
    const line = new THREE.Line(geo, mat);
    line.renderOrder = 1;
    molGroup.add(line);
    labels.push(line);
  }

  // Draw arc between the two directions
  const arcRadius = Math.min(0.4, a.distanceTo(vertex) * 0.4, b.distanceTo(vertex) * 0.4);
  const arcPoints: THREE.Vector3[] = [];
  const nSegments = 20;
  const angleRad = angleDeg * Math.PI / 180;
  // Build local frame: va is "x", cross is "z", perpendicular in plane is "y"
  const normal = new THREE.Vector3().crossVectors(va, vb).normalize();
  for (let i = 0; i <= nSegments; i++) {
    const t = (i / nSegments) * angleRad;
    // Rotate va by angle t around normal
    const q = new THREE.Quaternion().setFromAxisAngle(normal, t);
    const pt = va.clone().applyQuaternion(q).multiplyScalar(arcRadius).add(vertex);
    arcPoints.push(pt);
  }
  const arcGeo = new THREE.BufferGeometry().setFromPoints(arcPoints);
  const arcMat = new THREE.LineBasicMaterial({ color: lineColor, depthTest: false });
  const arcLine = new THREE.Line(arcGeo, arcMat);
  arcLine.renderOrder = 1;
  molGroup.add(arcLine);
  labels.push(arcLine);

  // Angle label at arc midpoint
  const midQ = new THREE.Quaternion().setFromAxisAngle(normal, angleRad * 0.5);
  const labelPos = va.clone().applyQuaternion(midQ).multiplyScalar(arcRadius * 1.8).add(vertex);
  const sprite = createTextSprite(`${angleDeg.toFixed(1)}\u00B0`);
  sprite.position.copy(labelPos);
  sprite.renderOrder = 2;
  molGroup.add(sprite);
  labels.push(sprite);
}

function addHeightAnnotation(molGroup: THREE.Group, labels: THREE.Object3D[],
  atomPos: THREE.Vector3, planePositions: THREE.Vector3[]) {
  // Compute plane centroid and normal
  const centroid = new THREE.Vector3();
  for (const p of planePositions) centroid.add(p);
  centroid.divideScalar(planePositions.length);

  // Plane normal from first two edges (assumes >= 3 coplanar points)
  let normal: THREE.Vector3;
  if (planePositions.length >= 3) {
    const e1 = planePositions[1].clone().sub(planePositions[0]);
    const e2 = planePositions[2].clone().sub(planePositions[0]);
    normal = new THREE.Vector3().crossVectors(e1, e2).normalize();
  } else {
    normal = new THREE.Vector3(0, 0, 1);
  }

  // Project atom onto plane
  const diff = atomPos.clone().sub(centroid);
  const signedDist = diff.dot(normal);
  const footPoint = atomPos.clone().sub(normal.clone().multiplyScalar(signedDist));
  const absDist = Math.abs(signedDist);

  const lineColor = 0xffcc00;

  // Vertical line from atom to foot point
  const lineGeo = new THREE.BufferGeometry().setFromPoints([atomPos.clone(), footPoint.clone()]);
  const lineMat = new THREE.LineBasicMaterial({ color: lineColor, depthTest: false });
  const line = new THREE.Line(lineGeo, lineMat);
  line.renderOrder = 1;
  molGroup.add(line);
  labels.push(line);

  // Small right-angle marker at foot point
  const right = diff.clone().sub(normal.clone().multiplyScalar(signedDist)).normalize();
  if (right.length() < 0.01) right.set(1, 0, 0);
  else right.normalize();
  const markSize = 0.08;
  const corner1 = footPoint.clone().add(right.clone().multiplyScalar(markSize));
  const corner2 = corner1.clone().add(normal.clone().multiplyScalar(signedDist > 0 ? markSize : -markSize));
  const corner3 = footPoint.clone().add(normal.clone().multiplyScalar(signedDist > 0 ? markSize : -markSize));
  const markGeo = new THREE.BufferGeometry().setFromPoints([corner1, corner2, corner3]);
  const markLine = new THREE.Line(markGeo, new THREE.LineBasicMaterial({ color: lineColor, depthTest: false }));
  markLine.renderOrder = 1;
  molGroup.add(markLine);
  labels.push(markLine);

  // Arrowheads — outward (away from each other)
  const dir = normal.clone().multiplyScalar(signedDist > 0 ? 1 : -1);
  // Arrow at atom: pointing away from plane (outward = +dir)
  const arrowTop = new THREE.Mesh(arrowGeo, new THREE.MeshBasicMaterial({ color: lineColor, depthTest: false }));
  arrowTop.position.copy(atomPos);
  arrowTop.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
  arrowTop.renderOrder = 1;
  molGroup.add(arrowTop);
  labels.push(arrowTop);

  // Arrow at foot: pointing away from atom (outward = -dir)
  const arrowBot = new THREE.Mesh(arrowGeo, new THREE.MeshBasicMaterial({ color: lineColor, depthTest: false }));
  arrowBot.position.copy(footPoint);
  arrowBot.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.clone().negate());
  arrowBot.renderOrder = 1;
  molGroup.add(arrowBot);
  labels.push(arrowBot);

  // Dashed line showing the plane (triangle outline through planePositions)
  if (planePositions.length >= 3) {
    const planeOutline = [...planePositions, planePositions[0]];
    const planeGeo = new THREE.BufferGeometry().setFromPoints(planeOutline);
    const planeMat = new THREE.LineDashedMaterial({ color: 0x888888, dashSize: 0.08, gapSize: 0.04, depthTest: false });
    const planeLine = new THREE.Line(planeGeo, planeMat);
    planeLine.computeLineDistances();
    planeLine.renderOrder = 0;
    molGroup.add(planeLine);
    labels.push(planeLine);
  }

  // Label
  const mid = atomPos.clone().add(footPoint).multiplyScalar(0.5);
  const offset = right.clone().multiplyScalar(0.3);
  const sprite = createTextSprite(`${absDist.toFixed(3)} \u00C5`);
  sprite.position.copy(mid.add(offset));
  sprite.renderOrder = 2;
  molGroup.add(sprite);
  labels.push(sprite);
}

function addForceVectors(molGroup: THREE.Group, labels: THREE.Object3D[],
  centered: Vec3[], forces: { x: number; y: number; z: number }[]) {
  const forceColor = 0x00cc88;
  const arrowHeadGeo = new THREE.ConeGeometry(0.05, 0.12, 8);

  for (let i = 0; i < centered.length && i < forces.length; i++) {
    const f = forces[i];
    const mag = Math.sqrt(f.x * f.x + f.y * f.y + f.z * f.z);
    if (mag < 1e-10) continue;

    // Log-scale: threshold ~3e-4, below → near zero; 0.1 → ~1.5
    // logLen = 0.6 * log10(mag / 3e-4), clamped [0, 2.0]
    const logLen = Math.max(0, Math.min(2.0, 0.6 * Math.log10(mag / 3e-4)));

    const origin = new THREE.Vector3(centered[i].x, centered[i].y, centered[i].z);
    const dir = new THREE.Vector3(f.x, f.y, f.z).normalize();
    const end = origin.clone().add(dir.clone().multiplyScalar(logLen));

    // Shaft
    const shaftGeo = new THREE.BufferGeometry().setFromPoints([origin, end]);
    const shaftMat = new THREE.LineBasicMaterial({ color: forceColor, linewidth: 2, depthTest: false });
    const shaft = new THREE.Line(shaftGeo, shaftMat);
    shaft.renderOrder = 1;
    molGroup.add(shaft);
    labels.push(shaft);

    // Arrowhead at end
    const arrow = new THREE.Mesh(arrowHeadGeo, new THREE.MeshBasicMaterial({ color: forceColor, depthTest: false }));
    arrow.position.copy(end);
    arrow.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
    arrow.renderOrder = 1;
    molGroup.add(arrow);
    labels.push(arrow);
  }
}

function populateMolGroup(state: ViewerState, centered: Vec3[], atomicNumbers: number[], annotations?: MolAnnotation[]) {
  const { molGroup } = state;

  // Dispose old meshes
  for (const m of state.atomMeshes) {
    (m.material as THREE.Material).dispose();
    molGroup.remove(m);
  }
  for (const m of state.bondMeshes) {
    (m.material as THREE.Material).dispose();
    molGroup.remove(m);
  }
  for (const m of state.labelObjects) {
    if ('material' in m) {
      const mat = (m as THREE.Mesh).material as THREE.Material;
      mat.dispose();
    }
    if ('geometry' in m) {
      (m as THREE.Mesh).geometry.dispose();
    }
    molGroup.remove(m);
  }
  state.atomMeshes = [];
  state.bondMeshes = [];
  state.labelObjects = [];
  state.atomicNumbers = atomicNumbers;

  // Build atoms
  for (let i = 0; i < centered.length; i++) {
    const p = centered[i];
    const z = atomicNumbers[i];
    const r = getAtomRadius(z);
    const mat = new THREE.MeshPhongMaterial({ color: getColor(z), shininess: 60 });
    const mesh = new THREE.Mesh(sphereGeo, mat);
    mesh.position.set(p.x, p.y, p.z);
    mesh.scale.setScalar(r * state.atomScale);
    molGroup.add(mesh);
    state.atomMeshes.push(mesh);
  }

  // Build bonds
  const bonds = detectBonds(centered, atomicNumbers);
  function addCylinder(start: THREE.Vector3, end: THREE.Vector3, color: number) {
    const dir = end.clone().sub(start);
    const len = dir.length();
    if (len < 0.001) return;
    const mat = new THREE.MeshPhongMaterial({ color, shininess: 40 });
    const mesh = new THREE.Mesh(cylinderGeo, mat);
    mesh.position.copy(start.clone().add(end).multiplyScalar(0.5));
    mesh.scale.set(bondRadius * state.bondScale, len, bondRadius * state.bondScale);
    const axis = new THREE.Vector3(0, 1, 0);
    const quat = new THREE.Quaternion().setFromUnitVectors(axis, dir.normalize());
    mesh.quaternion.copy(quat);
    molGroup.add(mesh);
    state.bondMeshes.push(mesh);
  }

  for (const bond of bonds) {
    const pi = centered[bond.i], pj = centered[bond.j];
    const a = new THREE.Vector3(pi.x, pi.y, pi.z);
    const b = new THREE.Vector3(pj.x, pj.y, pj.z);
    const mid = a.clone().add(b).multiplyScalar(0.5);
    addCylinder(a, mid, getColor(atomicNumbers[bond.i]));
    addCylinder(mid, b, getColor(atomicNumbers[bond.j]));
  }

  // Annotations: if provided, show only specified measurements; otherwise all pairs for <=2 atoms
  if (annotations) {
    let annIdx = 0;
    for (const ann of annotations) {
      if (ann.type === 'distance') {
        const [ai, aj] = ann.atoms;
        if (ai < centered.length && aj < centered.length) {
          const pi = centered[ai], pj = centered[aj];
          const a = new THREE.Vector3(pi.x, pi.y, pi.z);
          const b = new THREE.Vector3(pj.x, pj.y, pj.z);
          const dist = a.distanceTo(b);
          const pairDir = b.clone().sub(a).normalize();
          const ref = Math.abs(pairDir.y) < 0.9 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0);
          const perp = new THREE.Vector3().crossVectors(pairDir, ref).normalize();
          const offset = perp.multiplyScalar(0.3 * (annIdx % 2 === 0 ? 1 : -1));
          addDistanceAnnotation(molGroup, state.labelObjects, a, b, dist, offset);
          annIdx++;
        }
      } else if (ann.type === 'angle') {
        const [vi, ai, bi] = ann.atoms;
        if (vi < centered.length && ai < centered.length && bi < centered.length) {
          const v = new THREE.Vector3(centered[vi].x, centered[vi].y, centered[vi].z);
          const a = new THREE.Vector3(centered[ai].x, centered[ai].y, centered[ai].z);
          const b = new THREE.Vector3(centered[bi].x, centered[bi].y, centered[bi].z);
          addAngleAnnotation(molGroup, state.labelObjects, v, a, b);
        }
      } else if (ann.type === 'height') {
        const atomIdx = ann.atom;
        const planeIdxs = ann.planeAtoms;
        if (atomIdx < centered.length && planeIdxs.every(pi => pi < centered.length)) {
          const atomPos = new THREE.Vector3(centered[atomIdx].x, centered[atomIdx].y, centered[atomIdx].z);
          const planePos = planeIdxs.map(pi => new THREE.Vector3(centered[pi].x, centered[pi].y, centered[pi].z));
          addHeightAnnotation(molGroup, state.labelObjects, atomPos, planePos);
        }
      } else if (ann.type === 'forces') {
        addForceVectors(molGroup, state.labelObjects, centered, ann.forces);
      }
    }
  } else {
    // Default: all pairs (for 2-atom molecules)
    let pairIdx = 0;
    for (let i = 0; i < centered.length; i++) {
      for (let j = i + 1; j < centered.length; j++) {
        const pi = centered[i], pj = centered[j];
        const a = new THREE.Vector3(pi.x, pi.y, pi.z);
        const b = new THREE.Vector3(pj.x, pj.y, pj.z);
        const dist = a.distanceTo(b);
        const pairDir = b.clone().sub(a).normalize();
        const ref = Math.abs(pairDir.y) < 0.9 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0);
        const perp = new THREE.Vector3().crossVectors(pairDir, ref).normalize();
        const offset = perp.multiplyScalar(0.3 * (pairIdx % 2 === 0 ? 1 : -1));
        addDistanceAnnotation(molGroup, state.labelObjects, a, b, dist, offset);
        pairIdx++;
      }
    }
  }

  // Auto-scale camera unless locked — include force vector tips in extent
  if (!state.lockScale) {
    let maxExtent = 0;
    for (const p of centered) {
      const r = Math.sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
      if (r > maxExtent) maxExtent = r;
    }
    // Include force vector endpoints
    if (annotations) {
      for (const ann of annotations) {
        if (ann.type === 'forces') {
          for (let i = 0; i < centered.length && i < ann.forces.length; i++) {
            const f = ann.forces[i];
            const mag = Math.sqrt(f.x * f.x + f.y * f.y + f.z * f.z);
            const logLen = Math.max(0, Math.min(2.0, 0.6 * Math.log10(mag / 3e-4)));
            if (logLen > 0) {
              const dir = { x: f.x / mag, y: f.y / mag, z: f.z / mag };
              const tx = centered[i].x + dir.x * logLen;
              const ty = centered[i].y + dir.y * logLen;
              const tz = centered[i].z + dir.z * logLen;
              const tr = Math.sqrt(tx * tx + ty * ty + tz * tz);
              if (tr > maxExtent) maxExtent = tr;
            }
          }
        }
      }
    }
    state.camDist = Math.max(maxExtent * 2.5, 3);
    state.camera.position.z = state.camDist;
  }

  state.requestRender();
}

function createViewer(container: HTMLElement, positions: Vec3[], atomicNumbers: number[], annotations?: MolAnnotation[]) {
  if (positions.length === 0) {
    container.innerHTML = '<p style="color:var(--color-text-dim)">No atoms</p>';
    return;
  }

  let cx = 0, cy = 0, cz = 0;
  for (const p of positions) { cx += p.x; cy += p.y; cz += p.z; }
  cx /= positions.length; cy /= positions.length; cz /= positions.length;
  const centered = positions.map(p => ({ x: p.x - cx, y: p.y - cy, z: p.z - cz }));

  // If we already have a viewer for this container, just update the molecule
  const existing = viewerCache.get(container);
  if (existing) {
    populateMolGroup(existing, centered, atomicNumbers, annotations);
    return;
  }

  let maxExtent = 0;
  for (const p of centered) {
    const r = Math.sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
    if (r > maxExtent) maxExtent = r;
  }
  const camDist = Math.max(maxExtent * 2.5, 3);

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

  const state: ViewerState = {
    renderer, scene, camera, molGroup,
    atomMeshes: [], bondMeshes: [], labelObjects: [], atomicNumbers: [],
    atomScale: 1.0, bondScale: 1.0,
    autoRotY: 0, autoRotX: 0, animId: 0, camDist, lockScale: false,
    requestRender: () => {},
  };
  viewerCache.set(container, state);

  const initQ = new THREE.Quaternion();
  initQ.setFromEuler(new THREE.Euler(-0.35, 0.52, 0, 'YXZ'));
  molGroup.quaternion.copy(initQ);

  function updateBackground() {
    scene.background = new THREE.Color(isDark() ? 0x1e1e2e : 0xeef1f5);
  }
  updateBackground();

  const _worldY = new THREE.Vector3(0, 1, 0);
  const _worldX = new THREE.Vector3(1, 0, 0);
  const _tmpQ = new THREE.Quaternion();

  function isRotating() { return state.autoRotY !== 0 || state.autoRotX !== 0; }

  function render() {
    if (state.autoRotY !== 0) {
      _tmpQ.setFromAxisAngle(_worldY, 0.012 * state.autoRotY);
      molGroup.quaternion.premultiply(_tmpQ);
    }
    if (state.autoRotX !== 0) {
      _tmpQ.setFromAxisAngle(_worldX, 0.012 * state.autoRotX);
      molGroup.quaternion.premultiply(_tmpQ);
    }
    renderer.render(scene, camera);
    if (isRotating()) state.animId = requestAnimationFrame(render);
  }

  function requestRender() {
    if (!isRotating()) renderer.render(scene, camera);
  }
  state.requestRender = requestRender;

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
    camera.position.z = Math.max(1, Math.min(state.camDist * 4, camera.position.z * factor));
    requestRender();
  }, { passive: false });

  function rebuildScales() {
    for (let i = 0; i < state.atomMeshes.length; i++) {
      state.atomMeshes[i].scale.setScalar(getAtomRadius(state.atomicNumbers[i]) * state.atomScale);
    }
    for (const m of state.bondMeshes) {
      const sy = m.scale.y;
      m.scale.set(bondRadius * state.bondScale, sy, bondRadius * state.bondScale);
    }
    requestRender();
  }

  atomSlider.addEventListener('input', () => { state.atomScale = parseInt(atomSlider.value, 10) / 100; rebuildScales(); });
  bondSlider.addEventListener('input', () => { state.bondScale = parseInt(bondSlider.value, 10) / 100; rebuildScales(); });

  function updateRotBtnStyles() {
    for (const btn of rotBtns) btn.style.cssText = btnStyle;
    if (state.autoRotY === -1) rotLBtn.style.cssText = btnStyle + ';' + btnActiveStyle;
    if (state.autoRotY === 1) rotRBtn.style.cssText = btnStyle + ';' + btnActiveStyle;
    if (state.autoRotX === -1) rotUBtn.style.cssText = btnStyle + ';' + btnActiveStyle;
    if (state.autoRotX === 1) rotDBtn.style.cssText = btnStyle + ';' + btnActiveStyle;
  }

  function toggleRotY(dir: number) {
    state.autoRotY = state.autoRotY === dir ? 0 : dir;
    cancelAnimationFrame(state.animId);
    updateRotBtnStyles();
    if (isRotating()) state.animId = requestAnimationFrame(render);
    else requestRender();
  }

  function toggleRotX(dir: number) {
    state.autoRotX = state.autoRotX === dir ? 0 : dir;
    cancelAnimationFrame(state.animId);
    updateRotBtnStyles();
    if (isRotating()) state.animId = requestAnimationFrame(render);
    else requestRender();
  }

  rotLBtn.addEventListener('click', () => toggleRotY(-1));
  rotRBtn.addEventListener('click', () => toggleRotY(1));
  rotUBtn.addEventListener('click', () => toggleRotX(-1));
  rotDBtn.addEventListener('click', () => toggleRotX(1));

  // Populate initial molecule and render
  populateMolGroup(state, centered, atomicNumbers, annotations);

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
export function renderMoleculePreview(container: HTMLElement, xyzText: string, annotations?: MolAnnotation[]): void {
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

  createViewer(container, positions, atomicNumbers, annotations);
}

/** Lock camera scale (call after scan completes so slider interaction keeps fixed zoom) */
export function lockMoleculeScale(container: HTMLElement): void {
  const state = viewerCache.get(container);
  if (state) state.lockScale = true;
}

/** Unlock camera scale (call before scan starts so it auto-adjusts) */
export function unlockMoleculeScale(container: HTMLElement): void {
  const state = viewerCache.get(container);
  if (state) state.lockScale = false;
}
