/** Progress tracker modal — overlay popup showing calculation steps (adapted from GANSU-web) */

import type { ProgressEvent } from '../types';

type StepStatus = 'pending' | 'active' | 'done' | 'error';

interface StepState {
  el: HTMLElement;
  iconEl: HTMLElement;
  labelEl: HTMLElement;
  detailEl: HTMLElement;
  timeEl: HTMLElement;
  status: StepStatus;
  startTime: number;
  timerHandle: number;
}

interface StepDef {
  id: string;
  label: string;
}

function formatTime(seconds: number): string {
  if (seconds < 0.01) return '<0.01s';
  if (seconds < 10) return seconds.toFixed(2) + 's';
  if (seconds < 60) return seconds.toFixed(1) + 's';
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}m ${s.toFixed(0)}s`;
}

export class ProgressTracker {
  private overlay: HTMLElement;
  private totalTimeEl: HTMLElement;
  private titleEl: HTMLElement;
  private cancelBtn: HTMLButtonElement;
  private t0: number;
  private totalTimer: number;
  private steps: Map<string, StepState> = new Map();
  private onCancel?: () => void;

  constructor(stepDefs: StepDef[], onCancel?: () => void) {
    this.onCancel = onCancel;
    this.t0 = performance.now();

    this.overlay = document.createElement('div');
    this.overlay.className = 'pt-overlay';

    const card = document.createElement('div');
    card.className = 'pt-card';

    // Title
    const titleRow = document.createElement('div');
    titleRow.className = 'pt-title-row';
    this.titleEl = document.createElement('div');
    this.titleEl.className = 'pt-title';
    this.titleEl.innerHTML = '<span class="pt-title-icon"></span> RUNNING';

    const badge = document.createElement('span');
    badge.className = 'pt-badge active';
    badge.textContent = 'GPU';
    titleRow.appendChild(this.titleEl);
    titleRow.appendChild(badge);
    card.appendChild(titleRow);

    // Steps
    const stepsContainer = document.createElement('div');
    stepsContainer.className = 'pt-steps';

    for (const def of stepDefs) {
      const el = document.createElement('div');
      el.className = 'pt-step pending';

      const iconEl = document.createElement('span');
      iconEl.className = 'pt-icon';

      const labelEl = document.createElement('span');
      labelEl.className = 'pt-label';
      labelEl.textContent = def.label;

      const right = document.createElement('span');
      right.className = 'pt-right';
      const detailEl = document.createElement('span');
      detailEl.className = 'pt-detail';
      const timeEl = document.createElement('span');
      timeEl.className = 'pt-time';
      right.appendChild(detailEl);
      right.appendChild(timeEl);

      el.appendChild(iconEl);
      el.appendChild(labelEl);
      el.appendChild(right);
      stepsContainer.appendChild(el);

      this.steps.set(def.id, {
        el, iconEl, labelEl, detailEl, timeEl,
        status: 'pending', startTime: 0, timerHandle: 0,
      });
    }
    card.appendChild(stepsContainer);

    // Footer
    const footer = document.createElement('div');
    footer.className = 'pt-footer';
    this.cancelBtn = document.createElement('button');
    this.cancelBtn.className = 'pt-cancel';
    this.cancelBtn.textContent = 'Cancel';
    this.cancelBtn.addEventListener('click', () => this.onCancel?.());
    footer.appendChild(this.cancelBtn);
    this.totalTimeEl = document.createElement('span');
    this.totalTimeEl.className = 'pt-total';
    footer.appendChild(this.totalTimeEl);
    card.appendChild(footer);

    this.overlay.appendChild(card);
    document.body.appendChild(this.overlay);

    requestAnimationFrame(() => this.overlay.classList.add('visible'));

    this.totalTimer = window.setInterval(() => {
      this.totalTimeEl.textContent = formatTime((performance.now() - this.t0) / 1000);
    }, 100);
  }

  startStep(id: string, detail?: string): void {
    const s = this.steps.get(id);
    if (!s) return;
    if (s.status === 'active') {
      // Already active — just update detail, don't create duplicate timer
      if (detail) s.detailEl.textContent = detail;
      return;
    }
    s.status = 'active';
    s.el.className = 'pt-step active';
    s.startTime = performance.now();
    if (detail) s.detailEl.textContent = detail;
    s.timerHandle = window.setInterval(() => {
      s.timeEl.textContent = formatTime((performance.now() - s.startTime) / 1000);
    }, 100);
  }

  updateStep(id: string, detail: string): void {
    const s = this.steps.get(id);
    if (!s) return;
    if (s.status === 'pending') this.startStep(id, detail);
    else s.detailEl.textContent = detail;
  }

  completeStep(id: string, detail?: string): void {
    const s = this.steps.get(id);
    if (!s || s.status === 'done') return;
    clearInterval(s.timerHandle);
    s.status = 'done';
    s.el.className = 'pt-step done';
    if (detail) s.detailEl.textContent = detail;
    if (s.startTime > 0) {
      s.timeEl.textContent = formatTime((performance.now() - s.startTime) / 1000);
    }
  }

  handleProgress(event: ProgressEvent): void {
    if (event.stage === 'setup') {
      if (event.iteration === 0) this.startStep('setup', 'Initializing...');
      else this.completeStep('setup');
      return;
    }
    if (event.stage === 'integrals') {
      if (event.iteration === 0) this.startStep('integrals', 'Computing ERI...');
      else this.completeStep('integrals');
      return;
    }
    if (event.stage === 'scf') {
      const de = event.delta_e ?? 0;
      if (event.iteration === 0) {
        // Auto-complete preceding steps
        if (this.steps.get('setup')?.status === 'active') this.completeStep('setup');
        if (this.steps.get('integrals')?.status === 'active') this.completeStep('integrals');
        this.startStep('scf', 'iter 0');
      } else {
        this.updateStep('scf', `iter ${event.iteration}  ΔE=${de.toExponential(2)}`);
      }
    } else if (event.stage === 'posthf') {
      if (event.iteration === 0) {
        // Post-HF starting — complete SCF
        if (this.steps.get('scf')?.status === 'active') this.completeStep('scf');
        this.startStep('posthf', 'Starting...');
      } else {
        this.completeStep('posthf');
      }
    } else if (event.stage === 'ccsd') {
      if (this.steps.get('scf')?.status === 'active') this.completeStep('scf');
      this.updateStep('posthf',
        `CCSD iter ${event.iteration}  ΔE=${(event.delta_e ?? 0).toExponential(2)}`);
    } else if (event.stage === 'ccsd_lambda') {
      this.updateStep('posthf',
        `Λ iter ${event.iteration}  ‖Δλ‖=${(event.residual ?? 0).toExponential(2)}`);
    } else if (event.stage === 'excited') {
      // Excited state sub-steps (MO transform, operator build, solver)
      if (this.steps.get('scf')?.status === 'active') this.completeStep('scf');
      if (this.steps.get('posthf')?.status === 'active') this.completeStep('posthf');
      const labels: Record<number, string> = {
        0: 'MO transform...',
        1: 'Building operator...',
        2: 'Solving eigenstates...',
      };
      this.startStep('excited', labels[event.iteration] || 'Computing...');
    } else if (event.stage === 'davidson') {
      if (this.steps.get('posthf')?.status === 'active') this.completeStep('posthf');
      this.updateStep('excited',
        `Davidson iter ${event.iteration}  max|r|=${(event.max_residual ?? 0).toExponential(2)}`);
    } else if (event.stage === 'schur') {
      this.updateStep('excited',
        event.iteration === 0 ? 'Schur diagonalization...' : 'Schur done');
    } else if (event.stage === 'schur_omega') {
      const vals = [event.total_energy, event.delta_e, event.correlation_energy].filter(v => v !== undefined);
      const root = vals[0] !== undefined ? Math.floor(vals[0]) : '?';
      const omega = vals[1] !== undefined ? Number(vals[1]).toFixed(6) : '';
      const delta = vals[2] !== undefined ? Number(vals[2]).toExponential(2) : '';
      this.updateStep('excited', `Root ${root} \u03C9=${omega} \u0394\u03C9=${delta}`);
    }
  }

  complete(): void {
    // Complete all active/pending steps
    for (const [, s] of this.steps) {
      if (s.status === 'active' || s.status === 'pending') {
        clearInterval(s.timerHandle);
        s.status = 'done';
        s.el.className = 'pt-step done';
        if (s.startTime > 0) {
          s.timeEl.textContent = formatTime((performance.now() - s.startTime) / 1000);
        }
      }
    }
    this.titleEl.innerHTML = '<span class="pt-title-icon" style="background:var(--color-converged,#10b981)"></span> COMPLETE';
    this.cancelBtn.textContent = 'Close';
    this.cancelBtn.onclick = () => this.close();
    setTimeout(() => this.close(), 1500);
  }

  fail(_error: string): void {
    for (const [, s] of this.steps) {
      if (s.status === 'active' || s.status === 'pending') {
        clearInterval(s.timerHandle);
        s.status = 'error';
        s.el.className = 'pt-step error';
      }
    }
    this.titleEl.innerHTML = '<span class="pt-title-icon" style="background:var(--color-error,#e74c3c)"></span> ERROR';
    this.cancelBtn.textContent = 'Close';
    this.cancelBtn.onclick = () => this.close();
    // Auto-close after 3 seconds
    setTimeout(() => this.close(), 3000);
  }

  close(): void {
    clearInterval(this.totalTimer);
    for (const s of this.steps.values()) clearInterval(s.timerHandle);
    this.overlay.classList.remove('visible');
    this.overlay.addEventListener('transitionend', () => this.overlay.remove(), { once: true });
    setTimeout(() => { if (this.overlay.parentNode) this.overlay.remove(); }, 500);
  }
}

/** Build step definitions based on calculation parameters */
export function buildSteps(postHf: string, eriMethod: string = 'stored', runType: string = 'energy'): StepDef[] {
  const steps: StepDef[] = [
    { id: 'setup', label: 'Setup' },
  ];

  // Integrals
  if (eriMethod === 'ri' || eriMethod === 'RI') {
    steps.push({ id: 'ri-2c', label: 'RI: 2-Center Integrals' });
    steps.push({ id: 'ri-3c', label: 'RI: 3-Center Integrals' });
    steps.push({ id: 'ri-b', label: 'RI: B Matrix' });
  } else {
    steps.push({ id: 'integrals', label: 'Integrals' });
  }

  // SCF
  steps.push({ id: 'scf', label: 'SCF Loop' });

  // Post-HF
  const methodLabels: Record<string, string> = {
    mp2: 'MP2 Correlation', mp3: 'MP3 Correlation', mp4: 'MP4 Correlation',
    cc2: 'CC2 Correlation', ccsd: 'CCSD Correlation', ccsd_t: 'CCSD(T) Correlation',
    ccsd_density: 'CCSD + Lambda', fci: 'Full CI',
  };
  const excitedMethods: Record<string, string> = {
    cis: 'CIS Excited States', adc2: 'ADC(2) Excited States',
    adc2x: 'ADC(2)-x Excited States', eom_mp2: 'EOM-MP2 Excited States',
    eom_cc2: 'EOM-CC2 Excited States', eom_ccsd: 'EOM-CCSD Excited States',
  };

  if (postHf in methodLabels) {
    steps.push({ id: 'posthf', label: methodLabels[postHf] });
  }
  if (postHf in excitedMethods) {
    // EOM methods need CCSD first
    if (postHf === 'eom_ccsd') {
      steps.push({ id: 'posthf', label: 'CCSD Ground State' });
    }
    steps.push({ id: 'excited', label: excitedMethods[postHf] });
  }

  // Gradient / Hessian
  if (runType === 'gradient') {
    steps.push({ id: 'gradient', label: 'Nuclear Gradient' });
  } else if (runType === 'hessian') {
    steps.push({ id: 'hessian', label: 'Hessian / Frequencies' });
  } else if (runType === 'optimize') {
    steps.push({ id: 'optimize', label: 'Geometry Optimization' });
  }

  // Properties
  steps.push({ id: 'properties', label: 'Properties' });

  return steps;
}
