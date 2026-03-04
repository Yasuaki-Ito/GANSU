/** Settings panel — method, basis, charge, convergence, post-HF */

import type { CalculationParams } from '../types';
import { DEFAULT_PARAMS } from '../types';

export interface AuxBasis { name: string; dir: string; }

export function createSettingsPanel(
  container: HTMLElement,
  basisSets: string[],
  auxBasisSets: AuxBasis[] = [],
): { getParams: () => Partial<CalculationParams> } {
  container.innerHTML = `
    <div class="panel">
      <h2>Settings</h2>

      <div class="setting-row">
        <label>Method</label>
        <div class="toggle-group" id="method-group">
          <button class="toggle active" data-value="RHF">RHF</button>
          <button class="toggle" data-value="UHF">UHF</button>
          <button class="toggle" data-value="ROHF">ROHF</button>
        </div>
      </div>

      <div class="setting-row">
        <label>Basis Set</label>
        <select id="basis-select">
          ${basisSets.map(b => `<option value="${b}" ${b === 'sto-3g' ? 'selected' : ''}>${b}</option>`).join('')}
        </select>
      </div>

      <div class="setting-row">
        <label>Charge</label>
        <div class="toggle-group" id="charge-group">
          <button class="toggle" data-value="-2">-2</button>
          <button class="toggle" data-value="-1">-1</button>
          <button class="toggle active" data-value="0">0</button>
          <button class="toggle" data-value="1">+1</button>
          <button class="toggle" data-value="2">+2</button>
        </div>
      </div>

      <div class="setting-row hidden" id="rohf-param-row">
        <label>ROHF Param</label>
        <select id="rohf-param-select">
          <option value="Roothaan" selected>Roothaan</option>
          <option value="McWeeny-Diercksen">McWeeny-Diercksen</option>
          <option value="Davidson">Davidson</option>
          <option value="Guest-Saunders">Guest-Saunders</option>
          <option value="Binkley-Pople-Dobosh">Binkley-Pople-Dobosh</option>
          <option value="Faegri-Manne">Faegri-Manne</option>
          <option value="Goddard">Goddard</option>
          <option value="Plakhutin-Gorelik-Breslavskaya">Plakhutin-Gorelik-Breslavskaya</option>
        </select>
      </div>

      <div class="setting-row">
        <label>Multiplicity</label>
        <div class="toggle-group" id="mult-group">
          <button class="toggle active" data-value="0">Singlet</button>
          <button class="toggle" data-value="1">Doublet</button>
          <button class="toggle" data-value="2">Triplet</button>
        </div>
      </div>

      <div class="setting-row">
        <label>Initial Guess</label>
        <div class="toggle-group" id="guess-group">
          <button class="toggle active" data-value="core">Core H</button>
          <button class="toggle" data-value="sad">SAD</button>
          <button class="toggle" data-value="gwh">GWH</button>
        </div>
      </div>

      <div class="setting-row">
        <label>Convergence</label>
        <div class="toggle-group" id="conv-group">
          <button class="toggle active" data-value="diis">DIIS</button>
          <button class="toggle" data-value="damping">Damping</button>
          <button class="toggle" data-value="optimal_damping">Optimal Damping</button>
        </div>
      </div>

      <div class="setting-row" id="diis-params">
        <label>DIIS History</label>
        <input type="number" id="diis-size" value="8" min="2" max="20" class="num-input">
        <label class="checkbox-label" style="margin-top:4px"><input type="checkbox" id="chk-diis-transform"> Include transform</label>
      </div>

      <div class="setting-row hidden" id="damping-params">
        <label>Damping Factor</label>
        <input type="number" id="damping-factor" value="0.9" min="0.05" max="0.95" step="0.05" class="num-input">
      </div>

      <div class="setting-row">
        <label>ERI Method</label>
        <div class="toggle-group" id="eri-group">
          <button class="toggle active" data-value="stored">Stored</button>
          <button class="toggle" data-value="direct">Direct</button>
          <button class="toggle" data-value="ri">RI</button>
          <button class="toggle" data-value="direct_ri">Direct RI</button>
        </div>
      </div>

      <div class="setting-row hidden" id="aux-basis-row">
        <label>Auxiliary Basis</label>
        <select id="aux-basis-select">
          <option value="">-- Select --</option>
          ${auxBasisSets.map(b => `<option value="${b.name}" data-dir="${b.dir}">${b.dir === 'auxiliary_basis' ? 'aux_basis' : 'basis'}/${b.name}</option>`).join('')}
        </select>
      </div>

      <div class="setting-row">
        <label>Post-HF</label>
        <div class="toggle-group" id="posthf-group">
          <button class="toggle active" data-value="none">None</button>
          <button class="toggle" data-value="mp2">MP2</button>
          <button class="toggle" data-value="mp3">MP3</button>
          <button class="toggle" data-value="ccsd">CCSD</button>
          <button class="toggle" data-value="ccsd_t">CCSD(T)</button>
          <button class="toggle" data-value="fci">FCI</button>
        </div>
      </div>

      <div class="setting-row">
        <label>Analysis</label>
        <div class="checkbox-group">
          <label class="checkbox-label"><input type="checkbox" id="chk-mulliken"> Mulliken</label>
          <label class="checkbox-label"><input type="checkbox" id="chk-mayer"> Mayer</label>
          <label class="checkbox-label"><input type="checkbox" id="chk-wiberg"> Wiberg</label>
          <label class="checkbox-label"><input type="checkbox" id="chk-molden"> Molden</label>
        </div>
      </div>


      <details class="advanced-settings">
        <summary>Advanced</summary>
        <div class="setting-row">
          <label>Max Iterations</label>
          <input type="number" id="maxiter" value="100" min="1" max="1000" class="num-input">
        </div>
        <div class="setting-row">
          <label>Energy Threshold</label>
          <input type="text" id="conv-thresh" value="1e-6" class="num-input">
        </div>
        <div class="setting-row">
          <label>Schwarz Threshold</label>
          <input type="text" id="schwarz-thresh" value="1e-12" class="num-input">
        </div>
        <div class="setting-row">
          <label>Timeout (s)</label>
          <input type="number" id="timeout" value="600" min="10" max="3600" class="num-input">
        </div>
      </details>
    </div>
  `;

  // Toggle group logic
  container.querySelectorAll('.toggle-group').forEach(group => {
    group.addEventListener('click', (e) => {
      const btn = (e.target as HTMLElement).closest('.toggle') as HTMLButtonElement;
      if (!btn) return;
      group.querySelectorAll('.toggle').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
    });
  });

  // Show/hide convergence params
  const convGroup = container.querySelector('#conv-group')!;
  const diisParams = container.querySelector<HTMLElement>('#diis-params')!;
  const dampingParams = container.querySelector<HTMLElement>('#damping-params')!;
  convGroup.addEventListener('click', () => {
    const active = convGroup.querySelector('.active') as HTMLElement;
    const val = active?.dataset.value;
    diisParams.classList.toggle('hidden', val !== 'diis');
    dampingParams.classList.toggle('hidden', val !== 'damping');
  });

  // Show/hide ROHF parameter when method changes
  const methodGroup = container.querySelector('#method-group')!;
  const rohfParamRow = container.querySelector<HTMLElement>('#rohf-param-row')!;
  methodGroup.addEventListener('click', () => {
    const active = methodGroup.querySelector('.active') as HTMLElement;
    rohfParamRow.classList.toggle('hidden', active?.dataset.value !== 'ROHF');
  });

  // Show/hide auxiliary basis when ERI method changes
  const eriGroup = container.querySelector('#eri-group')!;
  const auxBasisRow = container.querySelector<HTMLElement>('#aux-basis-row')!;
  eriGroup.addEventListener('click', () => {
    const active = eriGroup.querySelector('.active') as HTMLElement;
    const val = active?.dataset.value;
    auxBasisRow.classList.toggle('hidden', val !== 'ri' && val !== 'direct_ri');
  });

  function getToggleValue(groupId: string): string {
    const active = container.querySelector(`#${groupId} .toggle.active`) as HTMLElement;
    return active?.dataset.value || '';
  }

  function getChecked(id: string): boolean {
    return (container.querySelector<HTMLInputElement>(`#${id}`)?.checked) || false;
  }

  function getNumValue(id: string, def: number): number {
    const val = parseFloat((container.querySelector<HTMLInputElement>(`#${id}`)?.value) || '');
    return isNaN(val) ? def : val;
  }

  return {
    getParams: (): Partial<CalculationParams> => ({
      method: getToggleValue('method-group') || DEFAULT_PARAMS.method,
      basis: (container.querySelector<HTMLSelectElement>('#basis-select')?.value) || DEFAULT_PARAMS.basis,
      charge: parseInt(getToggleValue('charge-group') || '0', 10),
      beta_to_alpha: parseInt(getToggleValue('mult-group') || '0', 10),
      initial_guess: getToggleValue('guess-group') || DEFAULT_PARAMS.initial_guess,
      convergence_method: getToggleValue('conv-group') || DEFAULT_PARAMS.convergence_method,
      diis_size: getNumValue('diis-size', DEFAULT_PARAMS.diis_size),
      diis_include_transform: getChecked('chk-diis-transform'),
      damping_factor: getNumValue('damping-factor', DEFAULT_PARAMS.damping_factor),
      rohf_parameter_name: (container.querySelector<HTMLSelectElement>('#rohf-param-select')?.value) || 'Roothaan',
      eri_method: getToggleValue('eri-group') || DEFAULT_PARAMS.eri_method,
      auxiliary_basis: (container.querySelector<HTMLSelectElement>('#aux-basis-select')?.value) || '',
      auxiliary_basis_dir: (container.querySelector<HTMLSelectElement>('#aux-basis-select')?.selectedOptions[0]?.dataset.dir) || 'auxiliary_basis',
      post_hf_method: getToggleValue('posthf-group') || DEFAULT_PARAMS.post_hf_method,
      mulliken: getChecked('chk-mulliken'),
      mayer: getChecked('chk-mayer'),
      wiberg: getChecked('chk-wiberg'),
      export_molden: getChecked('chk-molden'),
      maxiter: getNumValue('maxiter', DEFAULT_PARAMS.maxiter),
      convergence_energy_threshold: getNumValue('conv-thresh', DEFAULT_PARAMS.convergence_energy_threshold),
      schwarz_screening_threshold: getNumValue('schwarz-thresh', DEFAULT_PARAMS.schwarz_screening_threshold),
      timeout: getNumValue('timeout', DEFAULT_PARAMS.timeout),
    }),
  };
}
