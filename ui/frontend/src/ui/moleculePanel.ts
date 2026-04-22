/** Molecule input panel — sample selector, XYZ textarea, 3D preview */

import type { SampleMolecule } from '../types';
import { fetchSamples, fetchSampleContent } from '../api';
import { renderMoleculePreview } from '../viz/moleculeViewer3D';

export function createMoleculePanel(
  container: HTMLElement,
  sampleDirs: string[],
  initialSamples: SampleMolecule[],
  onXyzChange: (xyz: string) => void,
): { getXyz: () => string; getXyzFile: () => string; getXyzDir: () => string; setXyz: (xyz: string) => void } {
  let selectedFile = '';
  let currentDir = sampleDirs[0] || '.';

  container.innerHTML = `
    <div class="panel">
      <h2>Molecule</h2>
      ${sampleDirs.length > 1 ? `
      <div class="form-group">
        <label>Directory</label>
        <div class="toggle-group" id="dir-group">
          ${sampleDirs.map(d => `<button class="toggle ${d === currentDir ? 'active' : ''}" data-value="${d}">${d === '.' ? 'small' : d}</button>`).join('')}
        </div>
      </div>
      ` : ''}
      <div class="form-group">
        <label for="sample-select">Sample</label>
        <select id="sample-select">
          <option value="">-- Custom input --</option>
          ${initialSamples.map(s => `<option value="${s.filename}">${s.name} (${s.filename})</option>`).join('')}
        </select>
      </div>
      <div class="form-group">
        <label for="xyz-input">XYZ</label>
        <textarea id="xyz-input" rows="8" placeholder="2\n\nH  0.0 0.0 0.0\nH  0.0 0.0 0.74" spellcheck="false"></textarea>
        <p class="hint">Drag & drop .xyz file or paste text</p>
      </div>
      <div id="mol-preview"></div>
    </div>
  `;

  const select = container.querySelector<HTMLSelectElement>('#sample-select')!;
  const textarea = container.querySelector<HTMLTextAreaElement>('#xyz-input')!;
  const preview = container.querySelector<HTMLDivElement>('#mol-preview')!;

  // Directory switching
  const dirGroup = container.querySelector('#dir-group');
  if (dirGroup) {
    dirGroup.addEventListener('click', async (e) => {
      const btn = (e.target as HTMLElement).closest('.toggle') as HTMLButtonElement;
      if (!btn || !btn.dataset.value) return;
      dirGroup.querySelectorAll('.toggle').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentDir = btn.dataset.value;
      // Reload samples for this directory
      select.innerHTML = '<option value="">-- Loading... --</option>';
      try {
        const samples = await fetchSamples(currentDir);
        select.innerHTML = '<option value="">-- Custom input --</option>' +
          samples.map(s => `<option value="${s.filename}">${s.name} (${s.filename})</option>`).join('');
      } catch {
        select.innerHTML = '<option value="">-- Custom input --</option>';
      }
      selectedFile = '';
      preview.innerHTML = '';
    });
  }

  let debounceTimer: number;
  function updatePreview() {
    clearTimeout(debounceTimer);
    debounceTimer = window.setTimeout(() => {
      const xyz = textarea.value.trim();
      if (xyz) {
        renderMoleculePreview(preview, xyz, []);
      } else {
        preview.innerHTML = '';
      }
      onXyzChange(xyz);
    }, 300);
  }

  select.addEventListener('change', () => {
    selectedFile = select.value;
    if (selectedFile) {
      textarea.value = '';
      textarea.placeholder = `Using sample: ${selectedFile}`;
      // Fetch XYZ content for 3D preview
      fetchSampleContent(selectedFile, currentDir).then(content => {
        if (content && select.value === selectedFile) {
          renderMoleculePreview(preview, content, []);
        }
      }).catch(() => {
        preview.innerHTML = '<p style="color:var(--color-text-dim)">Sample file selected</p>';
      });
    } else {
      textarea.placeholder = '2\n\nH  0.0 0.0 0.0\nH  0.0 0.0 0.74';
      preview.innerHTML = '';
    }
    onXyzChange(textarea.value);
  });

  textarea.addEventListener('input', () => {
    if (textarea.value.trim()) {
      select.value = '';
      selectedFile = '';
    }
    updatePreview();
  });

  // Drag & drop
  textarea.addEventListener('dragover', (e) => { e.preventDefault(); textarea.classList.add('drag-over'); });
  textarea.addEventListener('dragleave', () => textarea.classList.remove('drag-over'));
  textarea.addEventListener('drop', (e) => {
    e.preventDefault();
    textarea.classList.remove('drag-over');
    const file = e.dataTransfer?.files[0];
    if (file && file.name.endsWith('.xyz')) {
      const reader = new FileReader();
      reader.onload = () => {
        textarea.value = reader.result as string;
        select.value = '';
        selectedFile = '';
        updatePreview();
      };
      reader.readAsText(file);
    }
  });

  return {
    getXyz: () => textarea.value.trim(),
    getXyzFile: () => selectedFile,
    getXyzDir: () => currentDir,
    setXyz: (xyz: string) => {
      textarea.value = xyz;
      selectedFile = '';
      onXyzChange(xyz);
      // Update 3D preview
      const previewEl = container.querySelector<HTMLElement>('#mol-preview');
      if (previewEl) renderMoleculePreview(previewEl, xyz, []);
    },
  };
}
