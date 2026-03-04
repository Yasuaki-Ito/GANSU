/** Progress panel — real-time stdout stream and SCF iteration display */

export function createProgressPanel(container: HTMLElement): {
  show: () => void;
  hide: () => void;
  addLine: (text: string) => void;
  clear: () => void;
  setStatus: (status: string) => void;
} {
  container.innerHTML = `
    <div class="panel progress-panel hidden" id="progress-panel">
      <h2>Running...</h2>
      <div class="progress-status" id="progress-status"></div>
      <div class="progress-output" id="progress-output"></div>
    </div>
  `;

  const panel = container.querySelector<HTMLElement>('#progress-panel')!;
  const status = container.querySelector<HTMLElement>('#progress-status')!;
  const output = container.querySelector<HTMLElement>('#progress-output')!;

  return {
    show: () => { panel.classList.remove('hidden'); output.innerHTML = ''; status.textContent = ''; },
    hide: () => panel.classList.add('hidden'),
    addLine: (text: string) => {
      const line = document.createElement('div');
      line.className = 'output-line';
      line.textContent = text;
      output.appendChild(line);
      output.scrollTop = output.scrollHeight;
    },
    clear: () => { output.innerHTML = ''; },
    setStatus: (s: string) => { status.textContent = s; },
  };
}
