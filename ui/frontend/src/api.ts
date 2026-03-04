/** API client for GANSU-UI backend */

import type { CalculationParams, CalculationResult, SampleMolecule, StreamEvent } from './types';

const API_BASE = import.meta.env.VITE_API_URL || '';

export async function fetchSampleDirs(): Promise<string[]> {
  const res = await fetch(`${API_BASE}/api/sample_dirs`);
  return res.json();
}

export async function fetchSamples(dir: string = '.'): Promise<SampleMolecule[]> {
  const res = await fetch(`${API_BASE}/api/samples?dir=${encodeURIComponent(dir)}`);
  return res.json();
}

export async function fetchSampleContent(filename: string, dir: string = '.'): Promise<string> {
  const res = await fetch(`${API_BASE}/api/samples/${encodeURIComponent(filename)}?dir=${encodeURIComponent(dir)}`);
  const data = await res.json();
  return data.content || '';
}

export async function fetchBasisSets(): Promise<string[]> {
  const res = await fetch(`${API_BASE}/api/basis`);
  return res.json();
}

export async function fetchAuxiliaryBasisSets(): Promise<{ name: string; dir: string }[]> {
  const res = await fetch(`${API_BASE}/api/auxiliary_basis`);
  return res.json();
}

export async function runCalculation(params: CalculationParams): Promise<CalculationResult> {
  const res = await fetch(`${API_BASE}/api/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  return res.json();
}

export function streamCalculation(
  params: CalculationParams,
  onEvent: (event: StreamEvent) => void,
): AbortController {
  const controller = new AbortController();

  fetch(`${API_BASE}/api/run/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
    signal: controller.signal,
  })
    .then(async (res) => {
      const reader = res.body?.getReader();
      if (!reader) return;
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // Parse SSE events
        const parts = buffer.split('\n\n');
        buffer = parts.pop() || '';
        for (const part of parts) {
          const line = part.trim();
          if (line.startsWith('data: ')) {
            try {
              const event = JSON.parse(line.slice(6)) as StreamEvent;
              onEvent(event);
            } catch { /* skip malformed */ }
          }
        }
      }
    })
    .catch((err) => {
      if (err.name !== 'AbortError') {
        console.error('Stream error:', err);
      }
    });

  return controller;
}
