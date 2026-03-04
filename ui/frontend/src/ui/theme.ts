/** Theme switching utility — light (default) / dark */

export interface ThemeColors {
  occupied: string;
  virtual: string;
  alpha: string;
  beta: string;
  gap: string;
  axis: string;
  leader: string;
  label: string;
  hint: string;
  grid: string;
  accent: string;
  error: string;
  dim: string;
  titleSvg: string;
  surface: string;
}

export function initTheme(): void {
  const saved = localStorage.getItem('gansu-theme');
  if (saved === 'dark') {
    document.documentElement.setAttribute('data-theme', 'dark');
  }
}

export function toggleTheme(): string {
  const next = isDark() ? 'light' : 'dark';
  if (next === 'dark') {
    document.documentElement.setAttribute('data-theme', 'dark');
  } else {
    document.documentElement.removeAttribute('data-theme');
  }
  localStorage.setItem('gansu-theme', next);
  return next;
}

export function isDark(): boolean {
  return document.documentElement.getAttribute('data-theme') === 'dark';
}

export function getThemeColors(): ThemeColors {
  const s = getComputedStyle(document.documentElement);
  const v = (name: string) => s.getPropertyValue(name).trim();
  return {
    occupied: v('--color-occupied'),
    virtual: v('--color-virtual'),
    alpha: v('--color-alpha'),
    beta: v('--color-beta'),
    gap: v('--color-gap'),
    axis: v('--color-axis'),
    leader: v('--color-leader'),
    label: v('--color-label'),
    hint: v('--color-hint'),
    grid: v('--color-grid'),
    accent: v('--color-accent'),
    error: v('--color-error'),
    dim: v('--color-text-dim'),
    titleSvg: v('--color-title-svg'),
    surface: v('--color-surface'),
  };
}
