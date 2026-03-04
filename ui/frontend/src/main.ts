import { initTheme } from './ui/theme';
import { initApp } from './app';
import './ui/styles.css';

initTheme();

const root = document.querySelector<HTMLDivElement>('#app');
if (root) {
  initApp(root);
}
