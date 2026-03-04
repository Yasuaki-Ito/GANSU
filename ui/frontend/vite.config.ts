import { defineConfig } from 'vite';
import path from 'path';
import os from 'os';

export default defineConfig({
  server: {
    host: true,
    port: 5173,
  },
  cacheDir: path.join(os.tmpdir(), 'gansu-ui-vite-cache'),
});
