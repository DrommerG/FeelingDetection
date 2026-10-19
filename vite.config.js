import { defineConfig } from 'vite';

export default defineConfig({
  base: process.env.GITHUB_ACTIONS ? '/visor-expresiones/' : '/',
  server: { host: '0.0.0.0' },
});