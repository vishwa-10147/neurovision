/// <reference types="vitest/config" />
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // The app imports example fixtures from ../shared; allow serving only that.
  server: { port: 5173, strictPort: true, fs: { allow: ['.', '../shared'] } },
  // The ~950 kB chunk is three.js + R3F, lazy-loaded only when the 3D view opens.
  build: { chunkSizeWarningLimit: 1000 },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    // Room for the 5 s async-wait limit in src/test/setup.ts.
    testTimeout: 15_000,
  },
})
