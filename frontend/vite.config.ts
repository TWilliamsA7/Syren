import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: { proxy: { '/api': 'http://localhost:8000' } },
  optimizeDeps: {
    exclude: ['maplibre-gl']
  },
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:8000'
    }
  }
})
