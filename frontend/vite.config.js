import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/query': 'http://localhost:7860',
      '/load':  'http://localhost:7860',
      '/ingest':'http://localhost:7860',
    }
  }
})
