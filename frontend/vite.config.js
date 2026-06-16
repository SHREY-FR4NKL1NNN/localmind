import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Vite dev server runs on 5173; the FastAPI backend (on 8000) whitelists this
// origin via CORS.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
  },
})
