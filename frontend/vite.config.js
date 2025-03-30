import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  base: '/vite-react-frontend/',
  plugins: [react(), tailwindcss(),],
  build: {
    outDir: 'dist', 
    emptyOutDir: true,
    assetsDir: 'assets',
    manifest: true,
    rollupOptions: {
      input: {
        main: './src/main.jsx'
      }
    }
  },
  server: {
    port: 3000
  }
})