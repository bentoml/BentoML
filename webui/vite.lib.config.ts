import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import svgr from 'vite-plugin-svgr'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react(), svgr()],
  build: {
    outDir: 'lib',
    lib: {
      entry: 'src/index.tsx',
      name: 'BentoMLUI',
      fileName: format => `bentoml-ui.${format}.js`,
      formats: ['umd', 'es'],
    },
  },
})
