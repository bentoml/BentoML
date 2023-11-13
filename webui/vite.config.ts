import process from 'node:process'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import svgr from 'vite-plugin-svgr'

const proxySetting = process.env.PROXY
  ? {
      server: {
        proxy: {
          '^/(?!node_modules/|src/|assets/|@).+$': {
            target: process.env.PROXY,
          },
        },
      },
    }
  : {}

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react(), svgr()],
  ...proxySetting,
})
