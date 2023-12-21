const path = require('node:path')
const fs = require('fs-extra')

const targetPath = path.resolve(__dirname, '../../src/_bentoml_impl/server/assets')
const outDir = path.resolve(__dirname, '../lib/assets')

async function main() {
  await fs.rm(targetPath, { recursive: true, force: true })
  await fs.copy(outDir, targetPath)
  await fs.copy(path.resolve(outDir, '../style.css'), path.resolve(targetPath, 'style.css'))
  await fs.copy(path.resolve(outDir, '../bentoml-ui.umd.js'), path.resolve(targetPath, 'bentoml-ui.umd.js'))
}

main()
