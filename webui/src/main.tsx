import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.tsx'
import { JSONSchemaProvider } from './components/JSONSchema.tsx'
import type { IAPISchema } from './types'
import './index.css'

function mount(schema: IAPISchema) {
  ReactDOM.createRoot(document.getElementById('root')!).render(
    <React.StrictMode>
      <JSONSchemaProvider value={schema}>
        <App />
      </JSONSchemaProvider>
    </React.StrictMode>,
  )
}

fetch('/schema.json')
  .then(res => res.json())
  .then(mount)
  .catch((err) => {
    ReactDOM.createRoot(document.getElementById('root')!).render(
      <React.StrictMode>
        <div>
          Error:
          {err.message}
        </div>
      </React.StrictMode>,
    )
  })
