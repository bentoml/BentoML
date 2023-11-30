import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import { JSONSchemaProvider } from './components/JSONSchema'
import type { IAPISchema } from './types'

export function mount(schema: IAPISchema, element: HTMLElement) {
  ReactDOM.createRoot(element).render(
    <React.StrictMode>
      <JSONSchemaProvider value={schema}>
        <App />
      </JSONSchemaProvider>
    </React.StrictMode>,
  )
}
