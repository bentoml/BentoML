import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import { JSONSchemaProvider } from './components/JSONSchema'
import type { IAPISchema } from './types'
import type { IMountOptions } from './hooks/useMountOptions'
import { OptionProvider } from './hooks/useMountOptions'

export function mount(schema: IAPISchema, element: HTMLElement, options?: IMountOptions) {
  ReactDOM.createRoot(element).render(
    <React.StrictMode>
      <OptionProvider value={options ?? {}}>
        <JSONSchemaProvider value={schema}>
          <App />
        </JSONSchemaProvider>
      </OptionProvider>
    </React.StrictMode>,
  )
}
