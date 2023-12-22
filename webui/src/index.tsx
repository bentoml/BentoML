import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import { JSONSchemaProvider } from './components/JSONSchema'
import type { IAPISchema } from './types'
import type { IMountOptions } from './hooks/useMountOptions'
import { OptionProvider, defaultOptions } from './hooks/useMountOptions'

export function BentoMLUI(props: { schema: IAPISchema, options?: Partial<IMountOptions> }) {
  const mergedOptions = { ...defaultOptions, ...(props.options ?? {}) }
  return (
    <OptionProvider value={mergedOptions}>
      <JSONSchemaProvider value={props.schema}>
        <App />
      </JSONSchemaProvider>
    </OptionProvider>
  )
}

export function mount(schema: IAPISchema, element: HTMLElement, options?: Partial<IMountOptions>) {
  ReactDOM.createRoot(element).render(
    <React.StrictMode>
      <BentoMLUI schema={schema} options={options} />
    </React.StrictMode>,
  )
}
