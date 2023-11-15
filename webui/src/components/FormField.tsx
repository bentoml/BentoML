import { useCallback } from 'react'
import { Input } from 'baseui/input'
import { Checkbox } from 'baseui/checkbox'
import type { DataType } from '../types'
import JSONInput from './form/JSONInput'

interface IFieldProps<T> {
  value: T
  schema: DataType
  onChange?: (value: T) => void
}

function renderExample(examples?: unknown[]) {
  if (!examples || examples.length === 0)
    return undefined
  const text = JSON.stringify(examples[0])
  return text.startsWith('"') ? text.slice(1, -1) : text
}

export default function FormField<T = unknown>({ value, schema, onChange }: IFieldProps<T>) {
  const placeholder = renderExample(schema.examples)

  const handleChange = useCallback((e: unknown) => {
    const { target } = e as React.ChangeEvent<HTMLInputElement>
    onChange?.((target.type === 'checkbox' ? target.checked : target.value) as T)
  }, [onChange])

  switch (schema.type) {
    case 'string':
      return <Input value={value as string} placeholder={placeholder} onChange={handleChange} />
    case 'number':
      return <Input value={value as string} placeholder={placeholder} onChange={handleChange} type="number" />
    case 'boolean':
      return <Checkbox checked={value as boolean} onChange={handleChange} />
    case 'object':
    case 'tensor':
    case 'dataframe':
      return <JSONInput value={value as string} onChange={v => onChange?.(v as T)} />
    default:
      return <Input value={value as string} placeholder={placeholder} onChange={handleChange} />
  }
}
