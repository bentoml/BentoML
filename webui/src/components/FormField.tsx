import { useCallback } from 'react'
import type { DataType } from '../types'
import Input from './form/Input'
import InputNumber from './form/InputNumber'
import Checkbox from './form/Checkbox'
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
  const handleChange = useCallback((val: unknown) => onChange?.(val as T), [onChange])

  switch (schema.type) {
    case 'integer':
    case 'number':
      return (
        <InputNumber
          step={schema.type === 'integer' ? 1 : 0.01}
          value={value as unknown as number}
          maximum={schema.maximum}
          minimum={schema.minimum}
          exclusiveMaximum={schema.exclusiveMaximum}
          exclusiveMinimum={schema.exclusiveMinimum}
          placeholder={placeholder}
          onChange={handleChange}
        />
      )
    case 'boolean':
      return <Checkbox checked={value as unknown as boolean} onChange={handleChange} />
    case 'object':
    case 'tensor':
    case 'dataframe':
      return <JSONInput value={value as unknown as string} onChange={handleChange} />
    case 'string':
    default:
      return <Input value={value as unknown as string} placeholder={placeholder} onChange={handleChange} />
  }
}
