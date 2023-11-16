import { createSchemaField } from '@formily/react'
import type { DataType, TObject } from '../../types'
import FormControl from './FormControl'
import Checkbox from './Checkbox'
import Input from './Input'
import InputNumber from './InputNumber'
import JSONInput from './JSONInput'

function renderExample(examples?: unknown[]) {
  if (!examples || examples.length === 0)
    return undefined
  const text = JSON.stringify(examples[0])
  return text.startsWith('"') ? text.slice(1, -1) : text
}

function getSchema(propertie: DataType) {
  const base = {
    'title': propertie.title,
    'description': propertie.description,
    'default': propertie.default,
    'x-decorator': 'FormControl',
  }
  const placeholder = renderExample(propertie.examples)

  switch (propertie.type) {
    case 'integer':
    case 'number': {
      const isInteger = propertie.type === 'integer'

      return {
        ...base,
        'x-component': 'InputNumber',
        'x-component-props': {
          isInteger,
          step: isInteger ? 1 : 0.01,
          maximum: propertie.maximum,
          minimum: propertie.minimum,
          exclusiveMaximum: propertie.exclusiveMaximum,
          exclusiveMinimum: propertie.exclusiveMinimum,
          placeholder,
        },
      }
    }
    case 'boolean':
      return {
        ...base,
        'x-component': 'Checkbox',
      }
    case 'object':
    case 'tensor':
    case 'dataframe':
      return {
        ...base,
        'x-component': 'JSONInput',
      }
    case 'string':
    default:
      return {
        ...base,
        'x-component': 'Input',
        'x-component-props': {
          placeholder,
        },
      }
  }
}

export function generateFormSchema(jsonSchema?: TObject) {
  if (!jsonSchema || !jsonSchema.properties)
    return { type: 'object' }

  return {
    ...jsonSchema,
    properties: Object.fromEntries(
      Object.entries(jsonSchema.properties).map(([key, value]) => [key, getSchema(value)]),
    ),
  }
}

export default createSchemaField({
  components: {
    FormControl,
    Checkbox,
    Input,
    InputNumber,
    JSONInput,
  },
})
