import { createSchemaField } from '@formily/react'
import type { ISchema } from '@formily/json-schema'
import type { DataType, TObject } from '../../types'
import FormControl from './FormControl'
import Checkbox from './Checkbox'
import Input from './Input'
import InputNumber from './InputNumber'
import JSONInput from './JSONInput'
import File from './File'
import { MultipleImages, SingleImage } from './image'
import { ArrayItem, ArrayItems } from './Array'

function renderExample(examples?: unknown[]) {
  if (!examples || examples.length === 0)
    return undefined
  const text = JSON.stringify(examples[0])
  return text.startsWith('"') ? text.slice(1, -1) : text
}

function getSchema(propertie: DataType, addition: ISchema = {}): ISchema {
  const base: ISchema = {
    'type': propertie.type,
    'title': propertie.title,
    'description': propertie.description,
    'default': propertie.default,
    'x-decorator': 'FormControl',
    ...addition,
  }
  const placeholder = renderExample(propertie.examples)

  switch (propertie.type) {
    case 'array':
      switch (propertie.items.type) {
        // Use special components to render page when the array type is file, image, audio etc.
        case 'file':
          switch (propertie.items.format) {
            case 'image':
              return {
                ...base,
                'x-component': 'MultipleImages',
              }
            default:
              return {
                ...base,
                'x-component': 'ArrayItems',
                'items': {
                  'type': 'void',
                  'x-component': 'ArrayItem',
                  'properties': {
                    input: getSchema(propertie.items),
                  },
                },
              }
          }
        default:
          return {
            ...base,
            'x-component': 'ArrayItems',
            'items': {
              'type': 'void',
              'x-component': 'ArrayItem',
              'properties': {
                input: getSchema(propertie.items),
              },
            },
          }
      }
    case 'integer':
    case 'number': {
      const isInteger = propertie.type === 'integer'

      return {
        ...base,
        'default': base.default ?? 0,
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
        'default': base.default ?? false,
        'x-component': 'Checkbox',
      }
    case 'object':
      if (propertie.properties) {
        return {
          ...base,
          properties: Object.fromEntries(
            Object.entries(propertie.properties).map(([key, value]) => [
              key,
              getSchema(value, { required: propertie.required?.includes(key) }),
            ]),
          ),
        }
      }
      else {
        return {
          ...base,
          // the type must is JSON string, otherwise the editor cannot be mounted
          'type': 'string',
          'default': base.default ?? '{}',
          'x-component': 'JSONInput',
        }
      }
    case 'file':
      switch (propertie.format) {
        case 'image':
          return {
            ...base,
            'type': 'file',
            'default': undefined,
            'x-component': 'SingleImage',
          }
        default:
          return {
            ...base,
            'type': 'file',
            'default': undefined,
            'x-component': 'File',
          }
      }
    case 'tensor':
    case 'dataframe':
      return {
        ...base,
        'type': 'file',
        'default': undefined,
        'x-component': 'File',
      }
    case 'string':
    default:
      return {
        ...base,
        'default': base.default ?? '',
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

  return getSchema(jsonSchema, { 'x-decorator': undefined })
}

export default createSchemaField({
  components: {
    FormControl,
    Checkbox,
    Input,
    InputNumber,
    JSONInput,
    File,
    SingleImage,
    MultipleImages,
    ArrayItems,
    ArrayItem,
  },
})
