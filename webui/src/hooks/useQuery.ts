import { useCallback, useContext } from 'react'
import type { Form } from '@formily/core'
import isObject from 'lodash/isObject'
import findKey from 'lodash/findKey'
import transform from 'lodash/transform'
import { JSONSchemaContext } from '../components/JSONSchema'
import type { DataType, TObject } from '../types'

export function useSchema() {
  return useContext(JSONSchemaContext)
}

/**
 * Checks if a value is a File or an array of File objects.
 * @param value The value to check.
 * @returns True if the value is a File or an array of File objects, false otherwise.
 */
function isFileOrFileArray(value: any): boolean {
  return value instanceof File
    || (Array.isArray(value) && value.every(item => item instanceof File))
}

/**
 * Checks if the given schema contains a file type.
 * @param value - The schema to check, represented as an object.
 * @returns Returns true if a file type is found in the schema, false otherwise.
 */
export function hasFileInSchema(value: Record<string, DataType>): boolean {
  return !!findKey(value, isFileField)
}

/**
 * Checks if the given schema contains a file type.
 * @param value - The schema to check, represented as an array.
 * @returns Returns true if a file type is found in the schema, false otherwise.
 */
export function isFileField(value: DataType): boolean {
  switch (value.type) {
    case 'object':
      return value.properties ? hasFileInSchema(value.properties) : false
    case 'array':
      return value.items.type === 'file'
    default:
      return value.type === 'file'
  }
}

/**
 * Transforms the source value according to the provided schema.
 * @param {any} srcValue - The value to be transformed.
 * @param {DataType} [schema] - The schema defining the transformation rules.
 * @returns {any} - The transformed data.
 */
export function transformData(srcValue: any, schema?: DataType) {
  if (!schema)
    return srcValue
  switch (schema?.type) {
    case 'object':
      if (schema.properties) {
        return transform<DataType, { [key: string]: any }>(schema.properties, (res, value, key) => {
          res[key] = transformData(srcValue[key], value)
          return res
        }, {})
      }
      else {
        try {
          return JSON.parse(srcValue)
        }
        catch {
          // return empty object when JSON string parse failed
          return {}
        }
      }
    case 'array':
      return srcValue?.map((item: any) => transformData(item, schema.items))
    default:
      return srcValue
  }
}

/**
 * Splits the source object into two separate objects based on field types: non-file fields and file fields.
 * @param srcValue The source object containing the fields to be split. This object can be nested.
 * @param parentPath The path used to construct the key for nested objects.
 * @returns An object containing separated nonFileFields and fileFields.
 */
export function splitFileAndNonFileFields(srcValue: object, parentPath = '') {
  const nonFileFields: { [key: string]: unknown } = {}
  const fileFields: { [key: string]: File | File[] } = {}

  Object.entries(srcValue).forEach(([key, value]) => {
    // constructs a nested path string based on parent path and current key.
    const path = parentPath ? `${parentPath}.${key}` : key

    if (isFileOrFileArray(value)) {
      fileFields[path] = value
    }
    else if (isObject(value)) {
      const res = splitFileAndNonFileFields(value, path)
      nonFileFields[key] = res.nonFileFields
      Object.assign(fileFields, res.fileFields)
    }
    else {
      nonFileFields[key] = value
    }
  })

  return { nonFileFields, fileFields }
}

export function useFormSubmit(form: Form, input?: TObject) {
  return useCallback(async (url: string) => {
    const submittedFormData = await form.submit<object>()
    const transformedData = transformData(submittedFormData, input)
    const hasFiles = hasFileInSchema(input ? { input } : {})

    if (hasFiles) {
      const { nonFileFields, fileFields } = splitFileAndNonFileFields(transformedData)
      const formData = new FormData()

      formData.append('__data', JSON.stringify(nonFileFields))
      Object.entries(fileFields).forEach(([key, value]) => {
        if (Array.isArray(value))
          value.forEach(item => formData.append(key, item))
        else
          formData.append(key, value)
      })

      return fetch(url, {
        method: 'POST',
        body: formData,
      })
    }
    else {
      return fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(submittedFormData),
      })
    }
  }, [form])
}
