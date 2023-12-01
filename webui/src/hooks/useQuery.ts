import { useCallback, useContext } from 'react'
import type { Form } from '@formily/core'
import isObject from 'lodash/isObject'
import findKey from 'lodash/findKey'
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
  return !!findKey(value, (o: DataType) => {
    switch (o.type) {
      case 'object':
        return o.properties ? hasFileInSchema(o.properties) : false
      case 'array':
        return o.items.type === 'file'
      default:
        return o.type === 'file'
    }
  })
}

/**
 * Converts an object by separating File objects and other data.
 * @param srcValue The source object to convert.
 * @param parentPath The path used to construct the key for nested objects.
 * @returns An object containing separated nonFileFields and fileFields.
 */
export function convert(srcValue: object, parentPath = '') {
  const nonFileFields: { [key: string]: unknown } = {}
  const fileFields: { [key: string]: File | File[] } = {}

  Object.entries(srcValue).forEach(([key, value]) => {
    // constructs a nested path string based on parent path and current key.
    const path = parentPath ? `${parentPath}.${key}` : key

    if (isFileOrFileArray(value)) {
      fileFields[path] = value
    }
    else if (isObject(value)) {
      const res = convert(value, path)
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
    const hasFiles = hasFileInSchema(input ? { input } : {})

    if (hasFiles) {
      const { nonFileFields, fileFields } = convert(submittedFormData)
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
        headers: { 'Content-Type': 'multipart/form-data' },
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
