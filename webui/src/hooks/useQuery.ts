import { useCallback, useContext } from 'react'
import type { Form } from '@formily/core'
import isObject from 'lodash/isObject'
import findKey from 'lodash/findKey'
import transform from 'lodash/transform'
import mime from 'mime'
import { JSONSchemaContext } from '../components/JSONSchema'
import type { DataType, IRoute } from '../types'
import { useMountOptions } from './useMountOptions'
import useToken from './useToken'

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
    // @ts-expect-error (no-fallthrough)
    case 'object':
      if (schema.properties) {
        return transform<DataType, { [key: string]: any }>(schema.properties, (res, value, key) => {
          res[key] = transformData(srcValue[key], value)
          return res
        }, {})
      }
    // eslint-disable-next-line no-fallthrough
    case 'tensor':
    case 'dataframe':
      try {
        return JSON.parse(srcValue)
      }
      catch {
        // return empty object when JSON string parse failed
        return {}
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

/**
 * Define a function to extract the filename from the Content-Disposition header
 * @param contentDisposition - The Content-Disposition header string from which to extract the filename.
 * @returns The extracted filename, if found; otherwise, null.
 */
function extractFilename(contentDisposition: string) {
  const regex = /filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/
  const matches = regex.exec(contentDisposition)

  if (matches != null && matches[1]) {
    // If a filename is matched, remove any surrounding quotes and return it
    return matches[1].replace(/['"]/g, '')
  }
  return null
}

export function useFormSubmit(form: Form, route?: IRoute) {
  const { needAuth } = useMountOptions()
  const [token] = useToken()
  const authHeaders: { Authorization?: string } = needAuth ? { Authorization: `Bearer ${token}` } : {}
  return useCallback(async () => {
    if (!route)
      throw new Error('Route config not found')
    const { input, output, route: path } = route
    const request = async () => {
      const submittedFormData = await form.submit<object>()
      const transformedData = transformData(submittedFormData, input)

      if (hasFileInSchema({ input })) {
        const { nonFileFields, fileFields } = splitFileAndNonFileFields(transformedData)
        const formData = new FormData()

        Object.entries(nonFileFields).forEach(([key, value]) => {
          formData.append(key, JSON.stringify(value))
        })
        Object.entries(fileFields).forEach(([key, value]) => {
          if (Array.isArray(value))
            value.forEach(item => formData.append(key, item))
          else
            formData.append(key, value)
        })

        return fetch(path, {
          method: 'POST',
          body: formData,
          headers: authHeaders,
        })
      }
      else {
        return fetch(path, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', ...authHeaders },
          body: JSON.stringify(transformedData),
        })
      }
    }
    const resp = await request()

    if (resp.status >= 400)
      throw new Error(`${resp.status} ${await resp.text()}`)

    if (!hasFileInSchema({ output })) {
      return resp.json()
    }
    else if (output.type === 'file') {
      const contentDisposition = resp.headers.get('Content-Disposition')
      const contentType = resp.headers.get('Content-Type')
      const blob = await resp.blob()
      const filename = contentDisposition
        ? extractFilename(contentDisposition)
        : null
      const extension = contentType ? `.${mime.getExtension(contentType)}` : ''

      return new File([blob], filename || (output.title ?? 'output') + extension)
    }
    else {
      throw new Error('Unsupport output type')
    }
  }, [form])
}
