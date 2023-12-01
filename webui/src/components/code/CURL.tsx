import isEmpty from 'lodash/isEmpty'
import { convert, hasFileInSchema } from '../../hooks/useQuery'
import type { TObject } from '../../types'
import Code from './Base'

interface ICURLCodeProps {
  values: object
  schema?: TObject
  path?: string
}

/**
 * Formats a JSON object into a string with custom indentation.
 * @param json - The JSON object to be formatted.
 * @param indent - The number of spaces used for indentation. Defaults to 4.
 * @returns A string representation of the JSON object with custom indentation.
 */
function formatJSON(json: object, indent = 4) {
  return JSON.stringify(json, null, indent)
    .split('\n')
    .map((line, index) => (index === 0 ? '' : ' '.repeat(indent)) + line)
    .join('\n')
}

/**
 * Formats a file or an array of files into a string representation.
 * @param files - An object where each value is either a File object or an array of File objects.
 * @param indent - The number of spaces used for indentation in the output string.
 * @returns A formatted string representing the input files, suitable for CLI commands or logging.
 */
function formatFiles(files: { [field: string]: File | File[] }, indent = 4) {
  if (isEmpty(files))
    return ''
  const context = Object.entries(files)
    // if the entry is an array of files, format each file in the array.
    // otherwise, format the single file.
    .flatMap(([key, value]) => Array.isArray(value)
      ? value.map(item => `-F '${key}=@${item.name}'`)
      : [`-F '${key}=@${value.name}'`],
    )
    .map(line => `${' '.repeat(indent)}${line}`)
    .join('\n')

  return `\n${context}`
}

function generateCode(data: object, path = '/', schema?: TObject) {
  const hasFiles = hasFileInSchema(schema ? { schema } : {})
  if (hasFiles) {
    const { nonFileFields, fileFields } = convert(data)

    return `$ curl -s -X POST \\
    -H "Content-Type: multipart/form-data" \\
    -F '__data=${formatJSON(nonFileFields, 4)}' \\ ${formatFiles(fileFields, 4)}
    http://localhost:3000${path}`
  }
  else {
    return `$ curl -s -X POST \\
    -H "Content-Type: application/json" \\
    -d '${formatJSON(data, 4)} \\
    http://localhost:3000${path}`
  }
}

function CURL({ path, values, schema }: ICURLCodeProps) {
  return (
    // values is proxy object, so it must rendered every time
    // otherwise it will fail to update
    <Code>{generateCode(values, path, schema)}</Code>
  )
}

export default CURL
