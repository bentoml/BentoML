import isEmpty from 'lodash/isEmpty'
import { hasFileInSchema, splitFileAndNonFileFields } from '../../hooks/useQuery'
import type { TObject } from '../../types'
import type { IClientProps } from './Base'
import Code, { formatJSON } from './Base'

/**
 * Formats a file or an array of files into a string representation.
 * @param files - An object where each value is either a File object or an array of File objects.
 * @param indent - The number of spaces used for indentation in the output string.
 * @returns A formatted string representing the input files, suitable for CLI commands or logging.
 */
function formatFiles(files: { [field: string]: File | File[] }, indent = 4) {
  if (isEmpty(files))
    return ''
  const content = Object.entries(files)
    // if the entry is an array of files, format each file in the array.
    // otherwise, format the single file.
    .flatMap(([key, value]) => Array.isArray(value)
      ? value.map(item => `-F '${key}=@${item.name}'`)
      : [`-F '${key}=@${value.name}'`],
    )
    .join(` \\\n${' '.repeat(indent)}`)

  return content
}

function formatDataField(data: Record<string, unknown>, indent = 4) {
  if (isEmpty(data))
    return ''
  return Object.entries(data)
    .map(([key, value]) => `-F ${key}=${JSON.stringify(value)}`)
    .join(` \\\n${' '.repeat(indent)}`)
}

function generateCode(data: object, path = '/', schema?: TObject) {
  const hasFiles = hasFileInSchema(schema ? { schema } : {})
  if (hasFiles) {
    const { nonFileFields, fileFields } = splitFileAndNonFileFields(data)

    return `$ curl -s -X POST \\
    ${formatDataField(nonFileFields, 4)} \\
    ${formatFiles(fileFields, 4)} \\
    http://localhost:3000${path}`
  }
  else {
    return `$ curl -s -X POST \\
    -H "Content-Type: application/json" \\
    -d '${formatJSON(data, 4)} \\
    http://localhost:3000${path}`
  }
}

function CURL({ path, values, schema }: IClientProps) {
  return (
    // values is proxy object, so it must rendered every time
    // otherwise it will fail to update
    <Code language="bash">{generateCode(values, path, schema)}</Code>
  )
}

export default CURL
