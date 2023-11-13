import AceEditor from 'react-ace'

import 'ace-builds/src-noconflict/mode-json'
import 'ace-builds/src-noconflict/theme-github'
import 'ace-builds/src-noconflict/ext-language_tools'
import { useCallback } from 'react'

interface IJSONInputProps {
  value: unknown
  onChange?: (value: unknown) => void
  placeholder?: string
}

export default function JSONInput({ value, onChange, placeholder }: IJSONInputProps) {
  const jsonString = JSON.stringify(value)

  const handleChange = useCallback((value: string) => {
    try {
      onChange?.(JSON.parse(value))
    }
    catch (e) {
      // noop
    }
  }, [onChange])

  return (
    <AceEditor
      placeholder={placeholder}
      mode="json"
      theme="github"
      onChange={handleChange}
      fontSize={14}
      showPrintMargin
      showGutter
      highlightActiveLine
      width="100%"
      height="300px"
      value={jsonString}
      setOptions={{
        showLineNumbers: true,
        tabSize: 2,
      }}
    />
  )
}
