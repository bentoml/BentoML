import { useStyletron, LightTheme } from 'baseui'
import type { editor } from 'monaco-editor'
import { connect } from '@formily/react'
import Editor, { loader } from '@monaco-editor/react'
import * as monaco from 'monaco-editor'
import EditorWorker from 'monaco-editor/esm/vs/editor/editor.worker?worker'
import JsonWorker from 'monaco-editor/esm/vs/language/json/json.worker?worker'

interface IJSONInputProps {
  value: string
  onChange?: (value: string) => void
}

window.self.MonacoEnvironment = {
  getWorker(_, label) {
    if (label === 'json')
      return new JsonWorker()

    return new EditorWorker()
  },
}

loader.config({ monaco })

const editorOptions: editor.IStandaloneEditorConstructionOptions = {
  acceptSuggestionOnCommitCharacter: true,
  acceptSuggestionOnEnter: 'on',
  accessibilitySupport: 'auto',
  autoIndent: 'advanced',
  automaticLayout: true,
  codeLens: true,
  colorDecorators: true,
  contextmenu: true,
  cursorBlinking: 'blink',
  cursorSmoothCaretAnimation: 'off',
  cursorStyle: 'line',
  disableLayerHinting: false,
  disableMonospaceOptimizations: false,
  dragAndDrop: false,
  fixedOverflowWidgets: false,
  folding: true,
  foldingStrategy: 'auto',
  fontLigatures: false,
  formatOnPaste: false,
  formatOnType: false,
  hideCursorInOverviewRuler: false,
  links: true,
  mouseWheelZoom: false,
  multiCursorMergeOverlapping: true,
  multiCursorModifier: 'alt',
  overviewRulerBorder: true,
  overviewRulerLanes: 2,
  quickSuggestions: true,
  quickSuggestionsDelay: 100,
  readOnly: false,
  renderControlCharacters: false,
  renderFinalNewline: 'off',
  renderLineHighlight: 'all',
  renderWhitespace: 'none',
  revealHorizontalRightPadding: 30,
  roundedSelection: true,
  rulers: [],
  scrollBeyondLastColumn: 5,
  scrollBeyondLastLine: true,
  selectOnLineNumbers: true,
  selectionClipboard: true,
  selectionHighlight: true,
  showFoldingControls: 'mouseover',
  smoothScrolling: false,
  suggestOnTriggerCharacters: true,
  wordBasedSuggestions: true,
  wordSeparators: '~!@#$%^&*()-=+[{]}|;:\'",.<>/?',
  wordWrap: 'off',
  wordWrapBreakAfterCharacters: '\t})]?|&,;',
  wordWrapBreakBeforeCharacters: '{([+',
  wordWrapColumn: 80,
  wrappingIndent: 'none',
}

export function JSONInput({ value, onChange }: IJSONInputProps) {
  const [, theme] = useStyletron()
  return (
    <Editor
      height="500px"
      theme={theme.name === LightTheme.name ? 'light' : 'vs-dark'}
      defaultLanguage="json"
      value={value}
      onChange={str => str && onChange?.(str)}
      options={editorOptions}
    />
  )
}

export default connect(JSONInput)
