import Highlight, { defaultProps } from 'prism-react-renderer'
import type { Language } from 'prism-react-renderer'
import { useStyletron } from 'baseui'
import { Button, KIND, SHAPE, SIZE } from 'baseui/button'
import { SnackbarProvider, useSnackbar } from 'baseui/snackbar'
import { IconCopy } from '@tabler/icons-react'
import Check from 'baseui/icon/check'
import { CopyToClipboard } from 'react-copy-to-clipboard'
import { useIsLight } from '../../hooks/useTheme'
import type { TObject } from '../../types'

interface ICodeProps {
  children: string
  language?: Language
}

// these themes come from uber
// https://github.com/uber/react-view/blob/master/src/light-theme.ts
const lightTheme = {
  plain: {
    fontSize: '14px',
    color: '#333',
    backgroundColor: 'rgb(253, 253, 253)',
    fontFamily: `Consolas, Monaco, 'Andale Mono', 'Ubuntu Mono', monospace`,
    margin: 0,
  },
  styles: [
    {
      types: ['comment', 'punctuation'],
      style: {
        color: 'rgb(170, 170, 170)',
      },
    },
    {
      types: ['operator'],
      style: {
        color: 'rgb(119, 119, 119)',
      },
    },
    {
      types: ['builtin', 'variable', 'constant', 'number', 'char', 'symbol'],
      style: {
        color: 'rgb(156, 93, 39)',
      },
    },
    {
      types: ['function'],
      style: {
        color: 'rgb(170, 55, 49)',
      },
    },
    {
      types: ['string'],
      style: {
        color: 'rgb(68, 140, 39)',
      },
    },
    {
      types: ['tag'],
      style: {
        color: 'rgb(75, 105, 198)',
      },
    },
    {
      types: ['attr-name'],
      style: {
        color: 'rgb(129, 144, 160)',
      },
    },
    {
      types: ['selector'],
      style: {
        color: 'rgb(122, 62, 157)',
      },
    },
    {
      types: ['keyword'],
      style: {},
    },
    {
      types: ['changed'],
      style: {
        color: 'rgb(0, 0, 0)',
        backgroundColor: 'rgb(255, 255, 221)',
      },
    },
    {
      types: ['deleted'],
      style: {
        color: 'rgb(0, 0, 0)',
        backgroundColor: 'rgb(255, 221, 221)',
      },
    },
    {
      types: ['inserted'],
      style: {
        color: 'rgb(0, 0, 0)',
        backgroundColor: 'rgb(221, 255, 221)',
      },
    },
  ],
}

// https://github.com/uber/baseweb/blob/master/documentation-site/components/yard/dark-theme.ts
const darkTheme = {
  plain: {
    color: '#d4d4d4',
    backgroundColor: '#292929',
    fontSize: '14px',
    fontFamily: `Consolas, Monaco, 'Andale Mono', 'Ubuntu Mono', monospace`,
    margin: 0,
  },
  styles: [
    {
      types: ['prolog'],
      style: {
        color: 'rgb(0, 0, 128)',
      },
    },
    {
      types: ['comment'],
      style: {
        color: 'rgb(106, 153, 85)',
      },
    },
    {
      types: ['builtin', 'tag', 'changed', 'punctuation', 'keyword'],
      style: {
        color: 'rgb(86, 156, 214)',
      },
    },
    {
      types: ['number', 'inserted'],
      style: {
        color: 'rgb(181, 206, 168)',
      },
    },
    {
      types: ['constant'],
      style: {
        color: 'rgb(100, 102, 149)',
      },
    },
    {
      types: ['attr-name', 'variable'],
      style: {
        color: 'rgb(156, 220, 254)',
      },
    },
    {
      types: ['deleted', 'string'],
      style: {
        color: 'rgb(206, 145, 120)',
      },
    },
    {
      types: ['operator'],
      style: {
        color: 'rgb(212, 212, 212)',
      },
    },
    {
      types: ['function'],
      style: {
        color: 'rgb(220, 220, 170)',
      },
    },
    {
      types: ['char'],
      style: {
        color: 'rgb(209, 105, 105)',
      },
    },
  ],
}

function Code({ children, language }: ICodeProps) {
  const [css, theme] = useStyletron()
  const isLight = useIsLight()
  const { enqueue } = useSnackbar()

  return (
    <div
      className={css({
        position: 'relative',
        borderRadius: '8px',
        borderLeftColor: isLight ? theme.colors.warning200 : theme.colors.mono500,
        backgroundColor: isLight ? 'rgb(253, 253, 253)' : '#292929',
        marginBottom: theme.sizing.scale400,
        marginTop: theme.sizing.scale400,
      })}
    >
      <CopyToClipboard
        text={children}
        onCopy={() => enqueue({
          message: 'Copied to clipboard',
          startEnhancer: ({ size }) => <Check size={size} />,
        }, 1000)}
      >
        <Button
          type="button"
          kind={KIND.tertiary}
          size={SIZE.compact}
          shape={SHAPE.circle}
          className={css({
            position: 'absolute',
            top: '10px',
            right: '10px',
          })}
        >
          <IconCopy size={18} />
        </Button>
      </CopyToClipboard>

      <Highlight
        {...defaultProps}
        code={children.replace(/[\r\n]+$/, '')}
        language={language as Language} // language is empty when we don't want to use highlight
        theme={theme.name.startsWith('light-theme') ? lightTheme : darkTheme}
      >
        {({ style, tokens, getLineProps, getTokenProps }) => (
          <div className={css({
            overflow: 'auto',
            padding: theme.sizing.scale400,
          })}
          >
            <pre
              dir="ltr"
              className={css({
                ...style,
                display: 'inline-block',
                padding: '12px',
              })}
            >
              {tokens.map((line, i) => (
                <div key={i} {...getLineProps({ line, key: i })}>
                  {line.map((token, key) => (
                    <span key={key} {...getTokenProps({ token, key })} />
                  ))}
                </div>
              ))}
            </pre>
          </div>
        )}
      </Highlight>
    </div>

  )
}

function CopyCode(props: ICodeProps) {
  return (
    <SnackbarProvider>
      <Code {...props} />
    </SnackbarProvider>
  )
}

/**
 * Formats a JSON object into a string with custom indentation.
 * @param json - The JSON object to be formatted.
 * @param indent - The number of spaces used for indentation. Defaults to 4.
 * @returns A string representation of the JSON object with custom indentation.
 */
export function formatJSON(json: object, indent = 4) {
  return JSON.stringify(json, null, indent)
    .split('\n')
    .join(`\n${' '.repeat(indent)}`)
}

export interface IClientProps {
  values: object
  schema?: TObject
  path?: string
}

export default CopyCode
