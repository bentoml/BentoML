import { connect } from '@formily/react'
import { useStyletron } from 'baseui'
import { Button, SHAPE } from 'baseui/button'
import type { FileUploaderProps } from 'baseui/file-uploader'
import { FileUploader as BaseUIFileUploader, StyledContentMessage } from 'baseui/file-uploader'
import { Delete } from 'baseui/icon'

interface Props extends FileUploaderProps {
  value?: File
  onChange?: (file?: File) => void
}

export function FileUploader({ value, onChange, onDrop, ...restProps }: Props) {
  const [css, theme] = useStyletron()

  return (
    <BaseUIFileUploader
      {...restProps}
      overrides={{
        ...restProps.overrides,
        HiddenInput: {
          props: {
            multiple: false,
          },
        },
        ContentMessage: {
          component: ({ ...restProps }) => (
            <StyledContentMessage {...restProps} className={css({ color: value ? '#000' : undefined })}>
              <span>{value?.name ?? 'Drop single file here'}</span>
            </StyledContentMessage>
          ),
        },
        ButtonComponent: {
          component: value
            ? ({ disabled, size, kind, children, ...restProps }) => {
                return (
                  <div className={css({ display: 'flex', marginTop: theme.sizing.scale500 })}>
                    <Button
                      {...restProps}
                      disabled={disabled}
                      size={size}
                      kind={kind}
                      overrides={{}}
                    >
                      {children}
                    </Button>
                    <Button
                      disabled={disabled}
                      size={size}
                      shape={SHAPE.circle}
                      kind={kind}
                      type="button"
                      onClick={() => onChange?.()}
                      className={css({ marginLeft: theme.sizing.scale300 })}
                    >
                      <Delete size={18} />
                    </Button>
                  </div>
                )
              }
            : undefined,
        },
      }}
      onDrop={(acceptedFiles = [], rejectedFiles, event) => {
        const [file] = acceptedFiles

        if (file)
          onChange?.(file)
        onDrop?.(acceptedFiles, rejectedFiles, event)
      }}
    />
  )
}

export default connect(FileUploader)
