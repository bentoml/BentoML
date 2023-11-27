import { useStyletron } from 'baseui'
import { KIND as BUTTON_KIND, SHAPE as BUTTON_SHAPE, SIZE as BUTTON_SIZE, Button } from 'baseui/button'
import type { FileUploaderProps } from 'baseui/file-uploader'
import { FileUploader as BaseUIFileUploader } from 'baseui/file-uploader'
import { Delete } from 'baseui/icon'
import Preview from './Preview'

interface ISingleImageProps extends FileUploaderProps {
  value?: File
  onChange?: (file?: File) => void
}

function Single({ value, onChange, onDrop, ...restProps }: ISingleImageProps) {
  const [css, theme] = useStyletron()

  return (
    <BaseUIFileUploader
      {...restProps}
      accept="image/*"
      overrides={{
        FileDragAndDrop: {
          props: {
            className: css({
              backgroundImage: value ? `url(${URL.createObjectURL(value)})` : undefined,
              backgroundSize: 'cover',
              backgroundPosition: 'center',
              height: '200px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }),
          },
        },
        HiddenInput: {
          props: {
            multiple: false,
          },
        },
        ContentMessage: {
          component: value ? () => null : undefined,
        },
        ButtonComponent: {
          component: value
            ? ({ disabled, size, kind }) => (
              <div className={css({ display: 'flex', marginTop: 0 })}>
                <Preview
                  value={value}
                  size={BUTTON_SIZE.compact}
                  kind={BUTTON_KIND.secondary}
                  shape={BUTTON_SHAPE.pill}
                />
                <Button
                  disabled={disabled}
                  size={size}
                  shape={BUTTON_SHAPE.circle}
                  kind={kind}
                  type="button"
                  onClick={() => onChange?.()}
                  className={css({ marginLeft: theme.sizing.scale300 })}
                >
                  <Delete size={18} />
                </Button>
              </div>
              )
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

export default Single
