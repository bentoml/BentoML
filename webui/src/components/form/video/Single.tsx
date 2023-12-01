import type { FileUploaderProps } from 'baseui/file-uploader'
import { FileUploader as BaseUIFileUploader } from 'baseui/file-uploader'
import Player from './Player'

interface ISingleProps extends FileUploaderProps {
  value?: File
  onChange?: (file?: File) => void
}

export function Single({ value, onChange, onDrop, ...restProps }: ISingleProps) {
  return value
    ? (
      <Player
        files={[value]}
        onRemove={() => onChange?.()}
      />
      )
    : (
      <BaseUIFileUploader
        {...restProps}
        accept="video/*"
        overrides={{
          ...restProps.overrides,
          HiddenInput: {
            props: {
              multiple: false,
            },
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
