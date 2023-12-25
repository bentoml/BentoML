import type { FileUploaderProps } from 'baseui/file-uploader'
import { FileUploader as BaseUIFileUploader } from 'baseui/file-uploader'
import VideoPlayer from '../../preview/Video'
import { Remove } from '../../preview/Base'

interface ISingleProps extends FileUploaderProps {
  value?: File
  onChange?: (file?: File) => void
}

export function Single({ value, onChange, onDrop, ...restProps }: ISingleProps) {
  return value
    ? (
      <VideoPlayer files={[value]}>
        {() => <Remove onClick={() => onChange?.()} />}
      </VideoPlayer>
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
