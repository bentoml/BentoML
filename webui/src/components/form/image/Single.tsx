import { useStyletron } from 'baseui'
import type { FileUploaderProps } from 'baseui/file-uploader'
import { FileUploader as BaseUIFileUploader } from 'baseui/file-uploader'
import ImagePreview, { Remove } from '../../preview/Image'

interface ISingleImageProps extends FileUploaderProps {
  value?: File
  onChange?: (file?: File) => void
}

function Single({ value, onChange, onDrop, ...restProps }: ISingleImageProps) {
  const [css] = useStyletron()

  return value
    ? (
      <ImagePreview value={value}>
        <Remove onClick={() => onChange?.()} />
      </ImagePreview>
      )
    : (
      <BaseUIFileUploader
        {...restProps}
        accept="image/*"
        overrides={{
          FileDragAndDrop: {
            props: {
              className: css({
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
