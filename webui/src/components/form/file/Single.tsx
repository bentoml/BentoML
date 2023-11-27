import type { FileUploaderProps } from 'baseui/file-uploader'
import { FileUploader as BaseUIFileUploader } from 'baseui/file-uploader'
import { IconFileInvoice } from '@tabler/icons-react'
import ListItem from './ListItem'

interface ISingleProps extends FileUploaderProps {
  value?: File
  onChange?: (file?: File) => void
}

export function SingleFile({ value, onChange, onDrop, ...restProps }: ISingleProps) {
  return value
    ? (
      <ListItem
        before={<IconFileInvoice size={18} />}
        value={value}
        onRemove={() => onChange?.()}
      />
      )
    : (
      <BaseUIFileUploader
        {...restProps}
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

export default SingleFile
