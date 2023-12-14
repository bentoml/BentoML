import type { FC, HTMLAttributes, PropsWithChildren } from 'react'
import { observer, useField, useFieldSchema } from '@formily/react'
import type { ArrayField } from '@formily/core'
import { useStyletron } from 'baseui'
import { FileUploader as BaseUIFileUploader } from 'baseui/file-uploader'
import BasePreview, { Remove } from '../../preview/Base'

type IMultipleProps = PropsWithChildren<HTMLAttributes<HTMLDivElement>>

export const Multiple: FC<IMultipleProps> = observer((props) => {
  const [css, theme] = useStyletron()
  const field = useField<ArrayField>()
  const schema = useFieldSchema()
  if (!schema)
    throw new Error('can not found schema object')
  const dataSource = Array.isArray(field.value) ? field.value : []

  return (
    <div {...props}>
      <BaseUIFileUploader
        onDrop={(acceptedFiles = []) => {
          field.push(...acceptedFiles)
        }}
      />
      {
        dataSource.length > 0 && (
          <div className={css({ marginTop: theme.sizing.scale200 })}>
            {
              dataSource.map((file: File, index) => (
                <BasePreview
                  key={index}
                  value={file}
                >
                  <Remove onClick={() => field.remove(index)} />
                </BasePreview>
              ))
            }
          </div>
        )
      }
    </div>
  )
})

export default Multiple
