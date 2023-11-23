import type { FC, HTMLAttributes, PropsWithChildren } from 'react'
import type { ArrayField } from '@formily/core'
import { observer, useField, useFieldSchema } from '@formily/react'
import { useStyletron } from 'baseui'
import { FileUploader as BaseUIFileUploader } from 'baseui/file-uploader'
import Player from './Player'

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
        accept="audio/*"
        onDrop={(acceptedFiles = []) => {
          field.push(...acceptedFiles)
        }}
      />
      {dataSource.length > 0 && (
        <div className={css({ marginTop: theme.sizing.scale200 })}>
          <Player
            files={dataSource}
            onRemove={index => field.remove(index)}
          />
        </div>
      )}
    </div>
  )
})

export default Multiple
