import type { FC, HTMLAttributes, PropsWithChildren } from 'react'
import type { ArrayField } from '@formily/core'
import { observer, useField, useFieldSchema } from '@formily/react'
import { useStyletron } from 'baseui'
import { FileUploader as BaseUIFileUploader } from 'baseui/file-uploader'
import AudioPlayer from '../../preview/Audio'
import { Remove } from '../../preview/Base'

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
          <AudioPlayer files={dataSource}>
            {(_, index, activeIndex, setActive) => (
              <Remove
                onClick={() => {
                  if (index === activeIndex && dataSource.length > 1)
                    setActive(dataSource[index === 0 ? 1 : 0])

                  field.remove(index)
                }}
              />
            )}
          </AudioPlayer>
        </div>
      )}
    </div>
  )
})

export default Multiple
