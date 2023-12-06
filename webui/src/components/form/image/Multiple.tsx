import type { FC } from 'react'
import { useStyletron } from 'baseui'
import type { ArrayField } from '@formily/core'
import { observer, useField, useFieldSchema } from '@formily/react'
import { FileUploader as BaseUIFileUploader } from 'baseui/file-uploader'
import type { FlexGridProps } from 'baseui/flex-grid'
import { FlexGrid, FlexGridItem } from 'baseui/flex-grid'
import ImagePreview, { Remove } from '../../preview/Image'

export const Multiple: FC<FlexGridProps> = observer((props) => {
  const [css, theme] = useStyletron()
  const field = useField<ArrayField>()
  const schema = useFieldSchema()
  if (!schema)
    throw new Error('can not found schema object')
  const dataSource = Array.isArray(field.value) ? field.value : []

  return (
    <FlexGrid
      flexGridColumnCount={[1, 1, 2, 3]}
      flexGridColumnGap={theme.sizing.scale800}
      flexGridRowGap={theme.sizing.scale800}
      {...props}
    >
      {
        dataSource.map((file: File, index) => {
          return (
            <FlexGridItem key={index}>
              <ImagePreview value={file}>
                <Remove onClick={() => field.remove(index)} />
              </ImagePreview>
            </FlexGridItem>
          )
        })
      }
      <FlexGridItem>
        <BaseUIFileUploader
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
          }}
          onDrop={(acceptedFiles = []) => {
            field.push(...acceptedFiles)
          }}
        />
      </FlexGridItem>
    </FlexGrid>
  )
})

export default Multiple
