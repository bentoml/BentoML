import type { FC } from 'react'
import { useStyletron } from 'baseui'
import type { ArrayField } from '@formily/core'
import { observer, useField, useFieldSchema } from '@formily/react'
import { KIND as BUTTON_KIND, SHAPE as BUTTON_SHAPE, SIZE as BUTTON_SIZE, Button } from 'baseui/button'
import { FileUploader as BaseUIFileUploader } from 'baseui/file-uploader'
import type { FlexGridProps } from 'baseui/flex-grid'
import { FlexGrid, FlexGridItem } from 'baseui/flex-grid'
import { Delete } from 'baseui/icon'
import Preview from './Preview'

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
            <FlexGridItem
              key={index}
              className={css({
                backgroundImage: `url(${URL.createObjectURL(file)})`,
                backgroundSize: 'cover',
                backgroundPosition: 'center',
                height: '200px',
                borderRadius: theme.borders.radius400,
                borderColor: theme.colors.fileUploaderBorderColorDefault,
                borderStyle: 'solid',
                borderWidth: theme.sizing.scale0,
                boxSizing: 'border-box',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              })}
            >
              <Preview
                value={file}
                size={BUTTON_SIZE.compact}
                kind={BUTTON_KIND.secondary}
                shape={BUTTON_SHAPE.pill}
              />
              <Button
                size={BUTTON_SIZE.compact}
                shape={BUTTON_SHAPE.circle}
                kind={BUTTON_KIND.secondary}
                type="button"
                onClick={() => {
                  field.remove(index)
                }}
                className={css({ marginLeft: theme.sizing.scale300 })}
              >
                <Delete size={18} />
              </Button>
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
