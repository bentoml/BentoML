import type { FC, HTMLAttributes, PropsWithChildren } from 'react'
import { createContext, useContext } from 'react'
import type { Schema } from '@formily/react'
import { RecursionField, observer, useField, useFieldSchema } from '@formily/react'
import type { ArrayField } from '@formily/core'
import { useStyletron } from 'baseui'
import { Button } from 'baseui/button'
import { Delete } from 'baseui/icon'

interface IArrayContext {
  field: ArrayField
  schema: Schema
}

interface IItemContext {
  index: number
}

type IArrayItemsProps = PropsWithChildren<HTMLAttributes<HTMLDivElement>>

const ArrayContext = createContext<IArrayContext | null>(null)

const ItemContext = createContext<IItemContext | null>(null)

const useArray = () => useContext(ArrayContext)

function useIndex() {
  const ctx = useContext(ItemContext)
  return ctx?.index
}

export const ArrayItems: FC<IArrayItemsProps> = observer((props) => {
  const field = useField<ArrayField>()
  const schema = useFieldSchema()
  if (!schema)
    throw new Error('can not found schema object')
  const dataSource = Array.isArray(field.value) ? field.value : []

  return (
    <ArrayContext.Provider value={{ field, schema }}>
      <div {...props}>
        {
          dataSource.map((_, index) => {
            const items = Array.isArray(schema.items)
              ? schema.items[index] || schema.items[0]
              : schema.items

            return (
              <ItemContext.Provider key={index} value={{ index }}>
                <RecursionField schema={items as Schema} name={index} />
              </ItemContext.Provider>
            )
          })
        }
        <Button
          type="button"
          onClick={() => {
            field?.push?.((schema.items as Schema)?.default)
          }}
        >
          Add
        </Button>
      </div>
    </ArrayContext.Provider>
  )
})

export function ArrayItem(props: PropsWithChildren) {
  const [css] = useStyletron()
  const index = useIndex()
  const array = useArray()

  if (!array || index === undefined)
    return null
  return (
    <div {...props} className={css({ display: 'flex' })}>
      <div className={css({ flex: 1, marginRight: '10px' })}>
        {props.children}
      </div>
      <Button
        type="button"
        kind="tertiary"
        shape="circle"
        onClick={() => {
          array.field?.remove?.(index)
        }}
      >
        <Delete size={24} />
      </Button>
    </div>
  )
}
