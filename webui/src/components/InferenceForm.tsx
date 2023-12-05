import type { FC, Key } from 'react'
import { createElement, useMemo, useState } from 'react'
import { useStyletron } from 'baseui'
import { Select } from 'baseui/select'
import { FormControl } from 'baseui/form-control'
import { createForm } from '@formily/core'
import { FormConsumer } from '@formily/react'
import { FlexGrid, FlexGridItem } from 'baseui/flex-grid'
import { Tab, Tabs } from 'baseui/tabs'
import useCurrentPath from '../hooks/useCurrentPath'
import { transformData, useFormSubmit, useSchema } from '../hooks/useQuery'
import Form from './form/Form'
import FormField, { generateFormSchema } from './form/FormField'
import Submit from './form/Submit'
import type { IClientProps } from './code/Base'
import CURL from './code/CURL'
import Python from './code/Python'

const codeExamples = {
  Python,
  CURL,
}

export default function InferenceForm() {
  const [css, theme] = useStyletron()
  const data = useSchema()
  const { currentRoute, setCurrentPath } = useCurrentPath()
  const [activeTab, setActiveTab] = useState<Key>('0')
  const [result, setResult] = useState<object | string>()
  const [error, setError] = useState<string | undefined>()
  const form = useMemo(() => createForm({}), [currentRoute])
  const submit = useFormSubmit(form, currentRoute?.input)
  const formSchema = generateFormSchema(currentRoute?.input)
  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()

    if (!currentRoute)
      return
    const resp = await submit(currentRoute.route)
    if (resp.status >= 400) {
      const e = await resp.text()
      setError(`Error: ${resp.status} ${e}`)
    }
    else {
      setError(undefined)
      setResult(await resp.json())
    }
  }

  return (
    <>
      <div style={{ display: 'flex', gap: 4, alignItems: 'center', margin: '1rem 0' }}>
        <strong>POST</strong>
        <div style={{ minWidth: '300px' }}>
          <Select
            options={data?.routes.map(route => ({ label: route.route, id: route.name }))}
            placeholder="Select endpoint"
            value={currentRoute ? [{ label: currentRoute.route, id: currentRoute.name }] : []}
            clearable={false}
            onChange={(params) => {
              if (!params.option)
                return
              setCurrentPath(params.option.label as string)
            }}
          />
        </div>
      </div>
      <FlexGrid
        flexGridColumnCount={[1, 1, 2, 2]}
        flexGridColumnGap="scale800"
        flexGridRowGap="scale800"
      >
        <FlexGridItem>
          <Form form={form} onSubmit={handleSubmit}>
            <Tabs
              activeKey={activeTab}
              onChange={({ activeKey }) => setActiveTab(activeKey)}
              overrides={{
                TabBar: {
                  props: {
                    className: css({
                      backgroundColor: 'transparent!important',
                      paddingLeft: '0!important',
                      paddingRight: '0!important',
                      borderBottomWidth: '1px',
                      borderBottomStyle: 'solid',
                      borderBottomColor: `${theme.colors.borderOpaque}`,
                    }),
                  },
                },
                TabContent: {
                  props: {
                    className: css({
                      paddingLeft: '0!important',
                      paddingRight: '0!important',
                    }),
                  },
                },
              }}
            >
              <Tab title="Form">
                <FormField schema={formSchema} />
                <Submit>Submit</Submit>
              </Tab>
              <>
                {
                  Object.entries(codeExamples).map(([title, comp]) => (
                    <Tab title={title} key={title}>
                      <FormConsumer>
                        {() => createElement(comp as FC<IClientProps>, {
                          path: currentRoute?.route,
                          values: transformData(form.values, currentRoute?.input),
                          schema: currentRoute?.input,
                        })}
                      </FormConsumer>
                    </Tab>
                  ))
                }
              </>
            </Tabs>
          </Form>
        </FlexGridItem>
        <FlexGridItem>
          <FormControl label="Result" error={error}>
            <div>{JSON.stringify(result)}</div>
          </FormControl>
        </FlexGridItem>
      </FlexGrid>
    </>
  )
}
