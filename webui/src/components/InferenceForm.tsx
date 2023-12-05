import type { Key, ReactElement } from 'react'
import { createElement, useEffect, useMemo, useState } from 'react'
import { Select } from 'baseui/select'
import { createForm } from '@formily/core'
import { FormConsumer } from '@formily/react'
import { FlexGrid, FlexGridItem } from 'baseui/flex-grid'
import useCurrentPath from '../hooks/useCurrentPath'
import { transformData, useFormSubmit, useSchema } from '../hooks/useQuery'
import { Tab, Tabs } from './Tabs'
import Form from './form/Form'
import FormField, { generateFormSchema } from './form/FormField'
import Submit from './form/Submit'
import type { IClientProps } from './code/Base'
import CURL from './code/CURL'
import Python from './code/Python'
import Output from './Output'

const codeExamples = {
  Python,
  CURL,
}

export default function InferenceForm() {
  const data = useSchema()
  const { currentRoute, setCurrentPath } = useCurrentPath()
  const [activeTab, setActiveTab] = useState<Key>('0')
  const [result, setResult] = useState<any>()
  const [error, setError] = useState<Error>()
  const form = useMemo(() => createForm({}), [currentRoute])
  const submit = useFormSubmit(form, currentRoute)
  const formSchema = generateFormSchema(currentRoute?.input)
  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()

    try {
      const res = await submit()

      setResult(res)
    }
    catch (err) {
      setError(err as Error)
    }
  }

  useEffect(() => {
    setResult(undefined)
    setError(undefined)
  }, [currentRoute, setResult, setError])

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
      {currentRoute && (
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
              >
                <Tab title="Form">
                  <FormField schema={formSchema} />
                  <Submit>Submit</Submit>
                </Tab>
                {
                  Object.entries(codeExamples).map(([title, comp]) => (
                    <Tab title={title} key={title}>
                      <FormConsumer>
                        {() => createElement<IClientProps>(comp, {
                          path: currentRoute.route,
                          values: transformData(form.values, currentRoute.input),
                          schema: currentRoute.input,
                        })}
                      </FormConsumer>
                    </Tab> as ReactElement<any, any>
                  ))
                }
              </Tabs>
            </Form>
          </FlexGridItem>
          <FlexGridItem>
            <Tabs activeKey="0">
              <Tab title="Result">
                {error
                  ? (
                    <div style={{ color: 'red' }}>
                      Error:
                      {' '}
                      {error.message}
                    </div>
                    )
                  : result
                    ? (
                      <Output
                        result={result}
                        schema={currentRoute.output}
                      />
                      )
                    : null}
              </Tab>
            </Tabs>
          </FlexGridItem>
        </FlexGrid>
      )}
    </>
  )
}
