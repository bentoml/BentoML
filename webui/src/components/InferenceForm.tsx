import type { Key, ReactElement } from 'react'
import { createElement, useEffect, useMemo, useState } from 'react'
import { Select } from 'baseui/select'
import { createForm } from '@formily/core'
import { FormConsumer } from '@formily/react'
import { FlexGrid, FlexGridItem } from 'baseui/flex-grid'
import useCurrentPath from '../hooks/useCurrentPath'
import { transformData, useFormSubmit, useSchema } from '../hooks/useQuery'
import { useMountOptions } from '../hooks/useMountOptions'
import type { IRoute } from '../types'
import { Tab, Tabs } from './Tabs'
import Form from './form/Form'
import FormField, { generateFormSchema } from './form/FormField'
import Submit from './form/Submit'
import type { IClientProps } from './code/Base'
import CURL from './code/CURL'
import Python from './code/Python'
import Output from './Output'
import AuthInput from './AuthInput'

const codeExamples = {
  Python,
  CURL,
}

interface IPanelProps {
  route: IRoute
}

function Panel({ route }: IPanelProps) {
  const [activeTab, setActiveTab] = useState<Key>('0')
  const [result, setResult] = useState<any>()
  const [error, setError] = useState<Error>()
  const form = useMemo(() => createForm({ validateFirst: true }), [route])
  const submit = useFormSubmit(form, route)
  const formSchema = generateFormSchema(route.input)
  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()

    try {
      const res = await submit()

      setResult(res)
      setError(undefined)
    }
    catch (err) {
      if (err instanceof Error) {
        setResult(undefined)
        setError(err)
      }
    }
  }

  useEffect(() => {
    setResult(undefined)
    setError(undefined)
  }, [route, setResult, setError])

  return (
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
                      path: route.route,
                      values: transformData(form.values, route.input),
                      schema: route.input,
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
                    schema={route.output}
                  />
                  )
                : null}
          </Tab>
        </Tabs>
      </FlexGridItem>
    </FlexGrid>
  )
}

export default function InferenceForm() {
  const data = useSchema()
  const { needAuth } = useMountOptions()
  const { currentRoute, setCurrentPath } = useCurrentPath()

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
        {needAuth && <AuthInput />}
      </div>
      {currentRoute && <Panel route={currentRoute} />}
    </>
  )
}
