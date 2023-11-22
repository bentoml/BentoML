import { useState } from 'react'
import { Select } from 'baseui/select'
import { FormControl } from 'baseui/form-control'
import { createForm } from '@formily/core'
import { FlexGrid, FlexGridItem } from 'baseui/flex-grid'
import useCurrentPath from '../hooks/useCurrentPath'
import { postData, useSchema } from '../hooks/useQuery'
import Form from './form/Form'
import FormField, { generateFormSchema } from './form/FormField'
import Submit from './form/Submit'

export default function InferenceForm() {
  const data = useSchema()
  const { currentRoute, setCurrentPath } = useCurrentPath()
  const [result, setResult] = useState<object | string>()
  const [error, setError] = useState<string | undefined>()
  const form = createForm({})
  const formSchema = generateFormSchema(currentRoute?.input)
  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()

    if (!currentRoute)
      return
    const formData = await form.submit()
    const resp = await postData(currentRoute.route, formData)
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
            <FormField schema={formSchema} />
            <Submit>Submit</Submit>
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
