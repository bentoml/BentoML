import { Select } from 'baseui/select'
import { FormControl } from 'baseui/form-control'
import { useCallback, useEffect, useState } from 'react'
import { Button } from 'baseui/button'
import { FlexGrid, FlexGridItem } from 'baseui/flex-grid'
import useCurrentPath from '../hooks/useCurrentPath'
import { postData, useSchema } from '../hooks/useQuery'
import FormField from './FormField'

export default function InferenceForm() {
  const { data, isLoading } = useSchema()
  const { currentRoute, setCurrentPath } = useCurrentPath()
  const [result, setResult] = useState<object | string>()
  const [error, setError] = useState<string | undefined>()
  const [formData, setFormData] = useState<Record<string, unknown>>({})

  useEffect(() => {
    if (!currentRoute)
      return
    setFormData(Object.entries(currentRoute.input.properties ?? {}).reduce((acc, [key, schema]) => {
      return {
        ...acc,
        [key]: schema.default,
      }
    }, {}))
  }, [setFormData, currentRoute])

  const handleChange = useCallback((key: string, value: unknown) => {
    setFormData((prev) => {
      return {
        ...prev,
        [key]: value,
      }
    })
  }, [setFormData])

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()

    if (!currentRoute)
      return
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
            isLoading={isLoading}
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
          <form onSubmit={handleSubmit}>
            {Object.entries(currentRoute?.input.properties ?? {}).map(([key, schema]) => {
              return (
                <FormControl label={schema.title} caption={schema.description} key={key}>
                  <FormField value={formData[key]} schema={schema} onChange={(value: unknown) => handleChange(key, value)} />
                </FormControl>
              )
            })}
            <Button type="submit">Submit</Button>
          </form>
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
