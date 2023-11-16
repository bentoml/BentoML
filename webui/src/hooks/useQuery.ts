import { useContext } from 'react'
import { JSONSchemaContext } from '../components/JSONSchema'

export const fetcher = (input: RequestInfo | URL, init?: RequestInit) => fetch(input, init).then(res => res.json())

export function useSchema() {
  return useContext(JSONSchemaContext)
}

export function postData(url: string, data: unknown) {
  return fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  })
}

export function postMultipart(url: string, data: FormData) {
  return fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'multipart/form-data' },
    body: data,
  })
}
