import { atom, useAtom } from 'jotai'
import { useEffect } from 'react'

const storageKey = 'token'
const tokenAtom = atom<string>('')

export default function useToken() {
  const [token, setToken] = useAtom(tokenAtom)

  useEffect(() => {
    setToken(window.localStorage.getItem(storageKey) || '')
  }, [setToken])

  useEffect(() => {
    if (token)
      window.localStorage.setItem(storageKey, token)
  }, [token])

  return [token, setToken] as const
}
