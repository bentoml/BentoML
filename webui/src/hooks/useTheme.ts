import { useEffect, useMemo } from 'react'
import { atom, useAtom } from 'jotai'
import { LightTheme, useStyletron } from 'baseui'
import useSystemTheme from './useSystemTheme'
import { useMountOptions } from './useMountOptions'

type BaseThemeType = 'light' | 'dark'
export type ThemType = BaseThemeType | 'system'

const storageKey = 'theme'
const themeAtom = atom<ThemType>('system')

export default function useTheme() {
  const [theme, setTheme] = useAtom(themeAtom)
  const { theme: mountTheme } = useMountOptions()

  const systemTheme = useSystemTheme()

  useEffect(() => {
    const v = window.localStorage.getItem(storageKey)
    if (v)
      setTheme(v as ThemType)
  }, [setTheme])

  useEffect(() => {
    setTheme(mountTheme)
  }, [mountTheme, setTheme])

  return [theme === 'system' ? systemTheme : theme, (t: BaseThemeType) => {
    const v = t === systemTheme ? 'system' : t
    window.localStorage.setItem(storageKey, v)
    setTheme(v)
  }] as [BaseThemeType, (t: BaseThemeType) => void]
}

export function useIsLight() {
  const [, theme] = useStyletron()

  return useMemo(() => theme.name === LightTheme.name, [theme])
}
