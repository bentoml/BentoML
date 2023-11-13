import { useEffect } from 'react'
import { atom, useAtom } from 'jotai'
import useSystemTheme from './useSystemTheme'

type BaseThemeType = 'light' | 'dark'
type ThemType = BaseThemeType | 'system'

const storageKey = 'theme'
const themeAtom = atom<ThemType>('system')

export default function useTheme() {
  const [theme, setTheme] = useAtom(themeAtom)

  const systemTheme = useSystemTheme()

  useEffect(() => {
    const v = window.localStorage.getItem(storageKey)
    if (v)
      setTheme(v as ThemType)
  }, [setTheme])

  return [theme === 'system' ? systemTheme : theme, (t: BaseThemeType) => {
    const v = t === systemTheme ? 'system' : t
    window.localStorage.setItem(storageKey, v)
    setTheme(v)
  }] as [BaseThemeType, (t: BaseThemeType) => void]
}
