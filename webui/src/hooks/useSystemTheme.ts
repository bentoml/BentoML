import { useCallback, useEffect, useMemo, useState } from 'react'

export default function useSystemTheme() {
  const getCurrentTheme = () => window.matchMedia('(prefers-color-scheme: dark)').matches
  const [isDarkTheme, setIsDarkTheme] = useState(getCurrentTheme())
  const mqListener = useCallback((e: MediaQueryListEvent) => {
    setIsDarkTheme(e.matches)
  }, [])

  useEffect(() => {
    const darkThemeMq = window.matchMedia('(prefers-color-scheme: dark)')
    darkThemeMq.addEventListener('change', mqListener)
    return () => darkThemeMq.removeEventListener('change', mqListener)
  }, [mqListener])

  return useMemo(() => isDarkTheme ? 'dark' : 'light', [isDarkTheme])
}
