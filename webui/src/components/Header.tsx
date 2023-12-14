import type { Theme } from 'baseui'
import { useStyletron } from 'baseui'
import { StyledLink } from 'baseui/link'
import { Button, KIND, SIZE } from 'baseui/button'
import { createUseStyles } from 'react-jss'
import { useCallback } from 'react'
import useTheme from '../hooks/useTheme'
import Bulb from '../assets/bulb.svg?react'

const useStyles = createUseStyles({
  header: {
    padding: '1rem 1.5rem',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    borderBottom: ({ theme }: { theme: Theme }) => `1px solid ${theme.borders.border300.borderColor}`,
  },
})

export default function Header() {
  const [themeType, setThemeType] = useTheme()
  const [,theme] = useStyletron()
  const classes = useStyles({ theme })

  const toggleTheme = useCallback(() => {
    setThemeType(themeType === 'dark' ? 'light' : 'dark')
  }, [themeType, setThemeType])

  return (
    <header className={classes.header}>
      <StyledLink href="https://bentoml.com" target="_blank">
        <img src={themeType === 'dark' ? '/assets/bentoml-logo-white.png' : '/assets/bentoml-logo-black.png'} alt="BentoML Logo" height={36} />
      </StyledLink>
      <Button size={SIZE.mini} kind={KIND.tertiary} onClick={toggleTheme}><Bulb /></Button>
    </header>
  )
}
