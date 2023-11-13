import { useStyletron } from 'baseui'
import { StyledLink } from 'baseui/link'

export default function Footer() {
  const [css, theme] = useStyletron()
  return (
    <footer className={css({
      borderTop: `1px solid ${theme.borders.border300.borderColor}`,
      padding: '1rem 0',
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      gap: '1rem',
      color: theme.colors.contentTertiary,
    })}
    >
      <div>
        Powered by
        <StyledLink href="https://bentoml.com" target="_blank">BentoML</StyledLink>
      </div>
    </footer>
  )
}
