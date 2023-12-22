import { useStyletron } from 'baseui'
import { useMountOptions } from '../hooks/useMountOptions'
import Header from './Header'
import Main from './Main'
import Footer from './Footer'

export default function Layout() {
  const [css, theme] = useStyletron()
  const { header } = useMountOptions()

  return (
    <div className={css({
      backgroundColor: theme.colors.backgroundSecondary,
      minHeight: '100vh',
      display: 'flex',
      flexDirection: 'column',
      color: theme.colors.contentPrimary,
    })}
    >
      {header && <Header />}
      <Main />
      <Footer />
    </div>
  )
}
