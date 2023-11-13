import { useStyletron } from 'baseui'
import Header from './Header'
import Main from './Main'
import Footer from './Footer'

export default function Layout() {
  const [css, theme] = useStyletron()

  return (
    <div className={css({
      backgroundColor: theme.colors.backgroundSecondary,
      minHeight: '100vh',
      display: 'flex',
      flexDirection: 'column',
      color: theme.colors.contentPrimary,
    })}
    >
      <Header />
      <Main />
      <Footer />
    </div>
  )
}
