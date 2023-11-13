import { BaseProvider, DarkTheme, LightTheme } from 'baseui'
import { Client as Styletron } from 'styletron-engine-atomic'
import { Provider as StyletronProvider } from 'styletron-react'
import useTheme from './hooks/useTheme'
import Layout from './components/Layout'

const engine = new Styletron()

function App() {
  const [theme] = useTheme()

  return (
    <StyletronProvider value={engine}>
      <BaseProvider theme={theme === 'dark' ? DarkTheme : LightTheme}>
        <Layout />
      </BaseProvider>
    </StyletronProvider>
  )
}

export default App
