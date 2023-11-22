import { useStyletron } from 'baseui'
import { DisplaySmall } from 'baseui/typography'
import { useSchema } from '../hooks/useQuery'
import Indicator from './Indicator'
import Router from './Router'

function Inner() {
  const data = useSchema()
  const [css, theme] = useStyletron()

  return (
    <>
      <DisplaySmall>{data?.name}</DisplaySmall>
      {data?.description && <p>{data?.description}</p>}
      <div className={css({
        borderBottom: `1px solid ${theme.borders.border300.borderColor}`,
      })}
      >
        <div><strong>System Endpoints</strong></div>
        <div>
          <Indicator endpoint="/readyz">
            A
            {' '}
            <code>200</code>
            {' '}
            OK status from
            {' '}
            <code>/readyz</code>
            {' '}
            endpoint indicated the service is ready to accept traffic. From that point and onward, Kubernetes will use
            {' '}
            <code>/livez</code>
            {' '}
            endpoint to perform periodic health checks.
          </Indicator>
          <Indicator endpoint="/livez">
            Health check endpoint for Kubernetes. Healthy endpoint responses with a
            {' '}
            <code>200</code>
            {' '}
            OK status.
          </Indicator>
        </div>
      </div>
      <Router />
    </>
  )
}

export default function Main() {
  const [css] = useStyletron()

  return (
    <main className={css({
      display: 'flex',
      flexDirection: 'column',
      flexGrow: 1,
      padding: '1rem',
      gap: '5px',
    })}
    >
      <Inner />
    </main>
  )
}
