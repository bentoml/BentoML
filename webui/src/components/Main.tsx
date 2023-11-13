import { useStyletron } from 'baseui'
import { Card, StyledBody } from 'baseui/card'
import { Spinner } from 'baseui/spinner'
import { DisplaySmall } from 'baseui/typography'
import { useSchema } from '../hooks/useQuery'
import Indicator from './Indicator'
import Router from './Router'

function Inner() {
  const { data, error, isLoading } = useSchema()
  const [css, theme] = useStyletron()

  if (isLoading) {
    return (
      <div style={{
        display: 'flex',
        justifyContent: 'center',
        padding: '1rem',
      }}
      >
        <Spinner />
      </div>
    )
  }
  if (error) {
    return (
      <Card>
        <StyledBody>
          Error:
          {error.message}
        </StyledBody>
      </Card>
    )
  }

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
