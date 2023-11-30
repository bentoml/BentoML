import useSWR from 'swr'
import { useStyletron } from 'baseui'
import { StyledLink } from 'baseui/link'
import Circle from '../assets/circle.svg?react'

interface IIndicatorProps {
  endpoint: string
  children: React.ReactNode
}

async function fetcher(url: string) {
  const resp = await fetch(url)
  const text = await resp.text()
  if (resp.status !== 200)
    throw new Error(`Failed to fetch ${url}: ${resp.status} ${text}`)

  return text
}

export default function Indicator({ endpoint, children }: IIndicatorProps) {
  const [css] = useStyletron()
  const { error, isLoading } = useSWR(endpoint, fetcher, {
    onErrorRetry: (error, _key, _config, revalidate, { retryCount }) => {
      // Never retry on 404.
      if (error.status === 404)
        return

      // Only retry up to 10 times.
      if (retryCount >= 10)
        return

      // Retry after 5 seconds.
      setTimeout(() => revalidate({ retryCount }), 5000)
    },
  })
  return (
    <div
      className={css({
        display: 'flex',
        alignItems: 'flex-top',
        gap: '0.5rem',
        margin: '1rem 0',
        lineHeight: 1.1,
      })}
    >
      <span className={css({ whiteSpace: 'nowrap' })}>
        GET
        {' '}
        <StyledLink href={endpoint} target="_blank" rel="noreferrer">
          <strong>{endpoint}</strong>
        </StyledLink>
        :
        {' '}
      </span>
      <Circle
        color={isLoading ? 'yellow' : error ? 'red' : 'green'}
        width={16}
        height={16}
        className={css({ flexShrink: 0, marginTop: '2px' })}
      />
      <span>{children}</span>
    </div>
  )
}
