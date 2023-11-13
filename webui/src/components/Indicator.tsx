import useSWR from 'swr'
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
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: '0.5rem',
      margin: '1rem 0',
    }}
    >
      <span>
        GET
        {' '}
        <StyledLink href={endpoint} target="_blank" rel="noreferrer">
          <strong>{endpoint}</strong>
        </StyledLink>
        :
        {' '}
      </span>
      <Circle color={isLoading ? 'yellow' : error ? 'red' : 'green'} width={16} height={16} style={{ flexShrink: 0 }} />
      <span>{children}</span>
    </div>
  )
}
