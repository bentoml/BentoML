import { useStyletron } from 'baseui'
import { DisplaySmall } from 'baseui/typography'
import { useSchema } from '../hooks/useQuery'
import Router from './Router'

function Inner() {
  const data = useSchema()

  return (
    <>
      <DisplaySmall>{data?.name}</DisplaySmall>
      {data?.description && <p>{data?.description}</p>}
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
