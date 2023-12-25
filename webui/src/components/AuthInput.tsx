import { Button } from 'baseui/button'
import { IconLock, IconLockOpen } from '@tabler/icons-react'
import { StatefulPopover } from 'baseui/popover'
import { useEffect, useMemo, useState } from 'react'
import { Block } from 'baseui/block'
import { Input } from 'baseui/input'
import useToken from '../hooks/useToken'

export default function AuthInput() {
  const [token, setToken] = useToken()
  const [tokenInput, setTokenInput] = useState('')

  const hasToken = useMemo(() => token !== null && token !== '', [token])

  useEffect(() => {
    if (token)
      setTokenInput(token)
  }, [token, setTokenInput])

  return (
    <StatefulPopover
      dismissOnClickOutside={false}
      content={({ close }) => (
        <Block padding="20px" display="flex">
          <Input
            placeholder="Input Token"
            type="password"
            value={tokenInput}
            onChange={(e) => {
              setTokenInput(e.target.value)
            }}
          />
          <Button
            style={{ marginLeft: '0.5rem' }}
            onClick={() => {
              setToken(tokenInput)
              close()
            }}
          >
            OK
          </Button>
        </Block>
      )}
      returnFocus
      autoFocus
    >
      <Button
        size="compact"
        shape="pill"
        startEnhancer={hasToken ? <IconLockOpen /> : <IconLock />}
      >
        {hasToken ? 'Authorized' : 'Unauthorized'}
      </Button>
    </StatefulPopover>
  )
}
