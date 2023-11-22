import type { PropsWithChildren } from 'react'
import { observer, useParentForm } from '@formily/react'
import type { ButtonProps } from 'baseui/button'
import { Button } from 'baseui/button'

export default observer<PropsWithChildren<ButtonProps>>(
  (props) => {
    const form = useParentForm()

    return (
      <Button
        {...props}
        isLoading={props.isLoading !== undefined ? props.isLoading : form.submitting}
      >
        {props.children}
      </Button>
    )
  },
  { forwardRef: true },
)
