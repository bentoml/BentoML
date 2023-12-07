import { FormControl as BaseUIFormControl } from 'baseui/form-control'
import { connect, mapProps } from '@formily/react'
import { isVoidField } from '@formily/core'

export default connect(
  BaseUIFormControl,
  mapProps((props, field) => {
    const mappedProps = {
      ...props,
      label: field.title || props.label,
      caption: field.description,
    }

    if (isVoidField(field))
      return mappedProps

    return {
      ...mappedProps,
      error: field.selfErrors.length ? field.selfErrors[0] : undefined,
    }
  }),
)
