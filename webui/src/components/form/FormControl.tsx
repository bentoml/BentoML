import { FormControl as BaseUIFormControl } from 'baseui/form-control'
import { connect, mapProps } from '@formily/react'

export default connect(
  BaseUIFormControl,
  mapProps((props, field) => {
    return {
      ...props,
      label: field.title || props.label,
      caption: field.description,
    }
  }),
)
