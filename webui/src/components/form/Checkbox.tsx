import { connect, mapProps } from '@formily/react'
import type { CheckboxProps } from 'baseui/checkbox'
import { Checkbox as BaseUICheckbox } from 'baseui/checkbox'

export interface ICheckboxProps extends Omit<CheckboxProps, 'onChange'> {
  onChange?: (value: boolean) => void
}

export function Checkbox({ onChange, ...restProps }: ICheckboxProps) {
  return (
    <BaseUICheckbox onChange={e => onChange?.(e.target.checked)} {...restProps} />
  )
}

export default connect(Checkbox, mapProps((props) => {
  return {
    ...props,
    checked: props.value as unknown as boolean,
  }
}))
