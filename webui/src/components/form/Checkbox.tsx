import type { CheckboxProps } from 'baseui/checkbox'
import { Checkbox as BaseUICheckbox } from 'baseui/checkbox'

export interface ICheckboxProps extends Omit<CheckboxProps, 'onChange'> {
  onChange?: (value: boolean) => void
}

export default function Checkbox({ onChange, ...restProps }: ICheckboxProps) {
  return (
    <BaseUICheckbox onChange={e => onChange?.(e.target.checked)} {...restProps} />
  )
}
