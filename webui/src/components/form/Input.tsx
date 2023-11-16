import type { InputProps } from 'baseui/input'
import { Input as BaseUIInput } from 'baseui/input'

export interface IInputProps extends Omit<InputProps, 'onChange'> {
  onChange?: (value: string) => void
}

export default function Input({ onChange, ...restProps }: IInputProps) {
  return (
    <BaseUIInput onChange={e => onChange?.(e.target.value)} {...restProps} />
  )
}
