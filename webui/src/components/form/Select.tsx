import type { SelectProps, Value } from 'baseui/select'
import { Select as BaseUISelect } from 'baseui/select'

interface ISelectProps extends Omit<SelectProps, 'value' | 'options' | 'onChange'> {
  value: string | number
  onChange?: (value?: string | number) => void
  options: Value
}

function Select({ value, onChange, options, clearable, ...restProps }: ISelectProps) {
  return (
    <BaseUISelect
      {...restProps}
      options={options}
      clearable={clearable}
      value={options?.filter(option => option.id === value)}
      onChange={(params) => {
        if (clearable || params.option?.id)
          onChange?.(params?.option?.id)
      }}
    />
  )
}

export default Select
