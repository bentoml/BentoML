import type { ChangeEvent } from 'react'
import { useCallback, useMemo } from 'react'
import type { InputProps } from 'baseui/input'
import { useStyletron } from 'baseui'
import { Input as BaseUIInput } from 'baseui/input'
import { Slider as BaseUISlider } from 'baseui/slider'

export interface IInputNumberProps extends Omit<InputProps, 'max' | 'min' | 'step' | 'value' | 'onChange'> {
  value: number
  step?: number
  minimum?: number
  maximum?: number
  exclusiveMinimum?: number
  exclusiveMaximum?: number
  onChange?: (value: number) => void
}

export default function InputNumber({
  value,
  step = 1,
  onChange,
  minimum,
  maximum,
  exclusiveMinimum,
  exclusiveMaximum,
  ...restProps
}: IInputNumberProps) {
  const [css] = useStyletron()
  const min = useMemo(() => {
    if (exclusiveMinimum !== undefined)
      return exclusiveMinimum + step
    if (minimum !== undefined)
      return minimum
    return undefined
  }, [minimum, exclusiveMinimum])
  const max = useMemo(() => {
    if (exclusiveMaximum !== undefined)
      return exclusiveMaximum - step
    if (maximum !== undefined)
      return maximum
    return undefined
  }, [maximum, exclusiveMaximum])
  // component show slider when both max and min exist
  const showSlider = useMemo(() => min !== undefined && max !== undefined, [min, max])
  const handleInputChange = useCallback((e: ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    // e.target.valueAsNumber is NaN when input's value is '-' or empty string
    // TODO: typescript unsupport type of ChangeEvent valueAsNumber, but most browsers support it
    // https://caniuse.com/mdn-api_htmlinputelement_valueasnumber
    let value: number = (e.target as any).valueAsNumber
    if (Number.isNaN(value))
      return onChange?.(e.target.value as unknown as number)

    // check range
    if (max && value > max)
      value = max
    if (min && value < min)
      value = min
    onChange?.(value)
  }, [onChange, min, max])

  return (
    <div
      className={css({
        display: 'flex',
        alignItems: 'flex-start',
      })}
    >
      <BaseUIInput
        {...restProps}
        step={step}
        min={min}
        max={max}
        value={value}
        type="number"
        overrides={{
          Root: {
            props: {
              className: css({ flex: 1 }),
            },
          },
        }}
        onChange={handleInputChange}
      />
      {showSlider && (
        <BaseUISlider
          value={[value || 0]}
          step={step}
          min={min}
          max={max}
          overrides={{
            Root: {
              props: {
                className: css({
                  flex: 2,
                  marginTop: '5px',
                }),
              },
            },
          }}
          onChange={({ value: [value] }) => value && onChange?.(value)}
        />
      )}
    </div>
  )
}
