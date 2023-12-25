import type { ButtonProps } from 'baseui/button'
import { Button } from 'baseui/button'

interface IDownloadProps extends Omit<ButtonProps, 'onClick'> {
  value: File
  className?: string
}

function Download({ value, ...props }: IDownloadProps) {
  return (
    <Button
      {...props}
      type="button"
      onClick={() => {
        const link = document.createElement('a')

        link.href = URL.createObjectURL(value)
        link.download = value.name
        link.click()
      }}
    />
  )
}

export default Download
