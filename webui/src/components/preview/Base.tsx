import type { PropsWithChildren } from 'react'
import { useStyletron } from 'baseui'
import { Button, KIND, SHAPE, SIZE } from 'baseui/button'
import { Delete } from 'baseui/icon'
import { IconDownload, IconFileInvoice } from '@tabler/icons-react'
import BaseDownload from './Download'

interface IBasePreviewProps {
  operation?: JSX.Element
  value: File
}

interface IOperationProps {
  onClick?: () => void
}

interface IDownloadProps {
  value: File
}

export function BasePreview({ operation, value, children }: PropsWithChildren<IBasePreviewProps>) {
  const [css, theme] = useStyletron()

  return (
    <div
      className={css({
        'display': 'flex',
        'alignItems': 'center',
        'padding': theme.sizing.scale100,
        'borderRadius': theme.borders.radius200,
        'transitionProperty': 'background',
        'transitionDuration': theme.animation.timing200,
        'transitionTimingFunction': theme.animation.linearCurve,
        ':hover': {
          background: theme.colors.buttonSecondaryHover,
        },
      })}
    >
      {operation ?? <IconFileInvoice size={18} />}
      <span
        title={value.name}
        className={css({
          ...(theme.typography.LabelSmall),
          margin: `0 ${theme.sizing.scale200}`,
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          flex: 1,
        })}
      >
        {value.name}
      </span>
      {children}
    </div>
  )
}

export function Remove({ onClick }: IOperationProps) {
  return (
    <Button
      type="button"
      kind={KIND.tertiary}
      size={SIZE.mini}
      shape={SHAPE.circle}
      onClick={onClick}
    >
      <Delete size={24} />
    </Button>
  )
}

export function Download({ value }: IDownloadProps) {
  return (
    <BaseDownload
      kind={KIND.tertiary}
      size={SIZE.mini}
      shape={SHAPE.circle}
      value={value}
    >
      <IconDownload size={18} />
    </BaseDownload>
  )
}

export default BasePreview
