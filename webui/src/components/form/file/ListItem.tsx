import { useStyletron } from 'baseui'
import { Button, KIND, SHAPE, SIZE } from 'baseui/button'
import { Delete } from 'baseui/icon'

interface IListItemProps {
  before: JSX.Element
  value: File
  onRemove: () => void
}

export function ListItem({ before, value, onRemove }: IListItemProps) {
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
      {before}
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
      <Button
        type="button"
        kind={KIND.tertiary}
        size={SIZE.mini}
        shape={SHAPE.circle}
        onClick={onRemove}
      >
        <Delete size={24} />
      </Button>
    </div>
  )
}

export default ListItem
