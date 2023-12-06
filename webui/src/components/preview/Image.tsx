import type { PropsWithChildren } from 'react'
import { useState } from 'react'
import { useStyletron } from 'baseui'
import { KIND as BUTTON_KIND, SHAPE as BUTTON_SHAPE, SIZE as BUTTON_SIZE, Button } from 'baseui/button'
import { ROLE as MODAL_ROLE, SIZE as MODAL_SIZE, Modal, ModalBody } from 'baseui/modal'
import { Delete, Show } from 'baseui/icon'
import { IconDownload } from '@tabler/icons-react'
import BaseDownload from './Download'

interface IImagePreviewProps {
  value: File
}

interface IRemoveProps {
  onClick: () => void
}

interface IDownloadProps {
  value: File
}

function ImagePreview({ value, children }: PropsWithChildren<IImagePreviewProps>) {
  const [css, theme] = useStyletron()
  const [isOpen, setIsOpen] = useState(false)

  return (
    <div
      className={css({
        backgroundImage: `url(${URL.createObjectURL(value)})`,
        backgroundSize: 'cover',
        backgroundPosition: 'center',
        height: '200px',
        borderRadius: theme.borders.radius400,
        borderColor: theme.colors.fileUploaderBorderColorDefault,
        borderStyle: 'solid',
        borderWidth: theme.sizing.scale0,
        boxSizing: 'border-box',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      })}
    >
      <Button
        type="button"
        onClick={() => setIsOpen(true)}
        startEnhancer={() => <Show size={18} />}
        size={BUTTON_SIZE.compact}
        kind={BUTTON_KIND.secondary}
        shape={BUTTON_SHAPE.pill}
      >
        Preview
      </Button>
      <Modal
        closeable
        isOpen={isOpen}
        onClose={() => setIsOpen(false)}
        animate
        autoFocus
        size={MODAL_SIZE.auto}
        role={MODAL_ROLE.dialog}
        overrides={{
          Dialog: {
            props: {
              className: css({ borderRadius: '0!important' }),
            },
          },
        }}
      >
        <ModalBody className={css({ margin: '0!important' })}>
          <img
            className={css({ width: '100%', maxWidth: '60vw', display: 'block' })}
            src={URL.createObjectURL(value)}
            title={value.name}
            alt={value.name}
          />
        </ModalBody>
      </Modal>
      {children}
    </div>
  )
}

export function Remove({ onClick }: IRemoveProps) {
  const [css, theme] = useStyletron()

  return (
    <Button
      size={BUTTON_SIZE.compact}
      shape={BUTTON_SHAPE.circle}
      kind={BUTTON_KIND.secondary}
      type="button"
      onClick={onClick}
      className={css({ marginLeft: theme.sizing.scale300 })}
    >
      <Delete size={18} />
    </Button>
  )
}

export function Download({ value }: IDownloadProps) {
  const [css, theme] = useStyletron()

  return (
    <BaseDownload
      size={BUTTON_SIZE.compact}
      shape={BUTTON_SHAPE.circle}
      kind={BUTTON_KIND.secondary}
      value={value}
      className={css({ marginLeft: theme.sizing.scale300 })}
    >
      <IconDownload size={18} />
    </BaseDownload>
  )
}

export default ImagePreview
