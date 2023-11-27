import { useState } from 'react'
import { useStyletron } from 'baseui'
import type { ButtonProps } from 'baseui/button'
import { Button } from 'baseui/button'
import { ROLE as MODAL_ROLE, SIZE as MODAL_SIZE, Modal, ModalBody } from 'baseui/modal'
import { Show } from 'baseui/icon'

interface IPreviewProps extends ButtonProps {
  value: File
}

function Preview({ value, ...buttonProps }: IPreviewProps) {
  const [css] = useStyletron()
  const [isOpen, setIsOpen] = useState(false)

  return (
    <>
      <Button
        type="button"
        onClick={() => setIsOpen(true)}
        startEnhancer={() => <Show size={18} />}
        {...buttonProps}
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
    </>
  )
}

export default Preview
