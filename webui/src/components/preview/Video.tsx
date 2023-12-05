import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useStyletron } from 'baseui'
import { Card, StyledAction, StyledBody } from 'baseui/card'
import { StyledDivider } from 'baseui/divider'
import { Button, KIND, SHAPE, SIZE } from 'baseui/button'
import { IconPlayerPauseFilled, IconPlayerPlayFilled } from '@tabler/icons-react'
import BasePreview from './Base'

interface IVideoPlayerProps {
  files: File[]
  children: (
    file: File,
    index: number,
    activeIndex: number,
    setActive: (file: File) => void,
  ) => JSX.Element
}

function VideoPlayer({ files, children }: IVideoPlayerProps) {
  const [css] = useStyletron()
  const videoRef = useRef<HTMLVideoElement>(null)
  const [activeFile, setActiveFile] = useState(files[0])
  const activeIndex = files.indexOf(activeFile)
  const videoSrc = useMemo(() => URL.createObjectURL(activeFile), [activeFile])
  const [playerState, setPlayerState] = useState(false)
  const toggle = useCallback((file: File) => {
    if (file !== activeFile) {
      const cb = () => {
        videoRef.current?.play()
        videoRef.current?.removeEventListener('loadedmetadata', cb)
      }

      setActiveFile(file)
      videoRef.current?.addEventListener('loadedmetadata', cb)
    }
    else if (playerState) {
      videoRef.current?.pause()
    }
    else {
      videoRef.current?.play()
    }
  }, [videoRef, playerState, activeFile])

  useEffect(() => {
    const video = videoRef.current
    const onPaused = () => {
      setPlayerState(false)
    }
    const onPlay = () => {
      setPlayerState(true)
    }
    const onEnded = () => {
      setPlayerState(false)
    }

    video?.addEventListener('pause', onPaused)
    video?.addEventListener('play', onPlay)
    video?.addEventListener('ended', onEnded)

    return () => {
      video?.removeEventListener('pause', onPaused)
      video?.removeEventListener('play', onPlay)
      video?.removeEventListener('ended', onEnded)
    }
  }, [])

  return (
    <Card overrides={{ Root: { props: { className: css({ backgroundColor: 'transparent!important' }) } } }}>
      <StyledBody>
        <video
          ref={videoRef}
          style={{ width: '100%', display: 'block' }}
          src={videoSrc}
          controls
        />
      </StyledBody>
      <StyledAction>
        <StyledDivider />
        {files.map((file, index) => (
          <BasePreview
            key={index}
            operation={(
              <Button
                type="button"
                size={SIZE.mini}
                kind={KIND.tertiary}
                shape={SHAPE.circle}
                onClick={() => toggle(file)}
              >
                {
                  index !== activeIndex
                    ? (
                      <IconPlayerPlayFilled size={14} />
                      )
                    : playerState
                      ? (
                        <IconPlayerPauseFilled size={14} />
                        )
                      : (
                        <IconPlayerPlayFilled size={14} />
                        )
                }
              </Button>
            )}
            value={file}
          >
            {children(file, index, activeIndex, setActiveFile)}
          </BasePreview>
        ))}
      </StyledAction>
    </Card>
  )
}

export default VideoPlayer
