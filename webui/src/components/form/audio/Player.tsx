import { useCallback, useEffect, useMemo, useState } from 'react'
import dayjs from 'dayjs'
import duration from 'dayjs/plugin/duration'
import { useStyletron } from 'baseui'
import { Card, StyledAction, StyledBody } from 'baseui/card'
import { Slider } from 'baseui/slider'
import { ButtonGroup, SHAPE, SIZE } from 'baseui/button-group'
import { Button, KIND } from 'baseui/button'
import { StyledDivider } from 'baseui/divider'
import {
  IconPlayerPauseFilled,
  IconPlayerPlayFilled,
  IconPlayerSkipBackFilled,
  IconPlayerSkipForwardFilled,
} from '@tabler/icons-react'
import ListItem from '../file/ListItem'

dayjs.extend(duration)

interface IPlayerProps {
  files: File[]
  onRemove: (index: number) => void
}

function Player({ files, onRemove }: IPlayerProps) {
  const [css, theme] = useStyletron()
  const timeStyle = css({
    ...(theme.typography.LabelSmall),
    width: '40px',
    textAlign: 'center',
  })
  const [activeFile, setActiveFile] = useState(files[0])
  const activeIndex = files.indexOf(activeFile)
  const audio = useMemo(() => new Audio(), [])
  const [playerState, setPlayerState] = useState(false)
  const [currentTime, setCurrentTime] = useState([0])
  const [duration, setDuration] = useState(audio.duration)
  const remove = useCallback((index: number) => {
    if (index === activeIndex && files.length > 1) {
      if (index === 0)
        setActiveFile(files[1])
      else
        setActiveFile(files[0])
    }
    onRemove(index)
  }, [activeIndex, setActiveFile, onRemove])
  const play = useCallback((file: File) => {
    const cb = () => {
      audio.play()
      audio.removeEventListener('loadedmetadata', cb)
    }

    setActiveFile(file)
    audio.addEventListener('loadedmetadata', cb)
  }, [audio, setActiveFile])
  const toggle = useCallback((file: File) => {
    if (file !== activeFile)
      return play(file)
    if (playerState)
      audio.pause()
    else
      audio.play()
  }, [audio, playerState, activeFile])

  useEffect(() => {
    const onLoadedMetadata = () => {
      setDuration(audio.duration)
    }
    const onTimeUpdate = () => {
      setCurrentTime([audio.currentTime])
    }
    const onPaused = () => {
      setPlayerState(false)
    }
    const onPlay = () => {
      setPlayerState(true)
    }
    const onEnded = () => {
      setPlayerState(false)
    }
    audio.addEventListener('loadedmetadata', onLoadedMetadata)
    audio.addEventListener('timeupdate', onTimeUpdate)
    audio.addEventListener('pause', onPaused)
    audio.addEventListener('play', onPlay)
    audio.addEventListener('ended', onEnded)

    return () => {
      audio.removeEventListener('loadedmetadata', onLoadedMetadata)
      audio.removeEventListener('timeupdate', onTimeUpdate)
      audio.removeEventListener('pause', onPaused)
      audio.removeEventListener('play', onPlay)
      audio.removeEventListener('ended', onEnded)
      audio.pause()
    }
  }, [audio])

  useEffect(() => {
    audio.src = URL.createObjectURL(activeFile)

    return () => {
      setCurrentTime([0])
      setPlayerState(false)
    }
  }, [audio, activeFile])

  return (
    <Card overrides={{ Root: { props: { className: css({ backgroundColor: 'transparent!important' }) } } }}>
      <StyledBody>
        <h4 className={css({ textAlign: 'center', margin: 0, fontSize: '14px' })}>
          {activeFile.name}
        </h4>
        <div className={css({ display: 'flex', alignItems: 'center' })}>
          <div className={timeStyle}>
            {dayjs.duration(currentTime[0], 's').format('mm:ss')}
          </div>
          <Slider
            value={currentTime}
            onChange={({ value }) => audio.currentTime = value[0]}
            max={duration}
            overrides={{
              InnerThumb: () => null,
              ThumbValue: () => null,
              TickBar: () => null,
              Thumb: {
                style: {
                  height: '14px',
                  width: '14px',
                },
              },
              Root: {
                props: {
                  className: css({ flex: 1 }),
                },
              },
            }}
          />
          <div className={timeStyle}>
            {dayjs.duration(duration, 's').format('mm:ss')}
          </div>
        </div>
        <ButtonGroup
          size={SIZE.compact}
          shape={SHAPE.circle}
          overrides={{
            Root: {
              props: {
                className: css({ justifyContent: 'center' }),
              },
            },
          }}
        >
          <Button
            type="button"
            disabled={activeIndex === 0}
            onClick={() => play(files[activeIndex - 1])}
          >
            <IconPlayerSkipBackFilled size={16} />
          </Button>
          <Button
            type="button"
            overrides={{ BaseButton: { props: { $kind: KIND.primary } } }}
            onClick={() => {
              playerState ? audio.pause() : audio.play()
            }}
          >
            {
              playerState
                ? (
                  <IconPlayerPauseFilled size={16} />
                  )
                : (
                  <IconPlayerPlayFilled size={16} />
                  )
            }
          </Button>
          <Button
            type="button"
            disabled={activeIndex === files.length - 1}
            onClick={() => play(files[activeIndex + 1])}
          >
            <IconPlayerSkipForwardFilled size={16} />
          </Button>
        </ButtonGroup>
      </StyledBody>
      <StyledAction>
        <StyledDivider />
        {files.map((file, index) => (
          <ListItem
            key={index}
            before={(
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
            onRemove={() => remove(index)}
          />
        ))}
      </StyledAction>
    </Card>
  )
}

export default Player
