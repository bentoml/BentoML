import { useMemo } from 'react'
import { FormControl } from 'baseui/form-control'
import type { DataType } from '../types'
import { hasFileInSchema } from '../hooks/useQuery'
import BaseCode from './code/Base'
import BasePreview, { Download as BaseDownload } from './preview/Base'
import ImagePreview, { Download as ImageDownload } from './preview/Image'
import AudioPlayer from './preview/Audio'
import VideoPlayer from './preview/Video'

interface IOutputProps {
  result: any
  schema: DataType
}

function Output({ result, schema }: IOutputProps) {
  const hasFiles = useMemo(() => hasFileInSchema({ schema }), [schema])

  if (!hasFiles) {
    if (schema.type === 'string') {
      return (
        <FormControl label={schema.title}>
          <BaseCode>{result}</BaseCode>
        </FormControl>
      )
    }
    else {
      return (
        <FormControl label={schema.title}>
          <BaseCode language="json">
            {JSON.stringify(result, null, 2)}
          </BaseCode>
        </FormControl>
      )
    }
  }
  else if (schema.type === 'file') {
    switch (schema.format) {
      case 'audio':
        return (
          <AudioPlayer files={[result]}>
            {file => <BaseDownload value={file} />}
          </AudioPlayer>
        )
      case 'video':
        return (
          <VideoPlayer files={[result]}>
            {file => <BaseDownload value={file} />}
          </VideoPlayer>
        )
      case 'image':
        return (
          <ImagePreview value={result}>
            <ImageDownload value={result} />
          </ImagePreview>
        )
      default:
        return (
          <BasePreview value={result}>
            <BaseDownload value={result} />
          </BasePreview>
        )
    }
  }
  else {
    return (
      <div style={{ color: 'red' }}>
        Error: Preview not supported for the output type
      </div>
    )
  }
}

export default Output
