import { createContext } from 'react'
import type { IAPISchema } from '../types'

export const JSONSchemaContext = createContext<IAPISchema>({} as IAPISchema)
export const JSONSchemaProvider = JSONSchemaContext.Provider
