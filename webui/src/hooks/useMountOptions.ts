import { createContext, useContext } from 'react'
import type { ThemType } from './useTheme'

export const defaultOptions = {
  needAuth: false,
  header: true,
  theme: 'system' as ThemType,
}

export interface IMountOptions {
  needAuth: boolean
  header: boolean
  theme: ThemType
}

const OptionContext = createContext<IMountOptions>(defaultOptions)

export function useMountOptions() {
  return useContext(OptionContext)
}

export const OptionProvider = OptionContext.Provider
