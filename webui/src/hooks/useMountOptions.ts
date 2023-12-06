import { createContext, useContext } from 'react'

export interface IMountOptions {
  needAuth?: boolean
}

const OptionContext = createContext<IMountOptions>({
  needAuth: false,
})

export function useMountOptions() {
  return useContext(OptionContext)
}

export const OptionProvider = OptionContext.Provider
