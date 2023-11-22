export interface TBase {
  type: string
  title?: string
  default?: unknown
  description?: string
  examples?: unknown[]
}

export type TString = TBase & { type: 'string'; is_stream?: boolean }
export type TBoolean = TBase & { type: 'boolean' }

interface INumberRange {
  minimum?: number
  maximum?: number
  exclusiveMinimum?: number
  exclusiveMaximum?: number
}

export type TNumber = TBase & INumberRange & { type: 'number' }
export type TInteger = TBase & INumberRange & { type: 'integer' }
export interface TArray<T extends DataType = DataType> extends TBase {
  type: 'array'
  items: T
}
export type TTensor = TBase & { type: 'tensor' }
export type TFile = TBase & { type: 'file'; format: 'binary' }
export type TImage = TBase & { type: 'file'; format: 'image' }
export type TAudio = TBase & { type: 'file'; format: 'audio' }
export type TDataframe = TBase & { type: 'dataframe' }
export interface TObject extends TBase {
  type: 'object'
  properties?: Record<string, DataType>
  required?: string[]
}

export type DataType = TString | TBoolean | TNumber | TInteger | TArray | TTensor | TFile | TImage | TObject | TAudio | TDataframe
export interface IRoute {
  name: string
  route: string
  description?: string
  input: TObject
  output: DataType
}

export interface IAPISchema {
  name: string
  type: string
  description?: string
  routes: IRoute[]
}
