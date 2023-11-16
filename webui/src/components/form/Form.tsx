import type { PropsWithChildren } from 'react'
import type { Form as FormType } from '@formily/core'
import { FormProvider } from '@formily/react'

export interface FormProps {
  form: FormType
  onSubmit?: (values: any) => any
}

export default function Form({ form, onSubmit, ...props }: PropsWithChildren<FormProps>) {
  const renderContent = () => (
    <form onSubmit={onSubmit}>
      {props.children}
    </form>
  )

  return <FormProvider form={form}>{renderContent()}</FormProvider>
}
