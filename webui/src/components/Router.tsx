import { RouterProvider, createHashRouter } from 'react-router-dom'
import InferenceForm from './InferenceForm'

export default function Router() {
  const router = createHashRouter([{
    path: '*',
    element: <InferenceForm />,
  }])

  return <RouterProvider router={router} />
}
