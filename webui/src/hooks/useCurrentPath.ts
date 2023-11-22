import { useLocation, useNavigate } from 'react-router-dom'
import { useEffect, useMemo } from 'react'
import { useSchema } from './useQuery'

export default function useCurrentPath() {
  const location = useLocation()
  const navigate = useNavigate()

  const data = useSchema()

  const currentPath = location.pathname === '/' ? data?.routes[0].route : location.pathname

  useEffect(() => {
    if (location.pathname === '/' && data?.routes)
      navigate({ pathname: data?.routes[0].route })
  }, [data?.routes, location.pathname, navigate])

  const setCurrentPath = (pathname: string) => {
    navigate({ pathname })
  }

  const currentRoute = useMemo(() => {
    return data?.routes.find(route => route.route === currentPath)
  }, [currentPath, data?.routes])

  return { currentPath, setCurrentPath, currentRoute } as const
}
