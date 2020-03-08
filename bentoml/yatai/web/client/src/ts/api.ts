import config from './config'

export const getMessage = async () => {
  const url = config.api + 'ListBento/'

  const response = await fetch(url, {
    cache: 'no-cache',
    headers: {
      'Content-Type': 'application/json'
    },
    redirect: 'follow',
    referrerPolicy: 'no-referrer'
  })

  return await response.json();
}
