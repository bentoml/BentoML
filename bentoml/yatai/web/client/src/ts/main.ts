import * as api from './api'

async function fetchMessage() {
  const btn = document.getElementById('testBtn') as HTMLButtonElement

  btn.disabled = true

  api.getMessage()
    .then(response => {
      btn.textContent = response.msg
    })
    .catch(err => {
      btn.textContent = 'Check console for errors.'
      console.log(err)
    })
    .finally(() => {
      setTimeout(() => {
        btn.textContent = 'Test API'
        btn.disabled = false
      }, 1200)
    })
}

document.getElementById('testBtn').addEventListener('click', fetchMessage)
