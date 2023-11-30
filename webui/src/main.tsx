import React from 'react'
import ReactDOM from 'react-dom/client'
import './index.css'
import { mount } from './index.tsx'

const root = document.getElementById('root')!

fetch('/schema.json')
  .then(res => res.json())
  .then(schema => mount(schema, root))
  .catch((err) => {
    ReactDOM.createRoot(root).render(
      <React.StrictMode>
        <div>
          Error:
          {err.message}
        </div>
      </React.StrictMode>,
    )
  })
