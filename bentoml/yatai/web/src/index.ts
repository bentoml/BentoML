import app from './server'

// Sanitize env vars
const {
  PORT: port = 3000,
} = process.env

app.listen(port, () => console.log(`Running at http://localhost:${port}`))
