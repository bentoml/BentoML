import { proxy } from 'http-proxy-middleware'

// we need a custom proxy config since for some reason the "proxy" option in package.json
// forwards even the websocket requests that are meant for the auto reloader...
// https://github.com/facebook/create-react-app/issues/7323
module.exports = app => {
  app.use(proxy(
    '/api',
    {target: process.env.FLASK_URL || 'http://127.0.0.1:5000'}
  ));
};
