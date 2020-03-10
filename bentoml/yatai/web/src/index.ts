import { getExpressApp } from './server'

const args = process.argv.slice(2);

const grpcServerAddress = process.env['BENTOML__YATAI_GRPC_SERVER_ADDRESS'] || args[0];
const port = process.env['BENTOML__YATAI_WEB_UI_PORT'] || args[1];

console.log(grpcServerAddress, port)
if (!grpcServerAddress || !port) {
  throw Error('Required field grpc server address or port is missing');
}

const app = getExpressApp(grpcServerAddress);

app.listen(port, () => console.log(`Running at http://localhost:${port}`))
