import { getExpressApp } from './server'

const args = process.argv.slice(2);

const grpcServerAddress = args[0];
const port = args[1];

if (!grpcServerAddress || !port) {
  throw Error('Required field grpc server address or port is missing');
}

const app = getExpressApp(grpcServerAddress);

app.listen(port, () => console.log(`Running at http://localhost:${port}`))
