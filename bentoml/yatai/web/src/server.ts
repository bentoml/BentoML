import path from 'path';
import { Request, Response } from "express";
import express from 'express'
import * as grpc from 'grpc';
import * as protoLoader from '@grpc/proto-loader';
import * as protobuf from 'protobufjs';
import { promisifyAll } from 'grpc-promise';
import { bentoml } from './compiled';

const app = express()

app.use(express.json())
app.use(
  express.static(path.join(__dirname, '../dist/client'))
)

const protoPath = path.join(__dirname, '../../../../protos/yatai_service.proto');

const packageDefinition = protoLoader.loadSync(
  protoPath,
  {
    keepCase: true,
    longs: String,
    enums: String,
    defaults: true,
    oneofs: true,
  },
);

console.log('dirname', __dirname);

const bento_proto = grpc.loadPackageDefinition(packageDefinition).bentoml;
const client = new bento_proto['Yatai'](
  'localhost:50051',
  grpc.credentials.createInsecure()
);
// Allow grpc call works with promise style
promisifyAll(client);

const root = new protobuf.Root();
const loadRoot = root.loadSync(protoPath);
const methods = loadRoot.nested.bentoml['nested']['Yatai']['methods'];

const _internalBentoMlService = ['HealthCheck', 'GetYataiServiceVersion'];
for (var i in methods) {
  if (!_internalBentoMlService.includes(i)) {
    const service = methods[i];
    const requestType = service.requestType;
    const requestMessage = bentoml[requestType];
    const serviceName = service.name.charAt(0).toLowerCase() + service.name.substr(1);
    const serviceCall = client[serviceName];

    const processRequest = async(req: Request, res: Response) => {
      const requestData = requestMessage.create(req.body);

      let result = await serviceCall().sendMessage(requestData)
        .then(response => response);
      return res.status(200).json(result);
    };

    app.get(`/api/${i}`, processRequest);
  }
}

// console.log(app._router.stack);

export default app
