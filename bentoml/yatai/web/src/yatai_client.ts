import * as grpc from 'grpc';
import { bentoml } from './generated/bentoml_grpc';

export const createYataiClient = (grpcAddress: string) => {
  const client = new grpc.Client(
    grpcAddress,
    grpc.credentials.createInsecure()
  );

  const rpcImpl = function(method, requestData, callback) {
    /* Conventionally in gRPC, the request path looks like
     "/package.names.ServiceName/MethodName/",
     so getPath would generate that from the method */
     const methodPath = `/bentoml.Yatai/${method.name}`;

    client.makeUnaryRequest(
      methodPath,
      arg => arg,
      arg => arg,
      requestData,
      null,
      null,
      callback
    )
  };

  const yataiClient = bentoml.Yatai.create(rpcImpl, false, false);
  return yataiClient;
};