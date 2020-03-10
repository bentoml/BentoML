import * as grpc from 'grpc';
import { bentoml } from './generated/bentoml_grpc';

export const createYataiClient = (grpcAddress: string) => {
  const client = new grpc.Client(
    grpcAddress,
    grpc.credentials.createInsecure()
  );
  const rpcImpl = function(method, requestData, callback) {
    client.makeUnaryRequest(
      method.name,
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