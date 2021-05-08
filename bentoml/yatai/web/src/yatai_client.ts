import { Client, credentials } from "@grpc/grpc-js";
import { bentoml } from "./generated/bentoml_grpc";

export const createYataiClient = (grpcAddress: string) => {
  const client = new Client(grpcAddress, credentials.createInsecure());

  const rpcImpl = function (method, requestData, callback) {
    /* Conventionally in gRPC, the request path looks like
     "/package.names.ServiceName/MethodName/",
     so getPath would generate that from the method */
    const methodPath = `/bentoml.Yatai/${method.name}`;

    client.makeUnaryRequest(
      methodPath,
      (arg) => arg,
      (arg) => arg,
      requestData,
      callback
    );
  };

  return bentoml.Yatai.create(rpcImpl, false, false);
};
