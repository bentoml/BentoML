"use strict";
const grpc = require("@grpc/grpc-js");
const pb = require("./bentoml/grpc/v1/service_pb");
const services = require("./bentoml/grpc/v1/service_grpc_pb");

function main() {
  const target = "localhost:3000";
  const client = new services.BentoServiceClient(
    target,
    grpc.credentials.createInsecure()
  );
  var ndarray = new pb.NDArray();
  ndarray
    .setDtype(pb.NDArray.DType.DTYPE_FLOAT)
    .setShapeList([1, 4])
    .setFloatValuesList([3.5, 2.4, 7.8, 5.1]);
  var req = new pb.Request();
  req.setApiName("classify").setNdarray(ndarray);

  client.call(req, function (err, resp) {
    if (err) {
      console.log(err.message);
      if (err.code === grpc.status.INVALID_ARGUMENT) {
        console.log("Invalid argument", resp);
      }
    } else {
      if (resp.getContentCase() != pb.Response.ContentCase.NDARRAY) {
        console.error("Only support NDArray response.");
      }
      console.log("result: ", resp.getNdarray().toObject());
    }
  });
}

main();
