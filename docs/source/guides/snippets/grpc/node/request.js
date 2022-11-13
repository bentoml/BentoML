const pb = require("./bentoml/grpc/v1/service_pb");

var ndarray = new pb.NDArray();
ndarray
  .setDtype(pb.NDArray.DType.DTYPE_FLOAT)
  .setShapeList([1, 4])
  .setFloatValuesList([3.5, 2.4, 7.8, 5.1]);
var req = new pb.Request();
req.setApiName("classify").setNdarray(ndarray);
