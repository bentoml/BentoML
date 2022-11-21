import BentoServiceModel

var shape: [Int32] = [1, 4]
var data: [Float] = [3.5, 2.4, 7.8, 5.1]

let ndarray: Bentoml_Grpc_v1_NDArray = .with {
  $0.shape = shape
  $0.floatValues = data
  $0.dtype = Bentoml_Grpc_v1_NDArray.DType.float
}

let request: Bentoml_Grpc_v1_Request = .with {
  $0.apiName = apiName
  $0.ndarray = ndarray
}

