from bentoml.grpc.v1 import service_pb2 as pb

req = pb.Request(
    api_name="classify",
    ndarray=pb.NDArray(
        dtype=pb.NDArray.DTYPE_FLOAT, shape=(1, 4), float_values=[5.9, 3, 5.1, 1.8]
    ),
)
